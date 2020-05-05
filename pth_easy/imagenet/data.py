import os
import torch
import torchvision.transforms as transforms
import torchvision
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from pth_easy.utils import get_world_size, get_rank
import numpy as np

from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def prepare_data_loaders_base(cfg, is_distributed=False):
    num_gpus = get_world_size()
    batch_size = cfg.batch_size
    assert (
        batch_size % num_gpus == 0
    ), "batch_size ({}) must be divisible by number of GPUs ({})".format(batch_size, num_gpus)
    img_per_gpu = batch_size // num_gpus

    train_data_dir = os.path.join(cfg.img_path, 'train')
    val_data_dir = os.path.join(cfg.img_path, 'val')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform_train)
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=img_per_gpu,
                                               num_workers=cfg.num_workers,
                                               sampler=train_sampler
                                               )

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_dataset = torchvision.datasets.ImageFolder(root=val_data_dir, transform=transform_val)
    if is_distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=img_per_gpu,
                                               num_workers=cfg.num_workers,
                                               sampler=val_sampler
                                               )

    return train_loader, train_sampler, val_loader

class PrefetchedWrapper(object):
    def prefetched_loader(loader, device):

        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).to(device).view(1, 3, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).to(device).view(1, 3, 1, 1)
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.to(device, non_blocking=True)
                next_target = next_target.to(device, non_blocking=True)
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return PrefetchedWrapper.prefetched_loader(self.dataloader, self.device)

def fast_collate(batch):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

def prepare_data_loaders_advance(cfg, device, is_distributed=False):
    num_gpus = get_world_size()
    batch_size = cfg.batch_size
    assert (
        batch_size % num_gpus == 0
    ), "batch_size ({}) must be divisible by number of GPUs ({})".format(batch_size, num_gpus)
    img_per_gpu = batch_size // num_gpus

    collate_fn = lambda b: fast_collate(b)

    train_data_dir = os.path.join(cfg.img_path, 'train')
    val_data_dir = os.path.join(cfg.img_path, 'val')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomHorizontalFlip()
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform_train)
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=img_per_gpu,
                                               num_workers=cfg.num_workers,
                                               sampler=train_sampler,
                                               collate_fn=collate_fn,
                                               pin_memory=True
                                               )

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    val_dataset = torchvision.datasets.ImageFolder(root=val_data_dir, transform=transform_val)
    if is_distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=img_per_gpu,
                                             num_workers=cfg.num_workers,
                                             sampler=val_sampler,
                                             collate_fn=collate_fn,
                                             pin_memory=True
                                               )
    train_loader = PrefetchedWrapper(train_loader, device)
    val_loader = PrefetchedWrapper(val_loader, device)
    return train_loader, train_sampler, val_loader

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)

        rank = device_id
        world_size = get_world_size()

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = rank,
                num_shards = world_size,
                random_shuffle = True)

        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512)

        self.res = ops.RandomResizedCrop(
                device="gpu",
                size=[224, 224],
                interp_type=types.INTERP_LINEAR,
                random_aspect_ratio=[0.75, 4./3.],
                random_area=[0.08, 1.0],
                num_attempts=100)

        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (224, 224),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                                            std = [0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        rank = device_id
        world_size = get_world_size()

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = rank,
                num_shards = world_size,
                random_shuffle = False)

        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.Resize(device = "gpu", resize_shorter = 256)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                output_dtype = types.FLOAT,
                output_layout = types.NCHW,
                crop = (224, 224),
                image_type = types.RGB,
                mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                std = [0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

class DALIWrapper(object):
    def gen_wrapper(dalipipeline, device):
        for data in dalipipeline:
            input = data[0]["data"]
            target = torch.reshape(data[0]["label"], [-1]).to(device).long()
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline, device, len):
        self.dalipipeline = dalipipeline
        self.device = device
        self.len = len

    def __len__(self):
        return self.len

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline, self.device)

def prepare_data_loaders_dali(cfg, device, is_distributed=False):
    num_gpus = get_world_size()
    rank = get_rank()
    batch_size = cfg.batch_size
    assert (
        batch_size % num_gpus == 0
    ), "batch_size ({}) must be divisible by number of GPUs ({})".format(batch_size, num_gpus)
    img_per_gpu = batch_size // num_gpus

    train_data_dir = os.path.join(cfg.img_path, 'train')
    val_data_dir = os.path.join(cfg.img_path, 'val')

    pipe_train = HybridTrainPipe(batch_size=img_per_gpu,
                           num_threads=cfg.num_workers,
                           device_id=rank,
                           data_dir=train_data_dir)

    pipe_train.build()
    train_loader = DALIClassificationIterator(pipe_train, size=int(pipe_train.epoch_size("Reader") / num_gpus))

    train_loader = DALIWrapper(train_loader, device, int(pipe_train.epoch_size("Reader") / batch_size))

    pipe_val = HybridValPipe(batch_size=img_per_gpu,
                                 num_threads=cfg.num_workers,
                                 device_id=rank,
                                 data_dir=val_data_dir)
    pipe_val.build()
    val_loader = DALIClassificationIterator(pipe_val, size=int(pipe_val.epoch_size("Reader") / num_gpus))
    val_loader = DALIWrapper(val_loader, device, int(pipe_val.epoch_size("Reader") / batch_size))

    # transform_val = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    # ])
    # collate_fn = lambda b: fast_collate(b)
    # val_dataset = torchvision.datasets.ImageFolder(root=val_data_dir, transform=transform_val)
    # if is_distributed:
    #     val_sampler = DistributedSampler(val_dataset, shuffle=False)
    # else:
    #     val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=img_per_gpu,
    #                                          num_workers=cfg.num_workers,
    #                                          sampler=val_sampler,
    #                                          collate_fn=collate_fn,
    #                                          pin_memory=True
    #                                          )
    # val_loader = PrefetchedWrapper(val_loader, device)

    return train_loader, None, val_loader

def prepare_data_loaders(cfg, device, is_distributed=False):
    if cfg.type == 'base':
        return prepare_data_loaders_base(cfg, is_distributed)
    elif cfg.type == 'advance':
        return prepare_data_loaders_advance(cfg, device, is_distributed)
    elif cfg.type == 'dali':
        return prepare_data_loaders_dali(cfg, device, is_distributed)
    else:
        pass

def prepare_test_data_loaders_base(data_path, batch_size, num_workers=4):
    img_per_gpu = batch_size

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=img_per_gpu, sampler=sampler, num_workers=num_workers)

    return data_loader

def prepare_test_data_loaders_advance(data_path, batch_size, num_workers, device):
    img_per_gpu = batch_size
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    val_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_val)
    val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)
    collate_fn = lambda b: fast_collate(b)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=img_per_gpu,
                                             num_workers=num_workers,
                                             sampler=val_sampler,
                                             collate_fn=collate_fn,
                                             pin_memory=True
                                             )
    val_loader = PrefetchedWrapper(val_loader, device)

    return val_loader

def prepare_test_data_loaders_dali(data_path, batch_size, num_workers, device):
    img_per_gpu = batch_size

    pipe_val = HybridValPipe(batch_size=img_per_gpu,
                             num_threads=num_workers,
                             device_id=0,
                             data_dir=data_path)
    pipe_val.build()
    val_loader = DALIClassificationIterator(pipe_val, size=pipe_val.epoch_size("Reader"))
    val_loader = DALIWrapper(val_loader, device, int(pipe_val.epoch_size("Reader") / batch_size))

    return val_loader

def prepare_test_data_loaders(cfg, img_path, batch_size, num_workers, device):
    if cfg.type == 'base':
        return prepare_test_data_loaders_base(img_path, batch_size, num_workers)
    elif cfg.type == 'advance':
        return prepare_test_data_loaders_advance(img_path, batch_size, num_workers, device)
    elif cfg.type == 'dali':
        return prepare_test_data_loaders_dali(img_path, batch_size, num_workers, device)
    else:
        pass
