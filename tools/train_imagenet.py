import argparse
import os
import torch
from torch import nn
import tqdm
import logging
import time
import shutil
from apex import amp

from pth_easy.imagenet import prepare_data_loaders, AverageMeter, LabelSmoothingCrossEntropy, NLLMultiLabelSmooth, MixUpWrapper
from pth_easy.utils import setup_logger, get_rank, synchronize, get_world_size, reduce_tensor
from pth_easy.pytorch_models import get_model
from pth_easy.solver import make_lr_scheduler, make_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='train pytorch model in imagenet.')
    parser.add_argument('--network', type=str, default='mobilenet_v1', help="network name")
    parser.add_argument('--resume', type=str, default='', help="path to latest checkpoint")
    parser.add_argument('--output-dir', type=str, default='mobilenet_v1', help='output path')
    parser.add_argument('--img-path', type=str, default='/home/ubuntu/fake_imagenet/', help='Path of the images')
    parser.add_argument('--batch-size', type=int, default=256, help='batchsize')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate if batchsize is 256')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num-epochs', type=int, default=120, help='num epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='num workers')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank')
    parser.add_argument("--logg-freq", type=int, default=100, help='logg freq')
    parser.add_argument("--label-smooth", type=bool, default=True, help='label smooth')
    parser.add_argument("--amp", type=bool, default=True, help='amp')
    parser.add_argument("--type", type=str, default='dali', help='train type base|advance(data prefetch)|dali')
    parser.add_argument('--opt_level', type=str, default='O1', help="amp opt level")
    parser.add_argument('--mixup', type=bool, default=False, help='mixup')
    args = parser.parse_args()
    return args

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(epoch, train_loader, model, loss_func, optimizer, scheduler, device, distributed, logger, cfg):

    batch_time = AverageMeter('batch_time')
    losses = AverageMeter('losses', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    iter_len = len(train_loader)

    model.train()
    loss_func.train()
    end = time.time()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        output = model(images)
        loss = loss_func(output, targets)

        optimizer.zero_grad()

        if cfg.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        if (i + 1) % cfg.logg_freq == 0:
            synchronize()
            batch_time.update((time.time() - end) / cfg.logg_freq)
            end = time.time()

            if not cfg.mixup:
                acc1 = accuracy(output, targets, topk=(1,))
                if distributed:
                    acc1 = reduce_tensor(acc1[0])
                else:
                    acc1 = acc1[0]

            if distributed:
                batch_size = get_world_size() * images.size(0)
                reduced_loss = reduce_tensor(loss.data)
            else:
                batch_size = images.size(0)
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), images.size(0))

            if not cfg.mixup:
                top1.update(acc1.item(), images.size(0))
                logger.info('Epoch: [{}][{}/{}] speed {:.0f}({:.0f}) imgs/s loss: {:.5f} top1: {:.2f} lr: {:.5f}'.format(
                    epoch, i+1, iter_len, batch_size/batch_time.val, batch_size/batch_time.avg,
                    reduced_loss, top1.avg, optimizer.param_groups[0]["lr"]))
            else:
                logger.info(
                    'Epoch: [{}][{}/{}] speed {:.0f}({:.0f}) imgs/s loss: {:.5f} lr: {:.5f}'.format(
                        epoch, i + 1, iter_len, batch_size / batch_time.val, batch_size / batch_time.avg,
                        reduced_loss, optimizer.param_groups[0]["lr"]))


def evaluate(val_loader, model, loss_func, device, distributed, logger):

    model.eval()
    loss_func.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    synchronize()

    with torch.no_grad():
        for images, targets in tqdm.tqdm(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            loss = loss_func(output, targets)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

            if distributed:
                reduced_loss = reduce_tensor(loss.data)
                acc1 = reduce_tensor(acc1[0])
                acc5 = reduce_tensor(acc5[0])
            else:
                reduced_loss = loss.data
                acc1 = acc1[0]
                acc5 = acc5[0]

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    synchronize()
    logger.info('Evaluation loss {} accuracy, top1 {} top5 {}\n'.format(losses.avg, top1.avg, top5.avg))
    return top1.avg


def main_worker(cfg, local_rank, distributed):
    logger = logging.getLogger('imagenet.trainer')
    logger.info("buid model")

    device = torch.device('cuda')
    model = get_model(cfg.network)
    model.to(device)

    num_traing_samples = 1281167 #51000 #
    num_batches = num_traing_samples // cfg.batch_size
    max_iters = num_batches * cfg.num_epochs
    warmup_iters = num_batches * cfg.warmup_epochs

    lr = cfg.learning_rate * float(cfg.batch_size) / 256.
    logger.info("base learning rate {:.5f}".format(lr))
    optimizer = make_optimizer(model, lr, cfg.weight_decay, nesterov=True)

    if cfg.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.opt_level)

    scheduler = make_lr_scheduler(optimizer, max_iters, warmup_iters)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )
        synchronize()

    train_loader, train_sampler, val_loader = prepare_data_loaders(cfg, device, is_distributed=distributed)

    if cfg.mixup:
        train_loader = MixUpWrapper(1000, train_loader, alpha=0.2)
        loss_func = NLLMultiLabelSmooth().to(device)
    elif cfg.label_smooth:
        loss_func = LabelSmoothingCrossEntropy().to(device)
    else:
        loss_func = nn.CrossEntropyLoss().to(device)

    best_acc1 = 0.0
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            logger.info("loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            best_acc1 = checkpoint['acc1']
            if distributed:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            cfg.start_epoch = checkpoint['epoch']

    logger.info("start training")
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        if distributed and cfg.type != 'dali':
           train_sampler.set_epoch(epoch)

        train(epoch, train_loader, model, loss_func, optimizer, scheduler, device, distributed, logger, cfg)
        acc1 = evaluate(val_loader, model, loss_func, device, distributed, logger)
        if get_rank() == 0:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if distributed else model.state_dict(),
                'acc1': acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_dir = '{}/checkpoint.pth'.format(cfg.output_dir)
            torch.save(state, save_dir)
            if acc1 > best_acc1:
                best_acc1 = acc1
                shutil.copyfile(save_dir, '{}/model_best.pth'.format(cfg.output_dir))
            logger.info("best acc: {:.5f}".format(best_acc1))


def main():
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    torch.backends.cudnn.benchmark = True

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    output_dir = args.output_dir
    if not os.path.exists(output_dir) and args.local_rank == 0:
        os.mkdir(output_dir)

    logger = setup_logger("imagenet", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    main_worker(args, args.local_rank, args.distributed)


if __name__ == '__main__':
    main()
