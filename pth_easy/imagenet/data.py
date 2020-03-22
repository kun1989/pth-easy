import torch
import torchvision.transforms as transforms
import torchvision

def prepare_test_data_loaders(data_path, eval_batch_size, num_workers=4,
                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    normalize = transforms.Normalize(mean, std)

    dataset = torchvision.datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=eval_batch_size, sampler=sampler, num_workers=num_workers)

    return data_loader
