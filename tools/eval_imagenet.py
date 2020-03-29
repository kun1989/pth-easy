import argparse
import torch
from pth_easy.pytorch_gluoncv_model_zoo import get_model
from pth_easy.imagenet import prepare_test_data_loaders, AverageMeter
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate pytorch model in imagenet.')
    parser.add_argument('--network', type=str, default='mobilenetv3_small', help="network name")
    parser.add_argument('--weight_path', type=str, default='model', help='Weight of the model')
    parser.add_argument('--img-path', type=str, default='/home/xie/data/imagenet_val', help='Path of the images')
    parser.add_argument('--batch-size', type=int, default=64, help='batchsize')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--numworkers', type=int, default=4, help='numworkers')
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

def evaluate_imagenet1k(name, weight_path, img_path, batch_size, device, numworkers):
    print('eval imagenet on {}\n'.format(name))
    torch_net = get_model(name)
    weight = "{}/{}.pth".format(weight_path, name)
    print('load weights from {}\n'.format(weight))
    torch_net.load_state_dict(torch.load(weight))
    torch_net.to(torch.device(device))
    torch_net.eval()
    data_loader = prepare_test_data_loaders(img_path, batch_size, numworkers)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in tqdm.tqdm(data_loader):
            image = image.to(device)
            target = target.to(device)
            output = torch_net(image)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

    print('\nEvaluation float accuracy, top1 {} top5 {}'.format(top1.avg, top5.avg))

if __name__ == '__main__':
    args = parse_args()
    evaluate_imagenet1k(args.network, args.weight_path, args.img_path, args.batch_size, args.device, args.numworkers)
