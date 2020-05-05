import argparse
import torch
from pth_easy.pytorch_models import get_model
from pth_easy.imagenet import prepare_test_data_loaders, AverageMeter
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate pytorch model in imagenet.')
    parser.add_argument('--network', type=str, default='mobilenet_v1', help="network name")
    parser.add_argument('--weight-path', type=str, default='mobilenet_v1/model_best.pth', help='Weight of the model')
    parser.add_argument('--img-path', type=str, default='/home/ubuntu/fake_imagenet/val', help='Path of the images')
    parser.add_argument('--batch-size', type=int, default=128, help='batchsize')
    parser.add_argument('--num-workers', type=int, default=4, help='num workers')
    parser.add_argument("--type", type=str, default='dali', help='train type base|advance(data prefetch)|dali')
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

def evaluate_imagenet1k(args):
    device = torch.device('cuda')
    print('eval imagenet on {}\n'.format(args.network))
    torch_net = get_model(args.network)

    print('load weights from {}\n'.format(args.weight_path))
    file = torch.load(args.weight_path)
    if 'state_dict' in file:
        torch_net.load_state_dict(file['state_dict'])
    else:
        torch_net.load_state_dict(file)
    torch_net.to(device)
    torch_net.eval()


    data_loader = prepare_test_data_loaders(args, args.img_path, args.batch_size, args.num_workers, device)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for images, target in tqdm.tqdm(data_loader):
            images = images.to(device)
            target = target.to(device)
            output = torch_net(images)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc1 = acc1[0]
            acc5 = acc5[0]

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    print('\nEvaluation float accuracy, top1 {} top5 {}'.format(top1.avg, top5.avg))

if __name__ == '__main__':
    args = parse_args()

    evaluate_imagenet1k(args)
