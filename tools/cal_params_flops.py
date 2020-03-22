import argparse
from pth_easy.pytorch_gluoncv_model_zoo import get_model
from pth_easy.utils import params_count, count_flops
import torch
from fvcore.nn.flop_count import flop_count
# from pth_easy.utils import fuse_model

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate pytorch model in imagenet.')
    parser.add_argument('--network', type=str, default='mobilenet0.5', help="network name")
    args = parser.parse_args()
    return args

def calc(name):
    print('merge bn on {}\n'.format(name))
    torch_net = get_model(name)
    # merge bn
    # torch_net = fuse_model(torch_net)
    print('params: {}'.format(params_count(torch_net)))
    flops = count_flops(torch_net, input_size=(1, 3, 224, 224))
    print('Mflops: conv2d {} fc {}'.format(flops['conv2d']/1e6, flops['fc']/1e6))

    # compare to fvcore
    flop_dict1, _ = flop_count(torch_net, (torch.ones((1, 3, 224, 224)),))
    print(flop_dict1)

if __name__ == '__main__':
    args = parse_args()
    calc(args.network)