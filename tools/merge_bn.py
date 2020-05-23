import argparse
import torch
from pth_easy.pytorch_gluoncv_model_zoo import get_model
from pth_easy.utils import fuse_model
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate pytorch model in imagenet.')
    parser.add_argument('--network', type=str, default='mobilenetv3_small', help="network name")
    parser.add_argument('--weight_path', type=str, default='models', help='Weight of the model')
    args = parser.parse_args()
    return args

def merge(name, weight_path):
    print('merge bn on {}\n'.format(name))
    torch_net = get_model(name)
    weight = "{}/{}.pth".format(weight_path, name)
    print('load weights from {}\n'.format(weight))
    torch_net.load_state_dict(torch.load(weight))
    torch_net.eval()
    img_data = np.random.randint(0, 255, (1, 3, 224, 224))
    img_data = img_data.astype(np.float32) / 255.0
    org_out = torch_net(torch.from_numpy(img_data)).detach().numpy()

    # merge_bn
    print('merge batch norm')
    torch_net = fuse_model(torch_net)

    merge_out = torch_net(torch.from_numpy(img_data)).detach().numpy()

    print('before: min {}, max: {}, mean: {}'.format(org_out.min(), org_out.max(), org_out.mean()))
    print('after: min {}, max: {}, mean: {}'.format(merge_out.min(), merge_out.max(), merge_out.mean()))

    error = merge_out - org_out
    print('error: min {}, max: {}, mean: {}'.format(error.min(), error.max(), error.mean()))

    print('save {}_mbn model\n'.format(name))
    torch.save(obj=torch_net.state_dict(), f="{}/{}_mbn.pth".format(weight_path, name))

if __name__ == '__main__':
    args = parse_args()
    merge(args.network, args.weight_path)
