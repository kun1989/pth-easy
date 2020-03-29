import argparse
import torch
from gluoncv.model_zoo import get_model as get_gluon_model
from pth_easy.pytorch_gluoncv_model_zoo import get_model as get_torch_model
import numpy as np
import mxnet as mx
import os

def parse_args():
    parser = argparse.ArgumentParser(description='covert gluoncv to pytorch model.')
    parser.add_argument('--network', type=str, default='mobilenetv3_small', help="network name")
    parser.add_argument('--save-path', type=str, default='model', help='Path of the model')
    args = parser.parse_args()
    return args

def convert(name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('convert {}\n'.format(name))
    gluon_net = get_gluon_model(name, pretrained=True)
    gluon_params = gluon_net._collect_params_with_prefix()
    gluon_params_keys = list(gluon_params.keys())

    torch_net = get_torch_model(name)
    torch_params = torch_net.state_dict()
    torch_params_keys = list(torch_params.keys())
    torch_params_keys = [key for key in torch_params_keys if not key.endswith("num_batches_tracked")]

    assert (len(gluon_params_keys) >= len(torch_params_keys))

    for i, (gl_key, th_key) in enumerate(zip(gluon_params_keys, torch_params_keys)):
        t = torch_params[th_key].shape
        g = gluon_params[gl_key].shape
        assert (torch_params[th_key].shape == gluon_params[gl_key].shape)
        torch_params[th_key].data.copy_(torch.from_numpy(gluon_params[gl_key]._data[0].asnumpy()))

    torch_net.eval()
    img_data = np.random.randint(0, 255, (1, 3, 224, 224))
    img_data = img_data.astype(np.float32) / 255.0

    gl_out = gluon_net(mx.nd.array(img_data)).asnumpy()
    th_out = torch_net(torch.from_numpy(img_data)).detach().numpy()

    print('pytorch: min {}, max: {}, mean: {}'.format(th_out.min(), th_out.max(), th_out.mean()))
    print('gluoncv: min {}, max: {}, mean: {}'.format(gl_out.min(), gl_out.max(), gl_out.mean()))

    error = gl_out - th_out
    print('error: min {}, max: {}, mean: {}'.format(error.min(), error.max(), error.mean()))

    print('save {} model\n'.format(name))
    torch.save(obj=torch_net.state_dict(), f="{}/{}.pth".format(save_path, name))

if __name__ == '__main__':
    args = parse_args()
    convert(args.network, args.save_path)
