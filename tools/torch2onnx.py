import onnx
import onnxruntime
import torch.onnx
import argparse
import torch
from pth_easy.pytorch_models import get_model
from pth_easy.utils import fuse_model
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='pytorch to onnx.')
    parser.add_argument('--network', type=str, default='mobilenet_v1', help="network name")
    parser.add_argument('--weight_path', type=str, default='mobilenet_v1/model_best.pth', help='Weight of the model')
    args = parser.parse_args()
    return args

def to_onnx(args):
    print('network {}\n'.format(args.network))
    torch_net = get_model(args.network)

    print('load weights from {}\n'.format(args.weight_path))
    file = torch.load(args.weight_path)
    if 'state_dict' in file:
        torch_net.load_state_dict(file['state_dict'])
    else:
        torch_net.load_state_dict(file)
    torch_net.eval()

    # merge_bn
    print('merge batch norm')
    torch_net = fuse_model(torch_net)

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = torch_net.conv1(x)

    torch.onnx.export(torch_net.conv1, x, 'test.onnx',
                      input_names=['input'], output_names=['output'],
                      )

    onnx_model = onnx.load("test.onnx")
    onnx.checker.check_model(onnx_model)


    ort_session = onnxruntime.InferenceSession("test.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    diff = np.abs(to_numpy(torch_out) - ort_outs[0])

    print('   diff: min: {}, max: {}, mean: {}, var: {}'.format(diff.min(), diff.max(), diff.mean(), np.var(diff)))


if __name__ == '__main__':
    args = parse_args()

    to_onnx(args)
