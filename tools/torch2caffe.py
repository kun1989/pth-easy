import os
os.environ['GLOG_minloglevel'] = '2' # shut down caffe log
import caffe
import onnx
import onnxruntime
import onnxsim
import torch.onnx
import argparse
import torch
from pth_easy.pytorch_models import get_model
import pth_easy.utils as utils
from pth_easy.utils.onnx2caffe import to_caffe
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='pytorch to onnx.')
    parser.add_argument('--network', type=str, default='mobilenet_v1', help="network name")
    parser.add_argument('--weight_path', type=str, default='models/mobilenet_v1.pth', help='Weight of the model')
    parser.add_argument('--output_dir', type=str, default='caffe_ouput', help="output dir")
    args = parser.parse_args()
    return args

def to_onnx(torch_net, onnx_name, dummy_input):

    torch.onnx.export(torch_net, dummy_input, onnx_name, input_names=['input'], output_names=['output'])

    print('onnx simplifier')
    model_opt, check_ok = onnxsim.simplify(onnx_name, check_n=3, skip_fuse_bn=True, input_shapes={})
    print('onnx saved')
    onnx.save(model_opt, onnx_name)

def load_torch_net(args):
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
    torch_net = utils.fuse_model(torch_net)

    return torch_net

def main(args):
    utils.mkdir(args.output_dir)

    # load pytorch model
    torch_net = load_torch_net(args)

    # export onnx
    onnx_name = os.path.join(args.output_dir, args.network + '.onnx')
    dummy_input = torch.randn(1, 3, 224, 224)
    to_onnx(torch_net, onnx_name, dummy_input)

    deploy_proto = os.path.join(args.output_dir, args.network + '.prototxt')
    model_weights = os.path.join(args.output_dir, args.network + '.caffemodel')

    graph = to_caffe(onnx_name, deploy_proto, model_weights)

    # check ouput
    x = torch.randn(1, 3, 224, 224)
    torch_out = torch_net(x)
    ort_session = onnxruntime.InferenceSession(onnx_name)
    ort_inputs = {ort_session.get_inputs()[0].name: x.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    input_name, _ = graph.inputs[0]
    caffe_model = caffe.Net(deploy_proto, model_weights, caffe.TEST)

    caffe_model.blobs[input_name].data[...] = x
    caffe_model.forward()

    diff1 = np.abs(torch_out.detach().numpy() - ort_outs[0])

    print('torch onnx diff: min: {}, max: {}, mean: {}, var: {}'.format(diff1.min(), diff1.max(), diff1.mean(),
                                                                        np.var(diff1)))
    diff2 = np.abs(torch_out.detach().numpy()- caffe_model.blobs['output'].data[...])
    print('torch caffe diff: min: {}, max: {:.5}, mean: {:.5}, var: {:.5}'.format(diff2.min(), diff2.max(), diff2.mean(), np.var(diff2)))

if __name__ == '__main__':
    args = parse_args()
    main(args)






