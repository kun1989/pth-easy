import os
import onnx
import torch.onnx
import argparse
import torch
from pth_easy.pytorch_models import get_model
from pth_easy.utils import fuse_model
import numpy as np
import caffe

def to_numpy(tensor):
    #Converts a tensor def object to a numpy array
    tensor_dtype = tensor.data_type
    np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]
    dims = tensor.dims
    return np.frombuffer(tensor.raw_data, dtype=np_dtype).reshape(dims)

def convertAttributeProto(onnx_arg):
    if onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))

class Node(object):
    def __init__(self, name, op_type, attrs, inputs, outputs):
        self.name = name
        self.op_type = op_type
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs
        self.input_tensors = {}
        self.parents = []
        self.children = []

def parse_args():
    parser = argparse.ArgumentParser(description='pytorch to onnx.')
    parser.add_argument('--network', type=str, default='mobilenet_v2', help="network name")
    parser.add_argument('--output-dir', type=str, default='models', help='output path')
    parser.add_argument('--weight_path', type=str, default='mobilenet_v2_no_mixup_dali_200/model_best.pth', help='Weight of the model')
    args = parser.parse_args()
    return args

def main(args):
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
    torch_out = torch_net(x)

    onnx_path = os.path.join(args.output_dir, args.network + '.onnx')
    prototxt_path = os.path.join(args.output_dir, args.network + '.prototxt')
    caffemodel_path = os.path.join(args.output_dir, args.network + '.caffemodel')

    torch.onnx.export(torch_net, x, onnx_path,
                      input_names=['input'], output_names=['output'])

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

#######
    graph = onnx_model.graph

    input_tensors = {t.name: to_numpy(t) for t in graph.initializer}

    nodes = {}
    for nd_ in graph.node:
        attrs = {}
        for a in nd_.attribute:
            attrs[a.name] = convertAttributeProto(a)
        name = "_".join(nd_.output)
        node = Node(name, nd_.op_type, attrs, list(nd_.input), list(nd_.output))

        for input_ in node.inputs:
            if input_ in input_tensors:
                node.input_tensors[input_] = input_tensors[input_]
            elif input_ in nodes:
                node.parents.append(input_)
            else:
                print("{} has no node".format(input_))
        for output_ in node.outputs:
            node.children.append(output_)

        nodes[name] = node

    aa = []





########
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    #
    # diff = np.abs(to_numpy(torch_out) - ort_outs[0])
    #
    # print('   diff: min: {}, max: {}, mean: {}, var: {}'.format(diff.min(), diff.max(), diff.mean(), np.var(diff)))


if __name__ == '__main__':
    args = parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(args)
