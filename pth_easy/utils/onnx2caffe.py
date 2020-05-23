import caffe
from caffe import layers as L, params as P
import onnx
from onnx import numpy_helper
import numpy as np


class Graph(object):
    def __init__(self, nodes, inputs, outputs):
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs
    def __repr__(self):
        return self.__class__.__name__

class Node(object):
    def __init__(self, name, type, attrs, inputs, outputs, params):
        self.name = name
        self.type = type
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'type={}, '.format(self.type)
        s += 'inputs={}, '.format(self.inputs)
        s += 'outputs={}, '.format(self.outputs)
        s += 'attrs={}, '.format(self.attrs)
        return s

def convert_attribute(onnx_attr):
    if onnx_attr.HasField('f'):
        return onnx_attr.f
    elif onnx_attr.HasField('i'):
        return onnx_attr.i
    elif onnx_attr.HasField('s'):
        return onnx_attr.s
    elif onnx_attr.HasField('t'):
        return numpy_helper.to_array(onnx_attr.t)
    elif len(onnx_attr.floats):
        return list(onnx_attr.floats)
    elif len(onnx_attr.ints):
        return list(onnx_attr.ints)
    elif len(onnx_attr.strings):
        return list(onnx_attr.strings)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_attr))

def get_attribute(onnx_attributes):
    attributes = {}
    for onnx_attr in onnx_attributes:
        attributes[onnx_attr.name] = convert_attribute(onnx_attr)
    return attributes

def build_node(onnx_node, all_params):
    attributes = get_attribute(onnx_node.attribute)
    name = "".join(onnx_node.output)

    inputs = []
    params = {}
    for input_ in list(onnx_node.input):
        if input_ in all_params:
            params[input_] = all_params[input_]
        else:
            inputs.append(input_)

    return Node(name, onnx_node.op_type, attributes, inputs, list(onnx_node.output), params)

def build_graph(graph):
    params = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    nodes = {}
    for onnx_node in graph.node:
        node = build_node(onnx_node, params)
        nodes[node.name] = node

    inputs = []
    for i in graph.input:
        input_name = i.name
        input_shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
        inputs.append((input_name, input_shape))

    outputs = []
    for o in graph.output:
        output_name = o.name
        output_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        outputs.append((output_name, output_shape))

    return Graph(nodes, inputs, outputs)

def build_conv(net, node):
    attrs = node.attrs
    dilations = attrs['dilations']
    group = attrs['group']
    kernel_shape = attrs['kernel_shape']
    pads = attrs['pads']
    strides = attrs['strides']
    bias_term = True

    weight = None
    bias = None
    for name, param in node.params.items():
        if 'weight' in name:
            weight = param
        else:
            bias = param
    if bias is None:
        bias_term = False
    num_output = weight.shape[0]

    net[node.name] = L.Convolution(bottom=node.inputs[0],
                                   num_output=num_output ,
                                   kernel_h=kernel_shape[0],
                                   kernel_w=kernel_shape[1],
                                   stride_h=strides[0],
                                   stride_w=strides[1],
                                   pad_h=pads[0],
                                   pad_w=pads[1],
                                   dilation=dilations[0],
                                   bias_term=bias_term,
                                   group=group)

def build_relu(net, node):
    if node.name == node.inputs[0]:
        inplace = True
    else:
        inplace = False
    net[node.name] = L.ReLU(net[node.inputs[0]], in_place=inplace)

def build_maxpool(net, node):
    attrs = node.attrs
    kernel_shape = attrs['kernel_shape']
    pads = attrs['pads']
    strides = attrs['strides']
    net[node.name] = L.Pooling(net[node.inputs[0]],
                               pool=P.Pooling.MAX,
                               kernel_h=kernel_shape[0],
                               kernel_w=kernel_shape[1],
                               stride_h=strides[0],
                               stride_w=strides[1],
                               pad_h=pads[0],
                               pad_w=pads[1],
                               round_mode=P.Pooling.FLOOR
                               )

def build_avgpool(net, node):
    attrs = node.attrs
    kernel_shape = attrs['kernel_shape']
    pads = attrs['pads']
    strides = attrs['strides']
    net[node.name] = L.Pooling(net[node.inputs[0]],
                               pool=P.Pooling.AVE,
                               kernel_h=kernel_shape[0],
                               kernel_w=kernel_shape[1],
                               stride_h=strides[0],
                               stride_w=strides[1],
                               pad_h=pads[0],
                               pad_w=pads[1],
                               round_mode=P.Pooling.FLOOR
                               )

def build_add(net, node):
    net[node.name] = L.Eltwise(net[node.inputs[0]], net[node.inputs[1]], operation=P.Eltwise.SUM)

def build_interp(net, node):
    params = None
    for key, value in node.params.items():
        if len(value):
            params = value
    net[node.name] = L.Interp(net[node.inputs[0]], height=params[2], width=params[3], align_corners=False)

def build_flatten(net, node):
    net[node.name] = L.Flatten(net[node.inputs[0]])

def build_reshape(net, node):
    param = []
    for key, value in node.params.items():
        param = value
    if param[0] == 1 and param[1] == -1:
        net[node.name] = L.Flatten(net[node.inputs[0]])
    else:
        raise ValueError("Unsupported reshape: {}".format(param))

def build_InnerProduct(net, node):
    bias_term = True

    weight = None
    bias = None
    for name, param in node.params.items():
        if 'weight' in name:
            weight = param
        else:
            bias = param
    if bias is None:
        bias_term = False
    num_output = weight.shape[0]

    net[node.name] = L.InnerProduct(bottom=node.inputs[0], num_output=num_output, bias_term=bias_term,)

ONNX_NODE_REGISTRY = {
    "Conv": build_conv,
    "Relu": build_relu,
    "MaxPool": build_maxpool,
    'AveragePool':build_avgpool,
    "Add": build_add,
    "Resize": build_interp,
    "Flatten":build_flatten,
    "Reshape":build_reshape,
    'Gemm':build_InnerProduct,
}

def build_deploy(graph, proto):
    net = caffe.NetSpec()
    for name, node in graph.nodes.items():
        converter_fn = ONNX_NODE_REGISTRY[node.type]
        converter_fn(net, node)

    # single input
    input_name, input_shape = graph.inputs[0]
    with open(proto, 'w') as f:
        f.write("input:\"{}\"\n".format(input_name))
        f.write('input_dim:{}\n'.format(input_shape[0]))
        f.write('input_dim:{}\n'.format(input_shape[1]))
        f.write('input_dim:{}\n'.format(input_shape[2]))
        f.write('input_dim:{}\n'.format(input_shape[3]))
        f.write(str(net.to_proto()))

def load_params(caffe_params, nodes):
    for name, param in caffe_params.items():
        node = nodes[name]
        if node.type == "Conv" or node.type == "Gemm":
            onnx_weight = None
            onnx_bias = None
            for key, value in node.params.items():
                if 'weight' in key:
                    onnx_weight = value
                else:
                    onnx_bias = value
            param[0].data[...] = onnx_weight
            if onnx_bias is not None:
                param[1].data[...] = onnx_bias
        else:
            raise ValueError("Unsupported type: {}".format(node.type))

def to_caffe(onnx_name, deploy_proto, model_weights):
    onnx_model = onnx.load(onnx_name)
    graph = build_graph(onnx_model.graph)

    build_deploy(graph, deploy_proto)
    caffe_model = caffe.Net(deploy_proto, caffe.TEST)

    load_params(caffe_model.params, graph.nodes)
    caffe_model.save(model_weights)

    return graph

