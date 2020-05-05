from .resnetv1 import *
from .mobilenetv2 import *
from .mobilenetv1 import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'resnet50_v1': resnet50_v1,
    'mobilenet_v2': mobilenet_v2_1_0,
    'mobilenet_v1': mobilenet_v1_1_0,
}


def get_model(name):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name]()
    return net

def get_model_list():
    return _models.keys()
