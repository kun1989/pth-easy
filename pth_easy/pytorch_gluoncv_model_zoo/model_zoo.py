from .mobilenetv1 import *
from .mobilenetv2 import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'mobilenet1.0': mobilenet_v1_1_0,
    'mobilenet0.75': mobilenet_v1_0_75,
    'mobilenet0.5': mobilenet_v1_0_5,
    'mobilenet0.25': mobilenet_v1_0_25,
    'mobilenetv2_1.0': mobilenet_v2_1_0,
    'mobilenetv2_0.75': mobilenet_v2_0_75,
    'mobilenetv2_0.5': mobilenet_v2_0_5,
    'mobilenetv2_0.25': mobilenet_v2_0_25,
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
