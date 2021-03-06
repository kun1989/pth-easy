import torch
from torch import nn
import copy
from ..nn import Conv2d

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)

    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)

    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1, 1, 1, 1])
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def fuse_conv_bn(conv, bn):
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_conv

def fuse_conv_bn_act(conv, bn, act):
    return torch.nn.Sequential(fuse_conv_bn(conv, bn), act)

def fuse_model(module):
    mod = module
    if isinstance(module, Conv2d) and isinstance(module.norm, nn.BatchNorm2d):
        if module.activation is not None:
            mod = fuse_conv_bn_act(module.conv, module.norm, module.activation)
        else:
            mod = fuse_conv_bn(module.conv, module.norm)
        return mod

    for name, child in module.named_children():
        mod.add_module(name, fuse_model(child))
    return mod