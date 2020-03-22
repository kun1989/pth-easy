import torch.nn as nn
import torch

def count_conv2d(m, input_, result_):
    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size

    out_h = result_.size(2)
    out_w = result_.size(3)

    # ops per output element
    # kernel_mul = kh * kw * cin // m.groups
    # kernel_add = kh * kw * cin // m.groups - 1
    # kernel_ops = kernel_mul + kernel_add
    # bias_ops = 1 # default merge bn
    # ops_per_element = kernel_ops + bias_ops
    ops_per_element = kh * kw * cin // m.groups
    # 1MAC = 2options www.zhihu.com/question/65305385
    # a multiply-add counts as one flop

    # total ops
    # num_out_elements = y.numel()
    output_elements = out_w * out_h * cout
    total_ops = output_elements * ops_per_element

    # in case same conv is used multiple times
    m.total_ops += total_ops

def count_fc(m, input_, result_):
    # per output element
    # total_mul = m.in_features
    # total_add = m.in_features - 1
    # bias_ops = 1
    # a multiply-add counts as one flop
    num_elements = result_.numel()
    total_ops = m.in_features * num_elements

    m.total_ops += total_ops

register_hooks = {
    nn.Conv2d: count_conv2d,
    nn.Linear: count_fc,
}


def add_hooks(m):
    if len(list(m.children())) > 0:
        return

    m.register_buffer('total_ops', torch.zeros(1))

    if type(m) in register_hooks:
        fn = register_hooks[type(m)]
        m.register_forward_hook(fn)

def count_flops(model, input_size):
    model.eval()
    model.apply(add_hooks)
    x = torch.ones(input_size)
    model(x)

    total_ops = dict()
    total_ops["conv2d"] = 0
    total_ops["fc"] = 0

    for m in model.modules():
        if len(list(m.children())) > 0:
            continue
        if type(m) == nn.Conv2d:
            total_ops["conv2d"] +=m.total_ops.item()
        elif type(m) == nn.Linear:
            total_ops["fc"] += m.total_ops.item()

    return total_ops
