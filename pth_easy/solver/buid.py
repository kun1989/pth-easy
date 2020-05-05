import torch
import math

def make_optimizer(model,
                   learning_rate,
                   weight_decay,
                   weight_decay_bias=0.0,
                   weight_decay_norm=0.0,
                   momentum = 0.9,
                   nesterov = False
                   ):
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params = []
    memo = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = learning_rate
            wd = weight_decay
            if isinstance(module, norm_module_types):
                wd = weight_decay_norm
            elif key == "bias":
                wd = weight_decay_bias
            params += [{"params": [value], "lr": lr, "weight_decay": wd}]

    optimizer = torch.optim.SGD(params, lr, momentum=momentum, nesterov=nesterov)
    return optimizer

def make_lr_scheduler(optimizer, max_iters, warmup_iters):
    return WarmupCosineLR(optimizer, max_iters, warmup_iters)

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iters,
        warmup_iters,
        last_epoch=-1,
    ):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            return [base_lr * alpha for base_lr in self.base_lrs]

        else:
            iter = self.last_epoch - self.warmup_iters
            max_iters = self.max_iters - self.warmup_iters
            return [
                base_lr
                * 0.5
                * (1.0 + math.cos(math.pi * iter / max_iters))
                for base_lr in self.base_lrs
            ]
