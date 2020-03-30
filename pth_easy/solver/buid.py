from typing import Any, Dict, List, Set
import torch

def build_optimizer(model, learning_rate, weight_decay, weight_decay_norm=0.0, momentum = 0.9):
    """
    Build an optimizer from config.
    """
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
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
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
                # NOTE: we now default BIAS_LR_FACTOR to 1.0 and WEIGHT_DECAY_BIAS to WEIGHT_DECAY
                # so that bias optimizer hyperparameters are by default exactly the same
                # as for regular weights.
                lr = learning_rate * 1.0
                wd = weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer
