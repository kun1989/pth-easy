from torch import nn
import torch.nn.functional as F

def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "bn": nn.BatchNorm2d,
            "syncbn": nn.SyncBatchNorm,
            "gn": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels)

