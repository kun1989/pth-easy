import torch
import torch.nn as nn
import numpy as np

def mixup(alpha, num_classes, data, target):
    with torch.no_grad():
        bs = data.size(0)
        c = np.random.beta(alpha, alpha)

        perm = torch.randperm(bs).cuda()

        md = c * data + (1 - c) * data[perm, :]
        mt = c * target + (1 - c) * target[perm, :]
        return md, mt

class MixUpWrapper(object):
    def __init__(self, num_classes, dataloader, alpha=0.2):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes

    def mixup_loader(self, loader):
        for input, target in loader:
            target = torch.nn.functional.one_hot(target, self.num_classes)
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)
