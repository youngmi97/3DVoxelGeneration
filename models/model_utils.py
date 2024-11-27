import torch
import torch.nn as nn

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Initialize tensor weights with truncated normal distribution"""
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(min=a*std, max=b*std)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output