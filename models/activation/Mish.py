import torch
import torch.nn as nn
from torch.nn import functional as F



class Mish(nn.Module):

    _constants_ = ['inplace']

    def __init__(self, inplace=False):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))
