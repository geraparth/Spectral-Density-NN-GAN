import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0, bottleneck=False, norm=True):
        super(Block, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch if not bottleneck else out_ch//2
        self.num_classes = num_classes
        self.norm = norm

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.norm:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if self.norm:
            if y is not None:
                h = self.b1(x, y, **kwargs)
            else:
                h = self.b1(x)
        else:
            h = x
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if self.norm:
            if y is not None:
                h = self.b2(h, y, **kwargs)
            else:
                h = self.b2(h)
        return self.c2(self.activation(h))