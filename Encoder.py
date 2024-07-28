import torch
from torch import nn


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, valid_lens, *args):
        raise NotImplementedError
