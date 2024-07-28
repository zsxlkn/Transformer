import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_output, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
