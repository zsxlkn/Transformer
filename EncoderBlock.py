import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from AddNorm import AddNorm
from PositionWiseFFN import PositionWiseFFN


class EncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, bias=use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.attention(X, X, X, valid_lens)
        Y = self.addnorm1(X, Y)
        return self.addnorm2(Y, self.ffn(Y))

if __name__ == '__main__':
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print(encoder_blk(X, valid_lens).shape)
