import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from AddNorm import AddNorm
from PositionWiseFFN import PositionWiseFFN
from Encoder import Encoder
from Decoder import Decoder
from EncoderBlock import EncoderBlock


class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.attention_weights = None
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'block {i}',
                                 module=EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                     ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False,
                                                     **kwargs))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X)
        self.attention_weights = [None] *len(self.blks)
        # X = self.blks(X, valid_lens) 这样的调用是错误的
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


if __name__ == '__main__':
    valid_lens = torch.tensor([3, 2])
    encoder = TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()

    var = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)
    print(var.shape)


