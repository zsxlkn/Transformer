import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from AddNorm import AddNorm
from PositionWiseFFN import PositionWiseFFN
from PositionWiseFFN import PositionWiseFFN
from Encoder import Encoder
from Decoder import Decoder
from EncoderBlock import EncoderBlock
from DecoderBlock import DecoderBlock


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self._attention_weights = torch.rand((2, num_layers, query_size, key_size))  # 初始化 attention weights
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block'+str(i), module=DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                                                   norm_shape, ffn_num_input, ffn_num_hiddens,
                                                                   num_heads, dropout, i, **kwargs))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.embedding(X)
        X= self.pos_encoding(X)
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
