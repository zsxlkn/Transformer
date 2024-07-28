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


class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                             num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                             num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # encoder的输出，encoder的valid length
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        # state[2]存的是过去的那些东西
        if state[2][self.i] is None: # 这时候是training的状态，
            # 如果state[2][self.i]为None，表示这是第一次进行解码，将X赋给key_values
            key_values = X
        else: # 这时候是预测阶段，将之前的解码结果与当前输入X拼接起来。
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training: # 训练阶段，dec_valid_lens用于掩码解码器输入序列中当前时间步之后的词元
            batch_size, num_steps, _ = X.shape
            dec_valid_len = torch.arange(1, num_steps+1, device=X.device,).repeat(batch_size, 1)
        else: # 如果是预测阶段,一个词一个词的生成。预测阶段，dec_valid_lens为None。
            dec_valid_len = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_len)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        z = self.addnorm2(Y, Y2)
        return self.addnorm3(z, self.ffn(z)), state


if __name__ == '__main__':
    print(123)