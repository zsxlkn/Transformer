import math
import torch
from torch import Tensor
from torch import nn
from d2l import torch as d2l
from DotProductAttention import DotProductAttention

# 就是把多个head 放在一个矩阵里 一起并行做了,一次性做完

def transpose_qkv(X: Tensor, num_heads: int):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X: Tensor, num_heads: int):
    """逆转transpose_qkv函数的操作"""
    # 输入形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
    #变为(batch_size， num_heads，查询的个数，num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # 变为(batch_size， 查询的个数，num_heads，num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 变为(batch_size， 查询的个数，num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self,queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries, keys, values = self.w_q(queries), self.w_k(keys), self.w_v(values)
        queries = transpose_qkv(queries, self.num_heads)
        keys = transpose_qkv(keys, self.num_heads)
        values = transpose_qkv(values, self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # output的形状:(batch_size*num_heads，查询的个数， num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)
if __name__ == '__main__':
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)
