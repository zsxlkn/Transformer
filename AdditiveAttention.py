import math
import torch
from torch import nn
from torch import Tensor
from masked_softmax import masked_softmax
from d2l import torch as d2l
import matplotlib.pyplot as plt
import matplotlib


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self,key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.attention_weights = None
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.droupout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor):

        queries, keys = self.w_q(queries), self.w_k(keys)
        # queries的形状(batch_size,num_queries,h)
        # keys 的形状(batch_size,num_keys,h)
        # 两者不能直接相加，中间维度不同，可以先做broadcast
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v(features) 会把h变成1
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 部分的key-value pair 不需要看
        # 返回（batch_size，查询的个数， values的维度）
        return torch.bmm(self.droupout(self.attention_weights), values)

if __name__ == '__main__':
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    print('queries-->', queries)
    print('keys-->', keys)
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    print('values-->', values)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()

    print(attention(queries, keys, values, valid_lens))
    print('attention.attention_weights-->', attention.attention_weights)
    matplotlib.use('TkAgg')
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries')
    plt.show()

