import math
import torch
from torch import nn
from torch import Tensor
from masked_softmax import masked_softmax
from d2l import torch as d2l
import matplotlib.pyplot as plt
import matplotlib
from helper_function_of_heatmap import heatmap, annotate_heatmap


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 构造P矩阵，形状和位置信息矩阵一样
        self.P = torch.zeros((1, max_len, num_hiddens), dtype=torch.float32, )
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == "__main__":
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
    plt.show()
    # tensor.T 只交换最后两个维度

    X = P[0, :, :].unsqueeze(0).unsqueeze(0)
    d2l.show_heatmaps(X, xlabel='Column (encoding dimension)',
                      ylabel='Row (position)', figsize=(3.5, 4), cmap='Reds')
    plt.show()



    y = P[0, :, 4:10].T
    print(y.shape)
    x = torch.arange(num_steps)
    linestyles = ['-', '--', ':']
    for i in range(y.shape[0]):
        plt.plot(x, y[i, :], label=f'Curve {4+ i + 1}', linestyle=linestyles[i % 3])
    plt.title('Multiple Curves from Matrix')
    plt.legend(loc='upper right')
    plt.xlabel("Row (position)")
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
