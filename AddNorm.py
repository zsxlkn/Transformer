import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))


if __name__ == '__main__':
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))))
