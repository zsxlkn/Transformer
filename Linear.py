import torch
from torch import nn


m = nn.Linear(20, 30)
print(torch.range(0.1, 1, 0.1).reshape(-1, 5))
input = torch.randn((64, 128, 20))
output = m(input)
print(output.size())