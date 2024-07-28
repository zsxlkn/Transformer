import torch
from torch import nn
module = nn.Sequential(nn.Linear(50, 25), nn.ReLU(), nn.Linear(25, 3))
input = torch.randn((2, 25, 50))
print(module(input).shape)