import torch
from torch import nn

X = torch.randn((3,5,4))
print(X)
batch_size, num_steps, _ = X.shape
dec_valid_len = torch.arange(1, num_steps+1, device=X.device,).repeat(batch_size, 1)
print(dec_valid_len)