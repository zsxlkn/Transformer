import torch
x = torch.tensor([1, 2, 3])
print(x)
print(x.repeat_interleave(2), '\n')
y = torch.tensor([[1, 2], [3, 4]])
print(torch.repeat_interleave(y, 2))
print(torch.repeat_interleave(y, 3, dim=1))
print(torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0))
print(torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0, output_size=3))