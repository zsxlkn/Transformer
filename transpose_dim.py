import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)
transposed_tensor = tensor.transpose(0, 1)
print(transposed_tensor)


tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor)
transposed_tensor = tensor.transpose(0, 1)
print(transposed_tensor)
