# tensor.permute 是 PyTorch 中用于重新排列张量（tensor）维度的函数。
# 这个函数不改变张量的数据，而只是改变张量的维度顺序
# 假设有一个形状为 (batch_size, channels, height, width) 的四维张量，通常用于图像数据。
# 使用 tensor.permute 可以改变维度的顺序，例如将其变为 (batch_size, height, width, channels)
import torch

# 创建一个形状为 (2, 3, 4, 5) 的张量
x = torch.randn(2, 3, 4, 5)

# 使用 permute 函数重新排列维度
y = x.permute(0, 2, 3, 1)

print(x.shape)  # 输出: torch.Size([2, 3, 4, 5])
print(y.shape)  # 输出: torch.Size([2, 4, 5, 3])

