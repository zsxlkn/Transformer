# unsqueeze 会增加一个新的单维度，使数组（张量）的形状变得更高维
import torch

# 创建一个形状为 (3, 4) 的张量
x = torch.randn(3, 4)
print("Original shape:", x.shape)
# Output: Original shape: torch.Size([3, 4])

# 在第0维度插入一个新的单维度
y = x.unsqueeze(0)
print("New shape with unsqueeze(0):", y.shape)
# Output: New shape with unsqueeze(0): torch.Size([1, 3, 4])

# 在第1维度插入一个新的单维度
z = x.unsqueeze(1)
print("New shape with unsqueeze(1):", z.shape)
# Output: New shape with unsqueeze(1): torch.Size([3, 1, 4])
