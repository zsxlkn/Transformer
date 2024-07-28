import torch
import torch.nn as nn
import torch.optim as optim

# # 创建一个二维张量，每行代表一个样本
# x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
#
# # 计算每个样本的 L2 范数
# l2_norms = torch.norm(x, p=2, dim=0)
# print(l2_norms)


# # 定义一个简单的模型
# model = nn.Linear(10, 1)
#
# # 定义一个优化器，使用 L2 正则化
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
#
# # 损失函数
# criterion = nn.MSELoss()
#
# # 输入和目标
# inputs = torch.randn(5, 10)
# targets = torch.randn(5, 1)
#
# # 前向传播
# outputs = model(inputs)
# loss = criterion(outputs, targets)
#
# # 反向传播和优化
# loss.backward()
# optimizer.step()
import torch

# 创建一个张量
x = torch.randn(100)

# 计算 L2 范数
l2_norm = torch.norm(x, p=2)

# 将张量的 L2 范数缩放为 1
normalized_x = x / l2_norm

# 检查归一化后的张量的 L2 范数是否为 1
new_l2_norm = torch.norm(normalized_x, p=2)

print("Original Tensor:", x)
print("L2 Norm of Original Tensor:", l2_norm)
print("Normalized Tensor:", normalized_x)
print("L2 Norm of Normalized Tensor:", new_l2_norm)




