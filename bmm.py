
# torch.bmm(batch1, batch2, *, out=None) -> Tensor
# batch1：形状为 (b, n, m) 的三维张量，其中 b 是批次大小，n 和 m 分别是矩阵的行数和列数。
# batch2：形状为 (b, m, p) 的三维张量，其中 b 是批次大小，m 和 p 分别是矩阵的行数和列数。
# out：可选，结果张量
# 形状为 (b, n, p) 的张量
import torch

# 创建两个形状为 (b, n, m) 和 (b, m, p) 的三维张量
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)

# 使用 torch.bmm 进行批量矩阵乘法
result = torch.bmm(batch1, batch2)
print(result.shape)  # 输出: torch.Size([10, 3, 5])
