# 遮蔽softmax操作，结果是前面valid_lens个和为一
import torch
from torch import nn


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.   irrelevant：不相关的`"""
    maxlen = X.size(dim=1)  # 获取张量 X 在第二个维度的大小，通常是序列的最大长度
    print("maxlen-->", maxlen)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)
    mask = mask[None, :] < valid_len[:, None]  # 使用None的位置新增一个维度
    X[~mask] = value
    return X


def masked_softmax(X: torch.Tensor, valid_lens: torch.Tensor):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    print('original valid_lens-->', valid_lens)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        print('original valid_lens.dim()-->', valid_lens.dim(), 'original valid_lens.shape-->', valid_lens.shape)
        shape = X.shape  # 保存输入X的形状
        if valid_lens.dim() == 1:  # valid_lens是一维的数字
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            # 如果valid_lens是一维的，就沿着input的第二个维度重复，成为一维向量
        else:
            valid_lens = valid_lens.reshape(-1)
            # 如果valid_lens是二维的，直接展开为一维向量
        print('final valid_lens=', valid_lens)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


if __name__ == '__main__':
    X = torch.rand(2, 3, 5)  # 2个2*4的二维矩阵
    print('X.size()-->', X.size())
    # print('X.shape-->', X.shape)
    print('original X-->', X)
    print(masked_softmax(X, torch.tensor([2, 3])))
    # print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))
