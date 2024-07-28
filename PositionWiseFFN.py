import torch
from torch import nn
# 陶老师说，这是在进行特征提取

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # 对最后一个维度进行处理
        return self.dense2(self.relu(self.dense1(X)))

if __name__ == '__main__':
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    print(ffn(torch.ones((2, 3, 4)))[0])