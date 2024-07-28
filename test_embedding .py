import torch
from torch import nn

# 创建最大词个数为10，每个词用维度为4表示
embedding = nn.Embedding(10, 10)

# 将第一个句子填充0，与第二个句子长度对齐
in_vector = torch.LongTensor([[1, 2, 3, 4, 0, 0], [1, 2, 5, 6, 5, 7]])
out_emb = embedding(in_vector)
print(in_vector.shape)
print((out_emb.shape))
print('orignal embedding.weight\n', embedding.weight)

optimizer = torch.optim.SGD(embedding.parameters(), lr=0.01)
criteria = nn.MSELoss()
ones = torch.ones(5, 10)
for i in range(1000):
    outputs = embedding(torch.LongTensor([1, 2, 3, 4, 9]))
    loss = criteria(outputs, ones)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print('new embedding.weight\n', embedding.weight)
embedding.eval()
print(embedding(torch.LongTensor([1, 2, 3, 4, 9])))
