# NLP Example
import torch
from torch import nn
batch, sentence_length, embedding_dim = 2, 3, 5
embedding = torch.arange(0., batch*sentence_length*embedding_dim, dtype=torch.float32)\
    .reshape(batch, sentence_length, embedding_dim)
embedding[0][0][0] = 200
print(embedding)

layer_norm = nn.LayerNorm([sentence_length, embedding_dim])
print('nn.LayerNorm([sentence_length, embedding_dim])-->\n', layer_norm(embedding))

layer_norm = nn.LayerNorm(embedding_dim)
print('nn.LayerNorm(embedding_dim)-->\n', layer_norm(embedding))

batch_norm = nn.BatchNorm1d(sentence_length, affine=False)
print('batch_norm:', batch_norm(embedding))
