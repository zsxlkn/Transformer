from torch import nn
import torch
blks = nn.Sequential()
for i in range(num_layers):
    blks.add_module(f'block {i}',module=EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                     ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False,
                                                     **kwargs))
X = self.blks(X, valid_lens)