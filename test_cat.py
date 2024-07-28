import torch

x = torch.randn(2, 3)
print('x-->\n', x)
y1 = torch.cat((x, x,), 0)
print('y1= torch.cat((x, x,), 0)-->\n', y1)
y2 = torch.cat((x, x,), 1)
print('y2 = torch.cat((x, x,), 1)-->\n', y2)