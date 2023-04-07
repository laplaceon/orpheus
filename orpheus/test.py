import torch
from torch import nn

avgpool = nn.AdaptiveAvgPool1d(1)

x = torch.rand(4, 80, 8192)
print("shape", avgpool(x).shape)