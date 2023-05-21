from core.loss import discretized_mix_logistic_loss, discretized_mix_logistic_loss_2, discretized_mix_logistic_loss_pp
from model.pqmf import PQMF

import torch
import numpy as np

from tqdm import tqdm

bs = 2
l = 8192
bands = 16

k = 4
pqmf = PQMF(bands, 100, True)

min = np.Inf
max = 0

# https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py

# for _ in tqdm(range(1000)):
target = torch.rand(bs, l, bands) * 2 - 1
input = torch.rand(bs, l, (bands * 3 + 1) * k) * 2 - 1

# in2 = torch.rand(bs, bands, l) * 2 - 1
# out2 = torch.rand(bs, (bands * 3 + 1) * k, l) * 2 - 1

loss1, ex = discretized_mix_logistic_loss_pp(target, input, k=k)
loss2, _ = discretized_mix_logistic_loss_2(input.transpose(1, 2), target.transpose(1, 2))

print(loss1.shape, loss2.shape, ex.shape)
# print(loss2)
print(torch.sum(loss1), torch.sum(loss2))

#     y_mb = pqmf.inverse(y.transpose(1, 2))
#     _min = y_mb.min().item()
#     _max = y_mb.max().item()

#     print(y.min().item(), _min, _max)

#     if min > _min:
#         min = _min
    
#     if max < _max:
#         max = _max
# print(min, max)