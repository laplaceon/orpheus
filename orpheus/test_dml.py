from core.loss import discretized_mix_logistic_loss, discretized_mix_logistic_loss_2
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
y = torch.rand(bs, l, bands) * 2 - 1
y_hat = torch.rand(bs, bands * 3 * k, l) * 2 - 1

in2 = torch.rand(bs, bands, l) * 2 - 1
out2 = torch.rand(bs, (bands * 3 + 1) * k, l) * 2 - 1

loss1 = discretized_mix_logistic_loss(y_hat, y, reduce=False)
loss2, _ = discretized_mix_logistic_loss_2(out2, in2)

print(loss1.shape, loss2.shape)
# print(loss1)
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