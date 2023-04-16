import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import scipy.stats as stats

def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def unpatchify(x, p):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """

    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def gen_random_mask(x, mask_ratio, patch_size):
    N = x.shape[0]
    L = (x.shape[2] // patch_size) ** 2
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.randn(N, L, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask

def gen_random_mask_1d(x, mask_ratio, patch_size):
    N = x.shape[0]
    L = x.shape[2] // patch_size
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.randn(N, L, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask

def upsample_mask(mask, scale):
    assert len(mask.shape) == 2
    return mask.repeat_interleave(scale, axis=1)

def window_partition(x, p):
    """
    imgs: (N, C, L)
    x: (N, L, window_len * C)
    """
    # p = self.patch_size
    assert x.shape[2] % p == 0

    l = x.shape[2] // p
    window_dim = p * x.shape[1]
    x = x.reshape(x.shape[0], -1, window_dim)
    return x

def window_reverse(x, p):
    """
    imgs: (N, L, window_len * C)
    x: (N, C, L)
    """

    c = x.shape[2] // p
    x = x.reshape(x.shape[0], c, -1)
    return x

# N = 8
# C = 1
# L = 131072

# mask_ratio_min, mask_ratio_max, mask_ratio_mu, mask_ratio_std = 0.5, 0.96875, 0.55, 0.225
# mask_ratio_min, mask_ratio_max, mask_ratio_mu, mask_ratio_std = 0.5, 0.97, 0.55, 0.25

# mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
#                                             (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
#                                             loc=mask_ratio_mu, scale=mask_ratio_std)

# mask_ratios = mask_ratio_generator.rvs(N)
# mask = gen_random_mask_1d(torch.randn(N, C, L), mask_ratios, 2048)
# print(mask.sum(), mask.shape[0] * mask.shape[1])

# print(mask_ratios)

# x = torch.rand(8, 3, 256, 256)
# patched = patchify(x, 16)
# depatched = unpatchify(patched, 16)
# mask = gen_random_mask(x, 0.6, 16)
# print(patched.shape, mask.shape, upsample_mask(mask, 8).shape, upsample_mask(mask, 4).shape)

# f, axarr = plt.subplots(1,2)

# masked = patched * mask

# x_n = 

# axarr[0].imshow(x[0].permute(1, 2, 0))
# axarr[1].imshow(x[1].permute(1, 2, 0))
# plt.show()

# x = torch.rand(N, C, L) * 2 - 1
# x1 = window_partition(x, 256)
# x2 = window_reverse(x1, 256)
# mask = gen_random_mask_1d(x, 0.6, 2048)
# upsampled = upsample_mask_1d(mask, 2048)
# print(mask[0])

# z = torch.randn(N, 128, 64)
# print(z[0])
# z = z * mask.unsqueeze(1)
# print(z[0])
# z = torch.sigmoid(z)
# print(z[0])
# print(upsampled[0])
# print(mask.shape, upsampled.shape)
# print((x2 * mask))
# print(F.mse_loss(x, x2))