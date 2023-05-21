import torch

def min_max_scale(x: torch.Tensor, min: int = 0, max: int = 1):
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    size = x.size()

    x = x.reshape(size[0], -1)

    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]

    return x.view(size)