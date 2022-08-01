import torch
from torch import nn, einsum
from functools import partial
from torch.utils.checkpoint import checkpoint

import math

from einops import rearrange

from x_transformers.x_transformers import AttentionLayers

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def window_partition(
        input: torch.Tensor,
        window_size: int = 7
) -> torch.Tensor:
    """ Window partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, L = input.shape
    # Unfold input
    windows = input.view(B, C, L // window_size, window_size)
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 3, 1).contiguous().view(-1, window_size, C)
    return windows

def window_reverse(
        windows: torch.Tensor,
        original_size: int,
        window_size: int = 7
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width
    L = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (L / window_size))
    # Fold grid tensor
    output = windows.view(B, L // window_size, window_size, -1)
    output = output.permute(0, 3, 1, 2).contiguous().view(B, -1, L)
    return output
