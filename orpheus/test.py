import torch

from model.decoder_causal import PredictiveDecoder

bs = 4
latent_dim = 128
seq_len = 64

z1 = torch.rand(bs, latent_dim, seq_len) * 2 - 1
z2 = torch.rand(bs, latent_dim, seq_len) * 2 - 1

middle_dec = PredictiveDecoder([240, 160, 160, 160, 512], latent_dim, [4, 4, 4, 4], [1, 1, 1, 1], [2, 3, 3, 3])
print(middle_dec(z1).shape)