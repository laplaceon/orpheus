import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
from einops import rearrange

from .pqmf import PQMF

from .encoder import Encoder
from .decoder import Decoder, MultiBranchProbabilisticDecoder
from .decoder_predictive import PredictiveDecoder

from .mask import upsample_mask

class Orpheus(nn.Module):
    def __init__(
        self,
        enc_h_dims = [16, 96, 192, 384, 768],
        dec_h_dims = [768, 384, 192, 96, 16],
        dec_prob_dim = 256,
        dec_num_mixtures = 4,
        latent_dim = 128,
        enc_scales = [4, 4, 4, 2],
        enc_attns = [[False, False, False], [False, False, False], [False, False, False]],
        dec_scales = [2, 4, 4, 4],
        enc_ds_expansion_factor = 2.,
        dec_ds_expansion_factor = 2.,
        enc_blocks_per_stages = [1, 1, 1, 1],
        enc_layers_per_blocks = [4, 3, 3, 3],
        dec_blocks_per_stages = [1, 1, 1, 1],
        dec_layers_per_blocks = [3, 3, 3, 4],
        enc_drop_path = 0.,
        dec_drop_path = 0.,
        masked_ratio = [0.45, 0.97, 0.55, 0.25],
        fast_recompose = True
    ):
        super().__init__()

        self.pqmf = PQMF(enc_h_dims[0], 100, fast_recompose)

        # num_prob_logits = (dec_h_dims[-1] * 2 + 1) * dec_num_mixtures
        num_prob_logits = dec_h_dims[-1] * 3 * dec_num_mixtures

        dec_h_dims[-1] = num_prob_logits

        self.encoder = Encoder(enc_h_dims, latent_dim, [None] + enc_scales, enc_ds_expansion_factor, [None] + enc_attns, enc_blocks_per_stages, enc_layers_per_blocks, drop_path=enc_drop_path)
        self.decoder = Decoder(dec_h_dims, latent_dim, dec_scales, dec_ds_expansion_factor, dec_blocks_per_stages, dec_layers_per_blocks, drop_path=dec_drop_path)
        # self.decoder_mbp = MultiBranchProbabilisticDecoder(dec_h_dims[-2], dec_prob_dim)

        self.num_mixtures = dec_num_mixtures

        self.patch_size = 2048
        self.num_bands = enc_h_dims[0]
        self.mask_embedding = nn.Parameter(torch.randn(1, latent_dim, 1))

        mask_ratio_min, mask_ratio_max, mask_ratio_mu, mask_ratio_std = masked_ratio[0], masked_ratio[1], masked_ratio[2], masked_ratio[3]

        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

    def decompose(self, x):
        return self.pqmf(x)

    def encode(self, x):
        return self.encoder(x)

    def recompose(self, x):
        return self.pqmf.inverse(x)

    def decode(self, z):
        return self.decoder(z)

    def gen_random_mask(self, x):
        N = x.shape[0]
        L = x.shape[2] // self.patch_size

        mask_ratios = self.mask_ratio_generator.rvs(N)
        len_keep = (L * (1 - mask_ratios)).astype(int)

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        for i in range(len(len_keep)):
            mask[i, :len_keep[i]] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.mask_embedding.requires_grad = False
        
        for param in self.pqmf.parameters():
            param.requires_grad = False

    def sum_mix(self, y_subbands_mix):
        _, _, L = y_subbands_mix.size()
        y_weights, y_means, y_scales = self.expand_dml(y_subbands_mix)
        y_means_weighted = y_means * F.softmax(y_weights, dim=-1)
        y_subbands = torch.sum(y_means_weighted, dim=2)
        return y_weights, y_means, y_scales, y_subbands.view(-1, self.num_bands, L)

    def forward_nm(self, x):
        z = self.encode(x)
        expected, _ = self.decode(z)

        return expected, z

    def expand_dml(self, logits):
        B, _, L = logits.size()

        logits = logits.reshape(-1, self.num_mixtures * 3, L).transpose(1, 2)

        logit_probs = logits[:, :, :self.num_mixtures]
        means = logits[:, :, self.num_mixtures:2 * self.num_mixtures]
        scales = torch.clamp(logits[:, :, 2 * self.num_mixtures:3 * self.num_mixtures], min=-7.0)

        # logit_probs = logits[:, :self.num_mixtures, :]  # B, M, L
        # l = logits[:, self.num_mixtures:, :]  # B, M*C*3 , L
        # l = l.reshape(B, self.num_bands, 2 * self.num_mixtures, L)  # B, C, 3 * M, L

        # means = torch.tanh(l[:, :, :self.num_mixtures, :])  # B, C, M, L
        # scales = l[:, :, self.num_mixtures: 2 * self.num_mixtures, :]

        # l = logits.reshape(B, self.num_bands, 3 * self.num_mixtures, L)  # B, C, 3 * M, L
        # logit_probs = l[:, :, :self.num_mixtures, :]
        # means = l[:, :, self.num_mixtures: 2 * self.num_mixtures, :]  # B, C, M, L
        # weights = F.softmax(logit_probs, dim=-1)

        return logit_probs, torch.tanh(means), scales

    def forward(self, x):
        mask = self.gen_random_mask(x)
        pre_pqmf_mask = upsample_mask(mask, self.patch_size).unsqueeze(1)
        post_pqmf_mask = upsample_mask(mask, self.patch_size // self.num_bands).unsqueeze(1)

        with torch.no_grad():
            x_subbands_true = self.decompose(x)

        x = x * (1. - pre_pqmf_mask)
        x_subbands = self.decompose(x)
        x_subbands = x_subbands * (1. - post_pqmf_mask)
        z = self.encoder(x_subbands, mask)

        mask_token = self.mask_embedding.repeat(z.shape[0], 1, z.shape[2])
        z_p = z * (1. - mask.unsqueeze(1)) + mask_token * mask.unsqueeze(1)
        y_subbands, y_pre = self.decode(z_p)

        # y_probs = self.decoder_mbp(y_pre)

        return y_subbands, None, x_subbands_true, z_p, mask