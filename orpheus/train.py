import os

import auraloss
import torchaudio

from slugify import slugify

import math

import core.loss as closs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.rae import Orpheus
from model.prior import GaussianPrior
from model.slicer import MPSSlicer

# from model.quantizer import mu_law_encoding, mu_law_encode

from torchaudio.functional import mu_law_encoding

from dataset import AudioFileDataset
from sklearn.model_selection import train_test_split

from torch_audiomentations import SomeOf, OneOf, PolarityInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter

# from utils import copy_params
from tqdm import tqdm

from math import sqrt

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

# Hyperparams
batch_size = 4

# Params
bitrate = 44100
sequence_length = 131072
middle_sequence_length = sequence_length // 8

n_fft = 1024
n_mels = 64

# apply_augmentations = SomeOf(
#     num_transforms = (1, 3),
#     transforms = [
#         PolarityInversion(),
#         AddColoredNoise(),
#         Gain(),
#         OneOf(
#             transforms = [
#                 HighPassFilter(),
#                 LowPassFilter()
#             ]
#         )
#     ]
# ).cuda()

class BarlowTwinsVAE(nn.Module):
    def __init__(self, backbone, batch_size, lambda_coeff=5e-3):
        super().__init__()

        self.backbone = backbone

        # self.projection = nn.Flatten()

        # self.bn = nn.BatchNorm1d(8192, affine=False)

        self.batch_size = batch_size
        self.lambd = lambda_coeff

        stft = closs.MultiScaleSTFT([2048, 1024, 512, 256, 128], bitrate, num_mels=128)
        self.distance = closs.AudioDistanceV1(stft, 1e-7)

    def off_diagonal(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cr_loss(self, mu_1, std_1, mu_2, std_2, gamma=1e-2):
        """
        distance between two gaussians
        """

        cr_loss = 0.5 * torch.sum(2 * torch.log(std_1 / std_2) - \
                1 + (std_2 ** 2 + (mu_2 - mu_1) ** 2) / std_1 ** 2,
                dim=1).mean()

        return cr_loss * gamma

    def forward(self, y1, y2, bt=False):
        z1, kl1, mu1, std1 = self.backbone.reparameterize(self.backbone.encode(y1), return_vars=True)
        z2, kl2, mu2, std2 = self.backbone.reparameterize(self.backbone.encode(y2), return_vars=True)

        # z1_p = self.projection(z1)
        # z2_p = self.projection(z2)

        # # empirical cross-correlation matrix
        # c = self.bn(z1_p).T @ self.bn(z2_p)

        # # sum the cross-correlation matrix between all gpus
        # c.div_(self.batch_size)

        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # off_diag = self.off_diagonal(c).pow_(2).sum()
        # bt_loss = on_diag + self.lambd * off_diag

        bt_loss = self.cr_loss(mu1, std1, mu2, std2)

        dist_loss = kl1 + kl2

        out1 = self.backbone.decode(z1)
        out2 = self.backbone.decode(z2)

        recons_loss = self.stft_loss(out1, y1) + self.stft_loss(out2, y2)

        return (bt_loss, dist_loss, recons_loss)

def recons_loss(inp, tgt, time_weight=1.0, freq_weight=1.0, reduction="mean"):
    # lcosh = auraloss.time.LogCoshLoss(reduction=reduction)

    fft_sizes = [2048, 1024, 512, 256, 128]
    hops = [int(0.25*fft) for fft in fft_sizes]

    stft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, hop_sizes=hops, win_lengths=fft_sizes, reduction=reduction)

    with torch.no_grad():
        time_loss = F.mse_loss(inp, tgt)
    freq_loss = stft(inp, tgt)

    return (time_loss, freq_loss)

def get_song_features(model, file):
    data, rate = torchaudio.load(file)
    bal = 0.5

    if data.shape[0] == 2:
        data = bal * data[0, :] + (1 - bal) * data[1, :]
    else:
        data = data[0, :]

    consumable = data.shape[0] - (data.shape[0] % sequence_length)

    data = torch.stack(torch.split(data[:consumable], sequence_length)).cuda()
    data_spec = data[:15].unsqueeze(1)

    with torch.no_grad():
        output = model(model.decompose(data_spec))
        output = model.recompose(output).flatten()
        # print(output[:5].shape)
        return output

    # lin_out = from_mel(output)

    # with torch.no_grad():
    #     wave = to_wave(lin_out)

    #     output = wave.flatten()

    #     return output

def eval(model, val_dl):
    a = 1

def real_eval(model, epoch):
    model.eval()

    test_files = [
        "Synthwave Coolin'.wav",
        "Waiting For The End [Official Music Video] - Linkin Park-HQ.wav",
        "q1.wav"
    ]

    for test_file in test_files:
        out = get_song_features(model, f"../input/{test_file}")
        print(out)
        # count_pos_neg(out)
        torchaudio.save(f"../output/{slugify(test_file)}_epoch{epoch+1}.wav", out.cpu().unsqueeze(0), bitrate)

def count_pos_neg(x):
    pos = (x >= 0).to(torch.int16)
    num_pos = (pos == 1).sum()
    num_neg = (pos == 0).sum()
    print(x.shape, num_pos.item(), num_neg.item())

def cyclic_kl(step, cycle_len, maxp=0.5, min_beta=0, max_beta=1):
    div_shift = 1 / (1 - min_beta/max_beta)
    return min(((step % cycle_len) / (cycle_len * maxp * div_shift)) + (min_beta / max_beta), 1) * max_beta

def train(model, prior, slicer, train_dl, lr=1e-4, reg_skip=32, reg_loss_cycle=32, mid_quantize_bins=512):
    opt = Adam(list(model.parameters()) + list(prior.parameters()) + list(slicer.parameters()), lr)
    stft = closs.MultiScaleSTFT([2048, 1024, 512, 256, 128], bitrate, num_mels=128)
    distance = closs.AudioDistanceV1(stft, 1e-7).cuda()

    step = 0

    i = 0
    prior.print_parameters()
    slicer.print_parameters()
    while True:
        model.train()
        prior.train()
        slicer.train()

        nb = 0
        r_loss_total = 0
        c_loss_total = 0
        f_loss_total = 0
        d_loss_total = 0
        total_batch = len(train_dl)
        for batch in tqdm(train_dl, position=0, leave=True):
            real_imgs = batch["input"].unsqueeze(1).cuda()
            bs = real_imgs.shape[0]

            beginning = real_imgs[:, :, :sequence_length]
            middle = real_imgs[:, :, sequence_length:sequence_length+middle_sequence_length]

            with torch.no_grad():
                x_quantized_mid = mu_law_encoding(middle.squeeze(1), mid_quantize_bins)

            opt.zero_grad()

            x_subbands_1 = model.decompose(beginning)

            z_1 = model.encode(x_subbands_1)

            y_subbands_1 = model.decode(z_1)

            mb_dist_1 = distance(y_subbands_1, x_subbands_1)["spectral_distance"]

            y_1 = model.recompose(y_subbands_1)
            
            fb_dist_1 = distance(y_1, beginning)["spectral_distance"]

            r_loss = mb_dist_1 + fb_dist_1

            y_pred = model.predict_middle(z_1)

            continuity_loss = F.cross_entropy(y_pred, x_quantized_mid)

            z_samples_1 = prior.sample(bs * 64)

            d_loss = slicer.fgw_dist(z_1.transpose(1, 2).reshape(-1, z_1.shape[1]), z_samples_1)
            
            c_loss_beta = 1.
            d_loss_beta = cyclic_kl(step, total_batch * reg_loss_cycle, maxp=1, max_beta=0.02) if i > reg_skip-1 else 0.

            with torch.no_grad():
                f_loss = F.mse_loss(y_1, beginning)

            loss = r_loss + (c_loss_beta * continuity_loss) + (d_loss_beta * d_loss)

            r_loss_total += mb_dist_1.item()
            c_loss_total += continuity_loss.item()
            f_loss_total += f_loss.item()
            d_loss_total += d_loss.item()

            loss.backward()
            opt.step()

            # if torch.isnan(r_loss[0]):
            #     raise SystemError

            nb += 1
            step += 1
        # lr_scheduler.step(loss)
        print(f"D Loss: {d_loss_total/nb}, R Loss: {r_loss_total/nb}, C Loss: {c_loss_total/nb}, F Loss: {f_loss_total/nb}")

        if (i+1) % 8 == 0:
            # prior.print_parameters()
            # slicer.print_parameters()
            torch.save(model.state_dict(), f"../models/rae_{i+1}.pt")
            torch.save(prior.state_dict(), f"../models/prior_{i+1}.pt")
            torch.save(opt.state_dict(), f"../models/opt_{i+1}.pt")
            real_eval(model, i)

        i += 1


model = Orpheus(sequence_length, fast_recompose=False).cuda()
prior = GaussianPrior(128, 3).cuda()
slicer = MPSSlicer(128, 3, 50).cuda()
# model_b = BarlowTwinsVAE(model, 2).cuda()

data_folder = "../data"

# audio_files = [f"{data_folder}/2000s/{x}" for x in os.listdir(f"{data_folder}/2000s") if x.endswith(".wav")] + [f"{data_folder}/2010s/{x}" for x in os.listdir(f"{data_folder}/2010s") if x.endswith(".wav")]
audio_files = [f"{data_folder}/{x}" for x in os.listdir(f"{data_folder}") if x.endswith(".wav")][:80]

X_train, X_test = train_test_split(audio_files, train_size=0.7, random_state=42)

multiplier = 32

train_ds = AudioFileDataset(X_train, sequence_length*1+middle_sequence_length, multiplier=multiplier)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_ds = AudioFileDataset(X_test, sequence_length, multiplier=multiplier)
# val_dl = DataLoader(val_ds, batch_size=ae_batch_size*2)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

train(model, prior, slicer, train_dl)
# checkpoint = torch.load("../models/ravae_stage1.pt")
# model_b.load_state_dict(checkpoint["model"])
# model = model_b.backbone
# real_eval(model, 500)
print(model)
