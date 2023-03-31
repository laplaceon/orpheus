import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import auraloss
import torchaudio

from pprint import pprint

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

from torch_audiomentations import SomeOf, OneOf, PolarityInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter, BandPassFilter, PeakNormalization

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
skip_max = 10

n_fft = 1024
n_mels = 64

apply_augmentations = SomeOf(
    num_transforms = (1, 3),
    transforms = [
        PolarityInversion(),
        AddColoredNoise(),
        Gain(),
        OneOf(
            transforms = [
                HighPassFilter(),
                LowPassFilter()
            ]
        )
    ]
).cuda()

peak_norm = PeakNormalization(apply_to="only_too_loud_sounds", p=1., sample_rate=bitrate).cuda()

def skip_predict(x, length, future_len, skip_max=10):
    """
    Takes a tensor x of shape (N, C, L) and returns two slices of the tensor.
    The first slice is the first length timesteps of x.
    The second slice is a random skip from 1 to skip_max followed by future_len timesteps.
    """
    N, _, L = x.shape
    assert length + future_len <= L, "The sum of length and future_len cannot be greater than the sequence length."
    assert skip_max <= L - length - future_len, "skip_max must be smaller than the number of timesteps left in the sequence."
    
    # Choose a random skip between 1 and skip_max
    skip = torch.randint(1, skip_max+1, size=(N,))
    
    seqs = []
    future_seqs = []

    for i in range(N):
        seqs.append(x[i, :, :length])
        future_seqs.append(x[i, :, length+skip[i]:length+skip[i]+future_len])
    
    return torch.stack(seqs), torch.stack(future_seqs), skip.unsqueeze(1).float()

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

def train(model, prior, slicer, train_dl, neg_dl, lr=1e-4, reg_skip=32, reg_loss_cycle=32, mid_quantize_bins=512):
    opt = Adam(list(model.parameters()) + list(prior.parameters()) + list(slicer.parameters()), lr)
    stft = closs.MultiScaleSTFT([2048, 1024, 512, 256, 128], bitrate, num_mels=128)
    distance = closs.AudioDistanceV1(stft, 1e-7).cuda()

    # model.load_state_dict(torch.load("../models/rae_96.pt"))
    # opt.load_state_dict(torch.load("../models/opt_96.pt"))

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

            with torch.no_grad():
                # mod = peak_norm(apply_augmentations(real_imgs, sample_rate=bitrate))
                mod = real_imgs
                beginning, middle, skips = skip_predict(mod, sequence_length, middle_sequence_length, skip_max)
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

            z_samples_1 = prior.sample(batch_size * 64)

            d_loss = slicer.fgw_dist(z_1.transpose(1, 2).reshape(-1, z_1.shape[1]), z_samples_1)
            
            r_loss_beta, c_loss_beta = 1., 4.
            d_loss_beta = 0.
            # d_loss_beta = cyclic_kl(step, total_batch * reg_loss_cycle, maxp=1, max_beta=1e-5) if i > reg_skip-1 else 0.

            with torch.no_grad():
                f_loss = F.mse_loss(y_1, beginning)

            loss = (r_loss_beta * r_loss) + (c_loss_beta * continuity_loss) + (d_loss_beta * d_loss)

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


model = Orpheus(aug_classes=5, fast_recompose=False).cuda()
prior = GaussianPrior(128, 3).cuda()
slicer = MPSSlicer(128, 3, 50).cuda()
# model_b = BarlowTwinsVAE(model, 2).cuda()

data_folder = "../data"

# audio_files = [f"{data_folder}/2000s/{x}" for x in os.listdir(f"{data_folder}/2000s") if x.endswith(".wav")] + [f"{data_folder}/2010s/{x}" for x in os.listdir(f"{data_folder}/2010s") if x.endswith(".wav")]
audio_files = [f"{data_folder}/{x}" for x in os.listdir(f"{data_folder}") if x.endswith(".wav")][:80]

X_train, X_test = train_test_split(audio_files, train_size=0.7, random_state=42)

multiplier = 32

train_ds = AudioFileDataset(X_train, sequence_length*1+middle_sequence_length+skip_max, multiplier=multiplier)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
train_dl_neg = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_ds = AudioFileDataset(X_test, sequence_length, multiplier=multiplier)
# val_dl = DataLoader(val_ds, batch_size=ae_batch_size*2)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

train(model, prior, slicer, train_dl, train_dl_neg)
# checkpoint = torch.load("../models/ravae_stage1.pt")
# model_b.load_state_dict(checkpoint["model"])
# model = model_b.backbone
# real_eval(model, 500)
print(model)
