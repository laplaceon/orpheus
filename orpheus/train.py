import os

import auraloss
import torchaudio

from slugify import slugify

import math

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from orpheus.model.rae import RawAudioEncoder

from dataset import AudioFileDataset
from sklearn.model_selection import train_test_split

# from utils import copy_params
from tqdm import tqdm

from math import sqrt

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

# Hyperparams
batch_size = 2

# Params
bitrate = 44100
sequence_length = 177984

def recons_loss(inp, tgt, time_weight=1.0, freq_weight=1.0, reduction="mean"):
    lcosh = auraloss.time.LogCoshLoss(reduction=reduction)

    fft_sizes = [2048, 1024, 512, 256, 128, 64]
    hops = [int(0.25*fft) for fft in fft_sizes]

    stft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, hop_sizes=hops, win_lengths=fft_sizes, reduction=reduction)

    time_loss = lcosh(inp, tgt)
    freq_loss = stft(inp, tgt)

    return time_weight * time_loss + freq_weight * freq_loss

def get_song_features(model, file):
    data, rate = torchaudio.load(file)
    bal = 0.5

    if data.shape[0] == 2:
        data = bal * data[0, :] + (1 - bal) * data[1, :]
    else:
        data = data[0, :]

    consumable = data.shape[0] - (data.shape[0] % sequence_length)

    data = torch.stack(torch.split(data[:consumable], sequence_length)).unsqueeze(1).cuda()

    with torch.no_grad():
        z = model.encode(data, True)[0]
        output = model.decode(z)

        output = output.squeeze(1).flatten()

        return output

def train(model, train_dl, lr=1e-3, beta_kl=1e-7, beta_rec=1.0, beta_kld_max=0.15):
    opt = Adam(model.parameters(), lr)
    lr_scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=True)

    step = 0

    i = 0
    while True:
        model.train()

        nb = 0
        r_loss_total = 0
        d_loss_total = 0
        for batch_idx, batch in tqdm(enumerate(train_dl)):
            real_imgs = batch["input"].cuda().unsqueeze(1)
            bs = real_imgs.shape[0]

            opt.zero_grad()

            temperature = 1.0 * math.exp(-3e-4 * step)
            # print("temperature", temperature)

            # Train Encoder, Decoder
            z_pre, z, kl, perplexity = model.encode(real_imgs, False, temperature)
            rec = model.decode(z)
            # mel_rec = model.decode_mel(z_pre)

            mse = recons_loss(rec, real_imgs, time_weight=20.0)

            # mse = F.mse_loss(rec, real_imgs, reduction="sum") / bs
            r_loss = 1024 * torch.log(mse)

            beta_kld = min(beta_kld_max, beta_kld_max * step / 16800)

            loss = r_loss + beta_kld * torch.abs(kl)

            # print(r_loss, d_loss)
            r_loss_total += r_loss.item()
            d_loss_total += kl.item()

            loss.backward()
            opt.step()
            step += 1

            if torch.isnan(r_loss):
                raise SystemError

            nb += 1
        lr_scheduler.step(loss)

        if temperature == 0:
            break
        # while beta_kl <= 1:
        #     beta_kl *= sqrt(10)
        # scheduler steps
        print(f"D Loss: {d_loss_total/nb}, R Loss: {r_loss_total/nb}")

        if (i+1) % 3 == 0:
            model.eval()

            test_files = [
                "Synthwave Coolin'.wav",
                "Waiting For The End [Official Music Video] - Linkin Park-HQ.wav"
            ]

            for test_file in test_files:
                out = get_song_features(model, f"../output/{test_file}")
                torchaudio.save(f"../output/{slugify(test_file)}_epoch{i+1}.wav", out.cpu().unsqueeze(0), bitrate)
        i += 1


model = RawAudioEncoder(sequence_length).cuda()

data_folder = "./data"

# audio_files = [f"{data_folder}/2000s/{x}" for x in os.listdir(f"{data_folder}/2000s") if x.endswith(".wav")] + [f"{data_folder}/2010s/{x}" for x in os.listdir(f"{data_folder}/2010s") if x.endswith(".wav")]
audio_files = [f"{data_folder}/{x}" for x in os.listdir(f"{data_folder}") if x.endswith(".wav")][:65]

X_train, X_test = train_test_split(audio_files, train_size=0.7, random_state=42)

multiplier = 32

train_ds = AudioFileDataset(X_train, sequence_length, multiplier=multiplier)
# val_ds = AudioFileDataset(X_test, sequence_length, multiplier=multiplier)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_dl = DataLoader(val_ds, batch_size=ae_batch_size*2)

train(model, train_dl)
# print(model)
