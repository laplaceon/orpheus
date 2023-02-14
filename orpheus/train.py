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

from model.rae import Orpheus

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
# sequence_length = 131072

n_fft = 1024
n_mels = 64
to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=bitrate, n_fft=n_fft, n_mels=n_mels).cuda()
from_mel = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=bitrate).cuda()
to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).cuda()
to_wave = torchaudio.transforms.GriffinLim(n_fft=n_fft).cuda()
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

def recons_loss(inp, tgt, time_weight=1.0, freq_weight=1.0, reduction="mean"):
    # lcosh = auraloss.time.LogCoshLoss(reduction=reduction)

    fft_sizes = [2048, 1024, 512, 256, 128, 64]
    hops = [int(0.25*fft) for fft in fft_sizes]

    stft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, hop_sizes=hops, win_lengths=fft_sizes, reduction=reduction)

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
    data_spec = to_mel(data[:15].unsqueeze(1))

    with torch.no_grad():
        output, kl = model(data_spec)
        output = output.flatten()
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
        "Waiting For The End [Official Music Video] - Linkin Park-HQ.wav"
    ]

    for test_file in test_files:
        out = get_song_features(model, f"../input/{test_file}")
        print(out)
        torchaudio.save(f"../output/{slugify(test_file)}_epoch{epoch+1}.wav", out.cpu().unsqueeze(0), bitrate)


def train(model, train_dl, lr=1e-4, beta=1.0):
    opt = Adam(model.parameters(), lr)
    lr_scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=10, verbose=True)

    step = 0

    i = 0
    while True:
        model.train()

        nb = 0
        r_loss_total = 0
        d_loss_total = 0
        for batch in tqdm(train_dl, position=0, leave=True):
            real_imgs = batch["input"].unsqueeze(1).cuda()
            # print(real_imgs.shape)
            bs = real_imgs.shape[0]

            # with torch.no_grad():
                # modded = apply_augmentations(real_imgs.unsqueeze(1), sample_rate=bitrate)
                # mel_imgs = to_db(to_mel(modded.squeeze(1)))
                # mel_imgs = to_db(to_mel(real_imgs))

            opt.zero_grad()

            z, d_loss = model(real_imgs)
            r_loss = recons_loss(z, real_imgs)
            loss = r_loss[0] + r_loss[1]

            # print(r_loss, d_loss)
            r_loss_total += r_loss[0].item()
            d_loss_total += d_loss.item()

            loss.backward()
            opt.step()
            step += 1

            if torch.isnan(r_loss[0]):
                raise SystemError

            nb += 1
        lr_scheduler.step(loss)
        # while beta_kl <= 1:
        #     beta_kl *= sqrt(10)
        # scheduler steps
        print(f"D Loss: {d_loss_total/nb}, R Loss: {r_loss_total/nb}")

        if (i+1) % 4 == 0:
            torch.save(model.state_dict(), f"../models/rae_{i+1}.pt")
        #     real_eval(model, i)

        i += 1


model = Orpheus(sequence_length).cuda()

data_folder = "../data"

# audio_files = [f"{data_folder}/2000s/{x}" for x in os.listdir(f"{data_folder}/2000s") if x.endswith(".wav")] + [f"{data_folder}/2010s/{x}" for x in os.listdir(f"{data_folder}/2010s") if x.endswith(".wav")]
audio_files = [f"{data_folder}/{x}" for x in os.listdir(f"{data_folder}") if x.endswith(".wav")][:65]

X_train, X_test = train_test_split(audio_files, train_size=0.7, random_state=42)

multiplier = 32

train_ds = AudioFileDataset(X_train, sequence_length, multiplier=multiplier)
# val_ds = AudioFileDataset(X_test, sequence_length, multiplier=multiplier)
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_dl = DataLoader(val_ds, batch_size=ae_batch_size*2)

# train(model, train_dl)
# model.load_state_dict(torch.load("../models/ravae_5l_vae_0.pt"))
# real_eval(model, 3)
# print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
