import torchaudio
import cdpam

import numpy as np

from pprint import pprint

from slugify import slugify

import core.loss as closs
import core.process as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from model.rae import Orpheus
from model.prior import GaussianPrior
from model.slicer import MPSSlicer

from early import EarlyStopping

from lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler

from torchaudio.functional import mu_law_encoding

from dataset import AudioFileDataset, aggregate_wavs
from sklearn.model_selection import train_test_split

from torchaudio.transforms import Resample

from torch_audiomentations import SomeOf, OneOf, PolarityInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter, BandPassFilter, PeakNormalization, PitchShift

from tqdm import tqdm

# torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

# Params
bitrate = 44100
sequence_length = 131072

apply_augmentations = SomeOf(
    num_transforms = (1, 3),
    transforms = [
        PolarityInversion(),
        # AddColoredNoise(),
        PitchShift(sample_rate=bitrate),
        Gain(),
        OneOf(
            transforms = [
                HighPassFilter(),
                LowPassFilter()
            ]
        )
    ]
)

peak_norm = PeakNormalization(apply_to="only_too_loud_sounds", p=1., sample_rate=bitrate).cuda()

class TrainerAE(nn.Module):
    def __init__(self, backbone, prior, slicer):
        super().__init__()

        self.backbone = backbone
        self.prior = prior
        self.slicer = slicer

        log_epsilon = 1e-7

        stft = closs.MultiScaleSTFT([2048, 1024, 512, 256, 128], bitrate, num_mels=128)
        # self.compound_distance = closs.AudioDistanceV2(stft, backbone.translator, 4096, 4, log_epsilon)
        self.distance = closs.AudioDistanceV1(stft, log_epsilon)
        self.perceptual_distance = cdpam.CDPAM()

        self.resample_fp = Resample(bitrate, 22050)

    def forward(self, x, x_quantized):
        y_subbands, y_probs, x_subbands, z, mask = self.backbone(x)

        ce_mask = mask.repeat(1, 2048 // 16)

        # Consider replacing with discretized mixture of logistic distributions loss
        continuity_loss = F.cross_entropy(y_probs, x_quantized, reduction="none")
        continuity_loss = (continuity_loss * ce_mask).sum() / ce_mask.sum()

        # continuity_loss, y_subbands = self.prob_distance(y_subbands, x_subbands, mask=ce_mask)

        # y_subbands_expected, y_subbands = self.backbone.expected_from_decoded(y_subbands)
        y = self.backbone.recompose(y_subbands)

        mb_dist = self.distance(y_subbands, x_subbands)
        fb_dist = self.distance(y, x)

        r_loss = mb_dist["spectral_distance"] + fb_dist["spectral_distance"]
        # continuity_loss = mb_dist["entropy_distance"]

        z_samples = self.prior.sample(z.shape[0] * z.shape[2])

        d_loss = slicer.fgw_dist(z.transpose(1, 2).reshape(-1, z.shape[1]), z_samples)

        with torch.no_grad():
            f_loss = self.perceptual_distance.forward(self.resample_fp(x.squeeze(1)), self.resample_fp(y.squeeze(1)))
            # f_loss = F.mse_loss(y, x)

        return (r_loss, continuity_loss, d_loss, torch.mean(f_loss))

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
        output = model.forward_nm(model.decompose(data_spec))
        output = model.recompose(output).flatten()
        # print(output[:5].shape)
        return output

def real_eval(model, epoch):
    model.eval()

    test_files = [
        "Synthwave Coolin'.wav",
        "Waiting For The End [Official Music Video] - Linkin Park-HQ.wav",
        "q1.wav"
    ]

    for test_file in test_files:
        out = get_song_features(model, f"../input/{test_file}")
        # print(out, torch.min(out).item(), torch.max(out).item())
        torchaudio.save(f"../output/{slugify(test_file)}_epoch{epoch+1}.wav", out.cpu().unsqueeze(0), bitrate)

def sample_from_prior(backbone, prior, num_samples):
    backbone.eval()
    prior.eval()

    with torch.no_grad():
        samples = prior.sample(num_samples).unsqueeze(0).transpose(1, 2)

        decoded, _ = backbone.decode(samples)
        full = backbone.recompose(decoded)

        torchaudio.save(f"../output/sampled.wav", full.squeeze(0).cpu(), bitrate)

def cyclic_kl(step, cycle_len, maxp=0.5, min_beta=0, max_beta=1):
    div_shift = 1 / (1 - min_beta/max_beta)
    return min(((step % cycle_len) / (cycle_len * maxp * div_shift)) + (min_beta / max_beta), 1) * max_beta

def eval(model, val_dl, hparams=None):
    loss_total = 0
    d_loss_total = 0
    r_loss_total = 0
    c_loss_total = 0
    f_loss_total = 0
    nb = 0
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, position=0, leave=True):
            inp = batch['input'].unsqueeze(1)

            mod = peak_norm(inp.cuda())
            x_avg = F.avg_pool1d(mod, 16)
            x_quantized = mu_law_encoding(x_avg.squeeze(1), hparams["quantize_bins"])

            r_loss, c_loss, d_loss, f_loss = model(mod, x_quantized)

            loss = r_loss + c_loss + f_loss

            loss_total += loss.item()
            d_loss_total += d_loss.item()
            r_loss_total += r_loss.item()
            c_loss_total += c_loss.item()
            f_loss_total += f_loss.item()

            nb += 1

    print(f"Valid loss: {loss_total/nb}, D loss: {d_loss_total/nb}, R loss: {r_loss_total/nb}, C loss: {c_loss_total/nb}, F loss: {f_loss_total/nb}")
    return loss_total/nb

def train(model, train_dl, val_dl, lr, hparams=None, mixed_precision=False, compile=False, warmup=None, checkpoint=None):
    print("Learning rate:", lr)
    opt = AdamW(model.parameters(), lr)
    scaler = GradScaler(enabled=mixed_precision)

    val_loss_min = np.Inf

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])

        val_loss_min = checkpoint["loss"]

    early_stopping = EarlyStopping(patience=10, verbose=True, val_loss_min=val_loss_min, path="../models/ravae_stage1.pt")

    if compile:
        model = torch.compile(model)

    if warmup is not None:
        scheduler = WarmupLRScheduler(
          opt, 
          init_lr=warmup[0], 
          peak_lr=lr, 
          warmup_steps=warmup[1] * len(train_dl),
    )

    step = 0
    epoch = 0

    model.prior.print_parameters()
    model.slicer.print_parameters()

    # real_eval(model.backbone, i)
    while True:
        model.train()

        nb = 0

        training_loss = 0
        r_loss_total = 0
        c_loss_total = 0
        f_loss_total = 0
        m_loss_total = 0
        d_loss_total = 0

        print(f"Epoch {epoch+1}")
        for batch in tqdm(train_dl, position=0, leave=True):
            real_imgs = batch["input"].unsqueeze(1)

            with torch.no_grad():
                beginning = peak_norm(apply_augmentations(real_imgs, sample_rate=bitrate).cuda())
                x_avg = F.avg_pool1d(beginning, 16)
                x_quantized = mu_law_encoding(torch.clamp(x_avg, -1, 1).squeeze(1), hparams["quantize_bins"])
                # x_quantized = utils.quantize_waveform(x_quantized, quantize_bins, pool=16).cuda()

            # Try set to none
            opt.zero_grad()

            with torch.autocast('cuda', dtype=torch.float16, enabled=mixed_precision):
                r_loss, c_loss, d_loss, f_loss = model(beginning, x_quantized)
                
                # skip = warmup[1] if warmup is not None else 2

                r_loss_beta, c_loss_beta, f_loss_beta = 1., 0.6, 0.2
                d_loss_beta = 0.
                # d_loss_beta = cyclic_kl(step, total_batch * reg_loss_cycle, maxp=1, max_beta=1e-5) if i > reg_skip-1 else 0.

                loss = (r_loss_beta * r_loss) + (c_loss_beta * c_loss) + (f_loss_beta * f_loss)

            training_loss += loss.item()
            r_loss_total += r_loss.item() / 2
            c_loss_total += c_loss.item()
            f_loss_total += f_loss.item()
            d_loss_total += d_loss.item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if warmup is not None:
                scheduler.step()

            nb += 1
            step += 1
        print(f"Train loss: {training_loss/nb}, D Loss: {d_loss_total/nb}, R Loss: {r_loss_total/nb}, C Loss: {c_loss_total/nb}, F Loss: {f_loss_total/nb}")
        epoch += 1

        valid_loss = eval(model, val_dl, hparams)
        early_stopping(valid_loss, model, {"opt": opt.state_dict()})
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            real_eval(model.backbone, epoch)


model = Orpheus(enc_ds_expansion_factor=1.6, dec_ds_expansion_factor=1.6, drop_path=0.05, fast_recompose=True)
prior = GaussianPrior(128, 3)
slicer = MPSSlicer(128, 3, 50)

trainer = TrainerAE(model, prior, slicer).cuda()

data_folder = "../data"

audio_files = aggregate_wavs([f"{data_folder}/Classical", f"{data_folder}/Electronic", f"{data_folder}/Hip Hop", f"{data_folder}/Jazz", f"{data_folder}/Metal", f"{data_folder}/Pop", f"{data_folder}/R&B", f"{data_folder}/Rock"])
X_train, X_test = train_test_split(audio_files, train_size=0.8, random_state=42)

hparams = {
    "quantize_bins": 256
}

training_params = {
    "batch_size": 24, # Set to multiple of 8 if mixed_precision is True
    "learning_rate": 1e-4,
    "dataset_multiplier": 256,
    "dataloader_num_workers": 4,
    "dataloader_pin_mem": False,
    "mixed_precision": True,
    "compile": False
}

train_ds = AudioFileDataset(X_train, sequence_length, multiplier=training_params["dataset_multiplier"])
val_ds = AudioFileDataset(X_test, sequence_length, multiplier=training_params["dataset_multiplier"])
train_dl = DataLoader(train_ds, batch_size=training_params["batch_size"], shuffle=True, num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])
val_dl = DataLoader(val_ds, batch_size=training_params["batch_size"], num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

train(trainer, train_dl, val_dl, lr=training_params["learning_rate"], mixed_precision=training_params["mixed_precision"], compile=training_params["compile"], hparams=hparams)
# checkpoint = torch.load("../models/ravae_stage1.pt")
# trainer.load_state_dict(checkpoint["model"])
# real_eval(trainer.backbone, 1001)
# sample_from_prior(trainer.backbone, trainer.prior, 64 * 4)
# print(model)
