import torchaudio

import numpy as np

from pprint import pprint

from slugify import slugify

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
from model.discriminator import MultiScaleSpectralDiscriminator, MultiScaleSpectralDiscriminator1d, CombineDiscriminators, MultiPeriodDiscriminator, MultiScaleDiscriminator, ConvNet
from model.mol_translate import MoLTranslate

from train_helper import TrainerAE

from early import EarlyStopping

from lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler

from dataset import AudioFileDataset, aggregate_wavs
from sklearn.model_selection import train_test_split

from torchaudio.transforms import Resample

from torch_audiomentations import SomeOf, OneOf, PolarityInversion, TimeInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter, BandPassFilter, PeakNormalization, PitchShift

from tqdm import tqdm

# Params
sample_rate = 44100
sequence_length = 131072

apply_augmentations = SomeOf(
    num_transforms = (1, 3),
    transforms = [
        PolarityInversion(),
        PitchShift(sample_rate=sample_rate),
        Gain(),
        OneOf(
            transforms = [
                HighPassFilter(),
                LowPassFilter()
            ]
        )
    ]
)

apply_augmentations_adv = SomeOf(
    num_transforms = (1, 3),
    transforms = [
        PolarityInversion(),
        TimeInversion(),
        AddColoredNoise(),
        PitchShift(sample_rate=sample_rate),
        Gain(),
        OneOf(
            transforms = [
                HighPassFilter(),
                LowPassFilter()
            ]
        )
    ]
)

peak_norm = PeakNormalization(apply_to="only_too_loud_sounds", p=1., sample_rate=sample_rate).cuda()

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
        output, _ = model.forward_nm(model.decompose(data_spec))
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
        print(out, torch.min(out).item(), torch.max(out).item())
        torchaudio.save(f"../output/{slugify(test_file)}_epoch{epoch}_ps.wav", out.cpu().unsqueeze(0), sample_rate)

def sample_from_prior(backbone, prior, num_samples):
    backbone.eval()
    prior.eval()

    with torch.no_grad():
        samples = prior.sample(num_samples).unsqueeze(0).transpose(1, 2)

        decoded, _ = backbone.decode(samples)
        full = backbone.recompose(decoded)

        torchaudio.save(f"../output/sampled.wav", full.squeeze(0).cpu(), sample_rate)

def cyclic_kl(step, cycle_len, maxp=0.5, min_beta=0, max_beta=1):
    div_shift = 1 / (1 - min_beta/max_beta)
    return min(((step % cycle_len) / (cycle_len * maxp * div_shift)) + (min_beta / max_beta), 1) * max_beta

def eval(model, val_dl, hparams=None, stage=1):
    valid_loss = 0
    r_loss_total = 0
    fb_loss_total = 0
    c_loss_total = 0
    f_loss_total = 0
    d_loss_total = 0
    nb = 0

    r_loss_beta, d_loss_beta, f_loss_beta = 1., 1e-4, 0.2
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, position=0, leave=True):
            inp = batch['input'].unsqueeze(1)
            
            mod = peak_norm(inp.cuda())
            # x_avg = F.avg_pool1d(mod, 16)
            # x_quantized = mu_law_encoding(x_avg.squeeze(1), hparams["quantize_bins"])

            if stage == 1:
                mb_loss, fb_loss, d_loss, f_loss = model(mod)
                r_loss = mb_loss + fb_loss
                loss = (r_loss_beta * r_loss) + (d_loss_beta * d_loss)
                # loss = (r_loss_beta * r_loss)
            elif stage == 2:
                mb_loss, fb_loss, d_loss, f_loss = model.forward_nm(mod)
                r_loss = mb_loss + fb_loss
                loss = (r_loss_beta * r_loss) + (f_loss_beta * f_loss)

            valid_loss += loss.item()
            r_loss_total += r_loss.item() / 2
            fb_loss_total += fb_loss.item()
            f_loss_total += f_loss.item()
            d_loss_total += d_loss.item()

            nb += 1

    print(f"Valid loss: {valid_loss/nb}, Dist Loss: {d_loss_total/nb}, Recon Loss: {r_loss_total/nb}, FB Loss: {fb_loss_total/nb}, Time Loss: {f_loss_total/nb}")
    return valid_loss/nb

def train(model, train_dl, val_dl, lr, hparams=None, stage=1, mixed_precision=False, compile=False, warmup=None, checkpoint=None, disc=None, disc_checkpoint=None, save_paths=None):
    print("Learning rate:", lr)

    betas = (0.9, 0.999)

    if stage == 1:
        model.stage1()
    elif stage == 2:
        disc.cuda()
        model.stage2()

        # betas = (0.7, 0.99)
        opt_dis = AdamW(disc.parameters(), lr, betas=betas)
    
    opt = AdamW(model.parameters(), lr, betas=betas)
    scaler = GradScaler(enabled=mixed_precision)

    val_loss_min = None

    resuming = False

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        if stage == 1 or (disc_checkpoint is not None):
            val_loss_min = checkpoint["loss"]
        print(f"Resuming from model with val loss: {checkpoint['loss']}")
        resuming = True
    
    if (disc_checkpoint is not None) and stage == 2:
        disc.load_state_dict(disc_checkpoint["model"])
        opt_dis.load_state_dict(disc_checkpoint["opt"])
        print("Resuming from disc checkpoint")

    # print(model.backbone)

    early_stopping = EarlyStopping(patience=10, verbose=True, val_loss_min=val_loss_min)

    if compile:
        model = torch.compile(model)

    if warmup is not None:
        scheduler = WarmupLRScheduler(
          opt, 
          init_lr=warmup[0], 
          peak_lr=lr, 
          warmup_steps=warmup[1] * len(train_dl)
    )

    step = 0
    epoch = 0

    model.prior.print_parameters()
    model.slicer.print_parameters()

    total_batch = len(train_dl)

    # real_eval(model.backbone, i)
    while True:
        model.train()

        nb = 0

        training_loss = 0
        r_loss_total = 0
        fb_loss_total = 0
        c_loss_total = 0
        f_loss_total = 0
        adv_loss_total = 0
        disc_loss_total = 0
        fm_loss_total = 0
        d_loss_total = 0

        print(f"Epoch {epoch+1}")
        for batch in tqdm(train_dl, position=0, leave=True):
            real_imgs = batch["input"].unsqueeze(1)

            with torch.no_grad():
                if stage == 1:
                    beginning = peak_norm(apply_augmentations(real_imgs, sample_rate=sample_rate).cuda())
                elif stage == 2:
                    beginning = peak_norm(apply_augmentations_adv(real_imgs, sample_rate=sample_rate).cuda())
            #     x_avg = F.avg_pool1d(beginning, 16)
            #     x_quantized = mu_law_encoding(torch.clamp(x_avg, -1, 1).squeeze(1), hparams["quantize_bins"])
            #     x_quantized = utils.quantize_waveform(x_quantized, quantize_bins, pool=16).cuda()

            with torch.autocast('cuda', dtype=torch.float16, enabled=mixed_precision):
                if stage == 1:
                    mb_loss, fb_loss, d_loss, f_loss = model(beginning)
                    r_loss = mb_loss + fb_loss
                    
                    # skip = warmup[1] if warmup is not None else 2

                    r_loss_beta = 1.
                    # d_loss_beta = cyclic_kl(step, total_batch * 1, maxp=1, max_beta=3e-7)
                    d_loss_beta = 1e-4 if ((stage == 1) and (epoch >= hparams["dist_skip_epochs"] or resuming)) else 0.

                    loss = (r_loss_beta * r_loss) + (d_loss_beta * d_loss)
                    # loss = (r_loss_beta * r_loss)
                elif stage == 2:
                    r_loss, adv_loss, disc_loss, fm_loss, f_loss = model.forward_wd(beginning, disc)

                    r_loss_beta, adv_loss_beta, disc_loss_beta, fm_loss_beta = 1., 1., 1., 20.
                    loss = (r_loss_beta * r_loss) + (adv_loss_beta * adv_loss) + (fm_loss_beta * fm_loss)
                    disc_loss = disc_loss_beta * disc_loss

            training_loss += loss.item()

            r_loss_total += r_loss.item() / 2
            f_loss_total += f_loss.item()

            if stage == 1:
                d_loss_total += d_loss.item()
                fb_loss_total += fb_loss.item()
            elif stage == 2:
                adv_loss_total += adv_loss.item()
                disc_loss_total += disc_loss.item()
                fm_loss_total += fm_loss.item()

            if stage == 2 and (step % 4 == 0):
                opt_dis.zero_grad()
                scaler.scale(disc_loss).backward()
                scaler.step(opt_dis)
            else:
                # Try set to none
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)

            scaler.update()

            if warmup is not None:
                scheduler.step()

            nb += 1
            step += 1
        if stage == 1:
            print(f"Train loss: {training_loss/nb}, Dist Loss: {d_loss_total/nb}, Recon Loss: {r_loss_total/nb}, FB Loss: {fb_loss_total/nb}, Time Loss: {f_loss_total/nb}")
        elif stage == 2:
            print(f"Train loss: {training_loss/nb}, Disc Loss: {disc_loss_total/nb}, Adv Loss: {adv_loss_total/nb}, FM Loss: {fm_loss_total/nb}, Recon Loss: {r_loss_total/nb}, Time Loss: {f_loss_total/nb}")
        
        epoch += 1

        valid_loss = eval(model, val_dl, hparams, stage)
        if early_stopping(valid_loss):
            if stage == 1:
                early_stopping.save_checkpoint(valid_loss, [{"model": model.state_dict(), "opt": opt.state_dict()}], [save_paths[0]])
            elif stage == 2:
                early_stopping.save_checkpoint(valid_loss, [
                    {"model": model.state_dict(), "opt": opt.state_dict()}, 
                    {"model": disc.state_dict(), "opt": opt_dis.state_dict()}
                ], save_paths)
            real_eval(model.backbone, epoch)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
model = Orpheus(enc_ds_expansion_factor=1.5, dec_ds_expansion_factor=1.5, enc_drop_path=0.05, dec_drop_path=0.05, fast_recompose=True)
# model = Orpheus(enc_ds_expansion_factor=1.5, dec_ds_expansion_factor=1.5, dec_drop_path=0.05, fast_recompose=True)
# model = Orpheus(enc_ds_expansion_factor=1.5, dec_ds_expansion_factor=1.5, fast_recompose=True)

# print(model)

K = 3
prior = GaussianPrior(128, K)
slicer = MPSSlicer(128, K, 50)
# disc_scales = [4096, 2048, 1024, 512, 256]
# conv_period = ConvNet(1, 1, (5, 1), (2, 1), nn.Conv2d)
# conv_scale = ConvNet(1, 1, 15, 7, nn.Conv1d)
# discriminator = CombineDiscriminators([
#     MultiPeriodDiscriminator([2, 3, 5, 7, 11], conv_period), 
#     MultiScaleDiscriminator(3, conv_scale), 
#     # MultiScaleSpectralDiscriminator1d(disc_scales)
# ])
discriminator = None

trainer = TrainerAE(model, prior, slicer)

data_folder = "../data"

audio_files = aggregate_wavs([f"{data_folder}/Classical", f"{data_folder}/Electronic", f"{data_folder}/Hip Hop", f"{data_folder}/Jazz", f"{data_folder}/Metal", f"{data_folder}/Pop", f"{data_folder}/R&B", f"{data_folder}/Rock"])
X_train, X_test = train_test_split(audio_files, train_size=0.8, random_state=42)

hparams = {
    "quantize_bins": 256,
    "dist_skip_epochs": 2
}

training_params = {
    "batch_size": 22, # Set to multiple of 8 if mixed_precision is True
    "learning_rate": 1.5e-4,
    "dataset_multiplier": 384,
    "dataloader_num_workers": 4,
    "dataloader_pin_mem": False,
    "mixed_precision": True,
    "compile": False,
    "warmup": None, #(1e-6, 1),
    "stage": 1,
    "save_paths": ["../models/ravae_stage1_sk2_3en5_1en4e16.pt", "../models/ravae_disc_wave_dp.pt"]
}

train_ds = AudioFileDataset(X_train, sequence_length, multiplier=training_params["dataset_multiplier"])
val_ds = AudioFileDataset(X_test, sequence_length, multiplier=training_params["dataset_multiplier"])
train_dl = DataLoader(train_ds, batch_size=training_params["batch_size"], shuffle=True, num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])
val_dl = DataLoader(val_ds, batch_size=training_params["batch_size"], num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])

# print(trainer)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

# ../models/ravae_stage1_cont_d_1e4.pt = completed first stage model
# ../models/ravae_disc_wave_1_5_e-4.pt = disc with l1 loss ~0.1
# ../models/ravae_stage2.pt = gen with l1 loss ~0.1

checkpoint = torch.load("../models/ravae_stage1.pt")
# checkpoint = None
# disc_checkpoint = torch.load("../models/ravae_disc_wave_cont.pt")
disc_checkpoint = None
train(trainer, train_dl, val_dl, lr=training_params["learning_rate"], 
      stage=training_params["stage"], mixed_precision=training_params["mixed_precision"], 
      compile=training_params["compile"], warmup=training_params["warmup"], hparams=hparams, 
      checkpoint=checkpoint, disc=discriminator, disc_checkpoint=disc_checkpoint, save_paths=training_params["save_paths"])

# trainer.load_state_dict(checkpoint["model"])
# real_eval(trainer.backbone, 1001)
# sample_from_prior(trainer.backbone, trainer.prior, 64 * 4)

# checkpoint = torch.load("../models/ravae_stage1.pt")
# trainer.load_state_dict(checkpoint["model"])

# torch.save(trainer.backbone.state_dict(), "../models/orpheus_stage2.pt")