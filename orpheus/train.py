import torchaudio

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
from model.discriminator import MultiScaleSpectralDiscriminator, MultiScaleSpectralDiscriminator1d, CombineDiscriminators, MultiPeriodDiscriminator, MultiScaleDiscriminator, ConvNet
from model.mol_translate import MoLTranslate

from early import EarlyStopping

from lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler

from dataset import AudioFileDataset, aggregate_wavs
from sklearn.model_selection import train_test_split

from torchaudio.transforms import Resample

from torch_audiomentations import SomeOf, OneOf, PolarityInversion, TimeInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter, BandPassFilter, PeakNormalization, PitchShift

from tqdm import tqdm

# Params
bitrate = 44100
sequence_length = 131072

apply_augmentations = SomeOf(
    num_transforms = (1, 3),
    transforms = [
        PolarityInversion(),
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

apply_augmentations_adv = SomeOf(
    num_transforms = (1, 3),
    transforms = [
        PolarityInversion(),
        TimeInversion(),
        AddColoredNoise(),
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

        scales = [2048, 1024, 512, 256, 128]
        num_mels = 128

        self.backbone = backbone
        self.prior = prior
        self.slicer = slicer
        
        # self.translator = MoLTranslate(scales, num_mels)

        log_epsilon = 1e-7
        self.num_skipped_features = 1

        stft = closs.MultiScaleSTFT(scales, bitrate, num_mels=num_mels)
        # self.entropy_distance = closs.AudioDistanceCE(stft, self.translator, 4096, 4)
        self.distance = closs.AudioDistanceV1(stft, log_epsilon)
        # self.perceptual_distance = cdpam.CDPAM()

        # self.resample_fp = Resample(bitrate, 22050)

        # self.relative_md = 

    def stage1(self):
        self.backbone.cuda()
        self.prior.cuda()
        self.slicer.cuda()
        self.distance.cuda()
    
    def stage2(self):
        self.backbone.freeze_encoder()
        self.backbone.cuda()
        self.distance.cuda()

    def split_features(self, features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
        return feature_real, feature_fake

    def forward_nm(self, x):
        x_subbands = self.backbone.decompose(x)
        y_subbands = self.backbone.forward_nm(x_subbands)

        y = self.backbone.recompose(y_subbands)

        mb_dist = self.distance(y_subbands, x_subbands)
        fb_dist = self.distance(y, x)

        r_loss = mb_dist["spectral_distance"] + fb_dist["spectral_distance"]

        # z_samples = self.prior.sample(z.shape[0] * z.shape[2])

        # d_loss = slicer.fgw_dist(z.transpose(1, 2).reshape(-1, z.shape[1]), z_samples)

        with torch.no_grad():
            # f_loss = self.perceptual_distance.forward(self.resample_fp(x.squeeze(1)), self.resample_fp(y.squeeze(1)))
            f_loss = F.l1_loss(y, x)

        return (r_loss, torch.tensor(0.), f_loss)

    def forward_wd(self, x, discriminator):
        x_subbands = self.backbone.decompose(x)
        y_subbands = self.backbone.forward_nm(x_subbands)

        y = self.backbone.recompose(y_subbands)

        mb_dist = self.distance(y_subbands, x_subbands)
        fb_dist = self.distance(y, x)

        r_loss = mb_dist["spectral_distance"] + fb_dist["spectral_distance"]

        xy = torch.cat([x, y], 0)
        features = discriminator(xy)

        feature_real, feature_fake = self.split_features(features)

        feature_matching_distance = 0.
        loss_dis = 0
        loss_adv = 0

        for scale_real, scale_fake in zip(feature_real, feature_fake):
            current_feature_distance = sum(
                map(
                    lambda a, b : closs.mean_difference(a, b, relative=True),
                    scale_real[self.num_skipped_features:],
                    scale_fake[self.num_skipped_features:],
                )) / len(scale_real[self.num_skipped_features:])

            feature_matching_distance = feature_matching_distance + current_feature_distance

            _dis, _adv = closs.hinge_gan(scale_real[-1], scale_fake[-1])

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv

        feature_matching_distance = feature_matching_distance / len(feature_real)

        with torch.no_grad():
            f_loss = F.l1_loss(y, x)

        return (r_loss, loss_adv, loss_dis, feature_matching_distance, f_loss)

    def forward(self, x):
        y_subbands, _, x_subbands, z, mask = self.backbone(x)

        # ce_mask = mask.repeat(1, 2048 // 16)

        # Consider replacing with discretized mixture of logistic distributions loss
        # continuity_loss = F.cross_entropy(y_probs, x_quantized, reduction="none")
        # continuity_loss = (continuity_loss * ce_mask).sum() / ce_mask.sum()

        # y_weights, y_means, y_scales = self.backbone.expand_dml(y_subbands)
        # y_means_weighted = y_means * F.softmax(y_weights, dim=-1).unsqueeze(1)
        # y_subbands = torch.sum(y_means_weighted, dim=2)

        # continuity_loss = self.entropy_distance(x_subbands, y_means, y_means_weighted, y_scales, mask=mask)["entropy_distance"]

        y = self.backbone.recompose(y_subbands)

        mb_dist = self.distance(y_subbands, x_subbands)
        fb_dist = self.distance(y, x)

        r_loss = mb_dist["spectral_distance"] + fb_dist["spectral_distance"]

        z_samples = self.prior.sample(z.shape[0] * z.shape[2])

        d_loss = slicer.fgw_dist(z.transpose(1, 2).reshape(-1, z.shape[1]), z_samples)

        with torch.no_grad():
            # f_loss = self.perceptual_distance.forward(self.resample_fp(x.squeeze(1)), self.resample_fp(y.squeeze(1)))
            f_loss = F.l1_loss(y, x)

        return (r_loss, d_loss, f_loss)

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
        print(out, torch.min(out).item(), torch.max(out).item())
        torchaudio.save(f"../output/{slugify(test_file)}_epoch{epoch}.wav", out.cpu().unsqueeze(0), bitrate)

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

def eval(model, val_dl, hparams=None, stage=1):
    valid_loss = 0
    r_loss_total = 0
    c_loss_total = 0
    f_loss_total = 0
    d_loss_total = 0
    nb = 0

    r_loss_beta, d_loss_beta, f_loss_beta = 1., 2e-4, 0.2
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, position=0, leave=True):
            inp = batch['input'].unsqueeze(1)
            
            mod = peak_norm(inp.cuda())
            # x_avg = F.avg_pool1d(mod, 16)
            # x_quantized = mu_law_encoding(x_avg.squeeze(1), hparams["quantize_bins"])

            if stage == 1:
                r_loss, d_loss, f_loss = model(mod)
                loss = (r_loss_beta * r_loss) + (d_loss_beta * d_loss)
            elif stage == 2:
                r_loss, d_loss, f_loss = model.forward_nm(mod)
                loss = (r_loss_beta * r_loss) + (f_loss_beta * f_loss)

            valid_loss += loss.item()
            r_loss_total += r_loss.item() / 2
            f_loss_total += f_loss.item()
            d_loss_total += d_loss.item()

            nb += 1

    print(f"Valid loss: {valid_loss/nb}, Dist Loss: {d_loss_total/nb}, Recon Loss: {r_loss_total/nb}, Time Loss: {f_loss_total/nb}")
    return valid_loss/nb

def train(model, train_dl, val_dl, lr, hparams=None, stage=1, mixed_precision=False, compile=False, warmup=None, checkpoint=None, disc=None, disc_checkpoint=None, save_paths=None):
    print("Learning rate:", lr)

    betas = (0.9, 0.999)

    if stage == 1:
        model.stage1()
    elif stage == 2:
        disc.cuda()
        model.stage2()

        # betas = (0.5, 0.99)
        opt_dis = AdamW(disc.parameters(), lr, betas=betas)
    
    opt = AdamW(model.parameters(), lr, betas=betas)
    scaler = GradScaler(enabled=mixed_precision)

    val_loss_min = None

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        if stage == 1 or (disc_checkpoint is not None):
            val_loss_min = checkpoint["loss"]
        print(f"Resuming from model with val loss: {checkpoint['loss']}")
    
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
                    beginning = peak_norm(apply_augmentations(real_imgs, sample_rate=bitrate).cuda())
                elif stage == 2:
                    beginning = peak_norm(apply_augmentations_adv(real_imgs, sample_rate=bitrate).cuda())
            #     x_avg = F.avg_pool1d(beginning, 16)
            #     x_quantized = mu_law_encoding(torch.clamp(x_avg, -1, 1).squeeze(1), hparams["quantize_bins"])
            #     x_quantized = utils.quantize_waveform(x_quantized, quantize_bins, pool=16).cuda()

            with torch.autocast('cuda', dtype=torch.float16, enabled=mixed_precision):
                if stage == 1:
                    r_loss, d_loss, f_loss = model(beginning)
                    
                    # skip = warmup[1] if warmup is not None else 2

                    r_loss_beta = 1.
                    # d_loss_beta = cyclic_kl(step, total_batch * 1, maxp=1, max_beta=3e-7)
                    d_loss_beta = 0.

                    loss = (r_loss_beta * r_loss) + (d_loss_beta * d_loss)
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
            print(f"Train loss: {training_loss/nb}, Dist Loss: {d_loss_total/nb}, Recon Loss: {r_loss_total/nb}, Time Loss: {f_loss_total/nb}")
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
            
# model = Orpheus(enc_ds_expansion_factor=1.5, dec_ds_expansion_factor=1.5, enc_drop_path=0.05, dec_drop_path=0.05, fast_recompose=True)
# model = Orpheus(enc_ds_expansion_factor=1.5, dec_ds_expansion_factor=1.5, dec_drop_path=0.025, fast_recompose=True)
model = Orpheus(enc_ds_expansion_factor=1.5, dec_ds_expansion_factor=1.5, fast_recompose=True)

# print(model)

prior = GaussianPrior(128, 3)
slicer = MPSSlicer(128, 3, 50)
disc_scales = [4096, 2048, 1024, 512, 256]
conv_period = ConvNet(1, 1, (5, 1), (2, 1), nn.Conv2d)
conv_scale = ConvNet(1, 1, 15, 7, nn.Conv1d)
discriminator = CombineDiscriminators([
    MultiPeriodDiscriminator([2, 3, 5, 7, 11], conv_period), 
    MultiScaleDiscriminator(3, conv_scale), 
    # MultiScaleSpectralDiscriminator1d(disc_scales)
])

trainer = TrainerAE(model, prior, slicer)

data_folder = "../data"

audio_files = aggregate_wavs([f"{data_folder}/Classical", f"{data_folder}/Electronic", f"{data_folder}/Hip Hop", f"{data_folder}/Jazz", f"{data_folder}/Metal", f"{data_folder}/Pop", f"{data_folder}/R&B", f"{data_folder}/Rock"])
X_train, X_test = train_test_split(audio_files, train_size=0.8, random_state=42)

hparams = {
    "quantize_bins": 256
}

training_params = {
    "batch_size": 36, # Set to multiple of 8 if mixed_precision is True
    "learning_rate": 1.5e-4,
    "dataset_multiplier": 384,
    "dataloader_num_workers": 4,
    "dataloader_pin_mem": False,
    "mixed_precision": True,
    "compile": False,
    "warmup": None, #(1e-6, 1),
    "stage": 2,
    "save_paths": ["../models/ravae_stage2_dp.pt", "../models/ravae_disc_wave_dp.pt"]
}

train_ds = AudioFileDataset(X_train, sequence_length, multiplier=training_params["dataset_multiplier"])
val_ds = AudioFileDataset(X_test, sequence_length, multiplier=training_params["dataset_multiplier"])
train_dl = DataLoader(train_ds, batch_size=training_params["batch_size"], shuffle=True, num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])
val_dl = DataLoader(val_ds, batch_size=training_params["batch_size"], num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

# ../models/ravae_stage1_cont_d_1e4.pt = completed first stage model
# ../models/ravae_disc_wave_1_5_e-4.pt = disc with l1 loss ~0.1
# ../models/ravae_stage2.pt = gen with l1 loss ~0.1

# checkpoint = torch.load("../models/ravae_stage1_cont_d_1e4.pt")
checkpoint = None
# disc_checkpoint = torch.load("../models/ravae_disc_wave_cont.pt")
disc_checkpoint = None
train(trainer, train_dl, val_dl, lr=training_params["learning_rate"], 
      stage=training_params["stage"], mixed_precision=training_params["mixed_precision"], 
      compile=training_params["compile"], warmup=training_params["warmup"], hparams=hparams, 
      checkpoint=checkpoint, disc=discriminator, disc_checkpoint=disc_checkpoint, save_paths=training_params["save_paths"])

# trainer.load_state_dict(checkpoint["model"])
# real_eval(trainer.backbone, 1001)
# sample_from_prior(trainer.backbone, trainer.prior, 64 * 4)

checkpoint = torch.load("../models/ravae_stage2.pt")
trainer.load_state_dict(checkpoint["model"])

torch.save(trainer.backbone.state_dict(), "../models/orpheus_stage2.pt")