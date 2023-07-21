from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union

from slugify import slugify

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchaudio

from torch.optim import AdamW
from torch.cuda.amp import GradScaler

from einops import reduce

from audio_diffusion_pytorch import DiffusionModel, VDiffusion, VSampler
from a_unet import NumberEmbedder, Module, Repeat, exists

from train_diffusion_helper import UNetV1

from model.rae import Orpheus

from exportable import Encoder, Upscaler

from torch.utils.data import DataLoader
from dataset import aggregate_wavs, AudioFileDataset
from sklearn.model_selection import train_test_split

from early import EarlyStopping
from tqdm import tqdm
from enum import Enum

from torch_audiomentations import SomeOf, Gain, PolarityInversion, PeakNormalization, HighPassFilter, LowPassFilter, BandPassFilter, OneOf, PitchShift

sequence_length = 131072 * 8
sample_rate = 44100
NUM_GENRES = 8


apply_augmentations = SomeOf(
    num_transforms = (1, None),
    transforms = [
        PolarityInversion(),
        PitchShift(sample_rate=sample_rate),
        Gain(),
        # OneOf(
        #     transforms = [
        #         HighPassFilter(),
        #         LowPassFilter(),
        #         # BandPassFilter()
        #     ]
        # )
    ]
).cuda()

peak_norm = PeakNormalization(apply_to="only_too_loud_sounds", p=1., sample_rate=sample_rate).cuda()

class Genre(Enum):
    CLASSICAL = 0
    ELECTRONIC = 1
    HIPHOP = 2
    JAZZ = 3
    METAL = 4
    POP = 5
    RANDB = 6
    ROCK = 7


def real_eval(model, encoder, upscaler, genre_embedder, epoch):
    genres_map = {
        0: "classical",
        1: "electronic",
        2: "hiphop",
        3: "jazz",
        4: "metal",
        5: "pop",
        6: "randb",
        7: "rock"
    }

    test_files = ["neurotic", "xotour", "Synthwave Coolin'"]

    print("Generating variations")
    for test_file in tqdm(test_files):
        # Turn noise into new audio sample with diffusion
        data, _ = torchaudio.load(f"../input/{test_file}.wav")
        bal = 0.5

        if data.shape[0] == 2:
            data = bal * data[0, :] + (1 - bal) * data[1, :]
        else:
            data = data[0, :]

        model.eval()
        genre_embedder.eval()

        with torch.no_grad():
            chopped = data[:sequence_length * 2].cuda().unsqueeze(0).unsqueeze(1)
            encoded_wave = encoder(chopped)
            noise = torch.randn_like(encoded_wave).cuda()
            noisy_latent = torch.lerp(encoded_wave, noise, 0.2)

            for i in range(NUM_GENRES):
                genre = F.one_hot(torch.tensor(i), NUM_GENRES).cuda().unsqueeze(0).float()
                genre = genre_embedder(genre.unsqueeze(1))

                sample = model.sample(
                    noisy_latent,
                    embedding=genre,
                    embedding_scale=7.5, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
                    num_steps=100 # Higher for better quality, suggested num_steps: 10-100
                )

                upscaled = upscaler(sample)
                # with_vocals = 
                torchaudio.save(f"../output/{slugify(test_file)}_{genres_map[i]}_epoch{epoch}.wav", upscaled.cpu().squeeze(0), sample_rate)

def eval(model, encoder, val_dl, cfg_rate, genre_embedder=None):
    valid_loss = 0
    nb = 0

    model.eval()
    if genre_embedder is not None:
        genre_embedder.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, position=0, leave=True):
            audio_wave = batch['input'].unsqueeze(1)
            genres = F.one_hot(batch["genre"].cuda(), 8).float().unsqueeze(1)
            if genre_embedder is not None:
                genres = genre_embedder(genres)
            encoded_wave = encoder(audio_wave.cuda())

            loss = model(
                encoded_wave,
                embedding=genres, # Text conditioning, one element per batch
                embedding_mask_proba=cfg_rate # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
            )

            valid_loss += loss.item()

            nb += 1
        
        print(f"Valid loss: {valid_loss/nb}")
        return valid_loss/nb

def train(model, encoder, upscaler, genre_embedder, train_dl, val_dl, lr=1e-4, cfg_rate=0., mixed_precision=False, checkpoint=None, save_path=None):
    opt = AdamW(list(model.parameters()) + list(genre_embedder.parameters()), lr, betas=(0.5, 0.99))
    scaler = GradScaler(enabled=mixed_precision)

    val_loss_min = None

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        val_loss_min = checkpoint["loss"]
        print(f"Resuming from model with val loss: {checkpoint['loss']}")

    early_stopping = EarlyStopping(patience=10, verbose=True, val_loss_min=val_loss_min)

    epoch = 0
    while True:
        model.train()
        genre_embedder.train()

        nb = 0
        training_loss = 0

        print(f"Epoch {epoch+1}")
        for batch in tqdm(train_dl, position=0, leave=True):
            audio_wave = batch["input"].unsqueeze(1)
            genres = F.one_hot(batch["genre"].cuda(), NUM_GENRES).float().unsqueeze(1)

            if genre_embedder is not None:
                genres = genre_embedder(genres)

            with torch.no_grad():
                # encoded_wave = encoder(audio_wave.cuda())
                encoded_wave = encoder(peak_norm(apply_augmentations(audio_wave.cuda(), sample_rate)))

            with torch.autocast('cuda', dtype=torch.float16, enabled=mixed_precision):
                loss = model(
                    encoded_wave,
                    embedding=genres, # Text conditioning, one element per batch
                    embedding_mask_proba=cfg_rate # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
                )

            training_loss += loss.item()

            opt.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            nb += 1

        print(f"Train loss: {training_loss/nb}")
        epoch += 1

        valid_loss = eval(model, encoder, val_dl, cfg_rate, genre_embedder)
        if early_stopping(valid_loss):
            early_stopping.save_checkpoint(valid_loss, [{"model": model.state_dict(), "opt": opt.state_dict()}, {"model": genre_embedder.state_dict()}], save_path)
            real_eval(model, encoder, upscaler, genre_embedder, epoch)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

# def TimeConditioningPlugin(
#     net_t: Type[nn.Module],
#     num_layers: int = 2,
# ) -> Callable[..., nn.Module]:
#     """Adds time conditioning (e.g. for diffusion)"""

#     def Net(modulation_features: Optional[int] = None, **kwargs) -> nn.Module:
#         msg = "TimeConditioningPlugin requires modulation_features"
#         assert exists(modulation_features), msg

#         embedder = NumberEmbedder(features=modulation_features)
#         mlp = Repeat(
#             nn.Sequential(
#                 nn.Linear(modulation_features, modulation_features), nn.GELU()
#             ),
#             times=num_layers,
#         )
#         net = net_t(modulation_features=modulation_features, **kwargs)  # type: ignore

#         def forward(
#             x: Tensor,
#             time: Optional[Tensor] = None,
#             features: Optional[Tensor] = None,
#             **kwargs,
#         ):
#             msg = "TimeConditioningPlugin requires time in forward"
#             assert exists(time), msg
#             # Process time to time_features
#             time_features = F.gelu(embedder(time))
#             time_features = mlp(time_features)
#             # Overlap features if more than one per batch
#             if time_features.ndim == 3:
#                 time_features = reduce(time_features, "b n d -> b d", "sum")
#             # Merge time features with features if provided
#             features = features + time_features if exists(features) else time_features
#             return net(x, features=features, **kwargs)

#         return Module([embedder, mlp, net], forward)

#     return Net


class GenreConditioningPlugin(nn.Module):
    def __init__(
        self,
        num_genres = 8,
        h_dim = 32
    ):
        super().__init__()

        # self.embedder = NumberEmbedder(num_genres, h_dim)

        self.net = nn.Sequential(
            nn.Linear(num_genres, h_dim),
            nn.GELU(),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.GELU()
        )

    def forward(self, genre):
        # x = F.gelu(self.embedder(genre))
        return self.net(genre)


model = DiffusionModel(
    net_t=UNetV1, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=128, # U-Net: number of input/output (audio) channels
    channels=[256, 256, 512, 512, 768, 768, 896], # U-Net: channels at each layer
    factors=[1, 2, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 1, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
    use_text_conditioning=False, # U-Net: enables text conditioning (default T5-base)
    use_embedding_cfg=True, # U-Net: enables classifier free guidance
    embedding_max_length=8, # U-Net: text embedding maximum length (default for T5-base)
    embedding_features=32, # U-Net: text mbedding features (default for T5-base)
    cross_attentions=[0, 0, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
)

# class DiffusionWrapper(nn.Module):
#     def __init__(self, diffusion, embedder):
#         super().__init__()
    
#         self.diffusion = diffusion
#         self.embedder = embedder

orpheus = Orpheus(enc_ds_expansion_factor=1.5, dec_ds_expansion_factor=1.5, fast_recompose=True)
orpheus_chk = torch.load("../models/orpheus_stage1_sk3_2en2_4m.pt")
orpheus.load_state_dict(orpheus_chk)

encoder = Encoder(orpheus.encoder, orpheus.pqmf)
upscaler = Upscaler(orpheus.decoder, orpheus.pqmf)

genre_embedder = GenreConditioningPlugin()

# model = DiffusionWrapper(diffusion_model, genre_embedder)

model.cuda()
encoder.cuda()
encoder.eval()
upscaler.cuda()
upscaler.eval()
genre_embedder.cuda()

data_folder = "../data"

audio_files = aggregate_wavs([f"{data_folder}/Classical", f"{data_folder}/Electronic", f"{data_folder}/Hip Hop", f"{data_folder}/Jazz", f"{data_folder}/Metal", f"{data_folder}/Pop", f"{data_folder}/R&B", f"{data_folder}/Rock"])
X_train, X_test = train_test_split(audio_files, train_size=0.8, random_state=42)

training_params = {
    "batch_size": 68,
    "learning_rate": 1e-4,
    "dataset_multiplier": 512,
    "dataloader_num_workers": 4,
    "dataloader_pin_mem": False,
    "mixed_precision": True,
    "cfg_rate": 0.1,
    "save_path": ["../models/diffuser.pt", "../models/genre_embedder.pt"]
}

train_ds = AudioFileDataset(X_train, sequence_length, multiplier=training_params["dataset_multiplier"])
val_ds = AudioFileDataset(X_test, sequence_length, multiplier=training_params["dataset_multiplier"])
train_dl = DataLoader(train_ds, batch_size=training_params["batch_size"], shuffle=True, num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])
val_dl = DataLoader(val_ds, batch_size=training_params["batch_size"], num_workers=training_params["dataloader_num_workers"], pin_memory=training_params["dataloader_pin_mem"])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
# print(model)

checkpoint = None
train(model, encoder, upscaler, genre_embedder, train_dl, val_dl, lr=training_params["learning_rate"], cfg_rate=training_params["cfg_rate"], 
      mixed_precision=training_params["mixed_precision"], checkpoint=checkpoint, save_path=training_params["save_path"])

# real_eval(model, encoder, upscaler, 0)