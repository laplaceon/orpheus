import pprint
import torch
from torch_audiomentations import SomeOf, OneOf, PolarityInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter
import torch.nn.functional as F


# Initialize augmentation callable
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
)

augments_map = {
    0: {"name": "polarity_inversion"},
    1: {"name": "noise"},
    2: {"name": "volume_mod"},
    3: {"name": "frequency_modulation", "children": [0, 1]}
}

# torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_device = "cpu"

bs = 8
neg_ratio = 0.2

# Make an example tensor with white noise.
# This tensor represents 8 audio snippets with 2 channels (stereo) and 2 s of 16 kHz audio.
audio_samples_1 = torch.rand(size=(bs, 2, 32000), dtype=torch.float32, device=torch_device) - 0.5
audio_samples_2 = torch.rand(size=(bs, 2, 32000), dtype=torch.float32, device=torch_device) - 0.5

def augment_to_label(augments, bs, neg_indices):
    labels = torch.zeros((bs, 5), dtype=torch.int)

    tp = augments.transforms[0].transform_parameters
    if len(tp) != 0:
        # print("polarity inversion applied")
        labels[:, 1][tp["should_apply"]] = 1
    
    tp = augments.transforms[1].transform_parameters
    if len(tp) != 0:
        # print("noise added")
        labels[:, 2][tp["should_apply"]] = 1

    tp = augments.transforms[2].transform_parameters
    if len(tp) != 0:
        # print("volume modulation applied")
        labels[:, 3][tp["should_apply"]] = 1

    freq_transforms = augments.transforms[3].transforms

    for transform in freq_transforms:
        tp = transform.transform_parameters
        if len(tp) != 0:
            # print("frequency modulation applied")
            labels[:, 4][tp["should_apply"]] = 1

            break
    
    print(labels)
    print(neg_indices)

    labels[neg_indices] = torch.zeros((5,), dtype=torch.int)
    labels[neg_indices, 0] = 1
    
    return labels


# Apply augmentation. This varies the gain and polarity of (some of)
# the audio snippets in the batch independently.

orig_samples = audio_samples_1.clone()

num_negatives = 1

indices = torch.randint(0, bs, size=(num_negatives,))
audio_samples_1[indices] = audio_samples_2[indices]

perturbed_audio_samples = apply_augmentations(audio_samples_1, sample_rate=16000)
aug_labels = augment_to_label(apply_augmentations, bs, indices)

loss = 2 * F.multilabel_soft_margin_loss(torch.randn(bs, 5), aug_labels)
print(loss)