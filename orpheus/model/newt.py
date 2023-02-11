import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma * x + beta

class TimeDistributedLayerNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        return self.layer_norm(x.transpose(1, 2)).transpose(1, 2)

class TimeDistributedMLP(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, depth: int = 3):
        super().__init__()
        assert depth >= 3, "Depth must be at least 3"
        layers = []
        for i in range(depth):
            layers.append(
                nn.Conv1d(
                    in_size if i == 0 else hidden_size,
                    hidden_size if i < depth - 1 else out_size,
                    1,
                )
            )
            if i < depth - 1:
                layers.append(TimeDistributedLayerNorm(hidden_size))
                layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Sine(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.sin(x)

class TrainableNonlinearity(nn.Module):
    def __init__(
        self, channels, width, nonlinearity=nn.ReLU, final_nonlinearity=Sine, depth=3
    ):
        super().__init__()
        self.input_scale = nn.Parameter(torch.randn(1, channels, 1) * 10)
        layers = []
        for i in range(depth):
            layers.append(
                nn.Conv1d(
                    channels if i == 0 else channels * width,
                    channels * width if i < depth - 1 else channels,
                    1,
                    groups=channels,
                )
            )
            layers.append(nonlinearity() if i < depth - 1 else final_nonlinearity())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.input_scale * x)

class NEWT(nn.Module):
    def __init__(
        self,
        n_waveshapers: int,
        control_embedding_size: int,
        shaping_fn_size: int = 16,
        out_channels: int = 1,
    ):
        super().__init__()

        self.n_waveshapers = n_waveshapers

        self.mlp = TimeDistributedMLP(
            control_embedding_size, control_embedding_size, n_waveshapers * 4, depth=4
        )

        self.waveshaping_index = FiLM()
        self.shaping_fn = TrainableNonlinearity(
            n_waveshapers, shaping_fn_size, nonlinearity=Sine
        )
        self.normalising_coeff = FiLM()

        self.mixer = nn.Sequential(
            nn.Conv1d(n_waveshapers, out_channels, 1),
        )

    def forward(self, exciter, control_embedding):
        film_params = self.mlp(control_embedding)
        film_params = F.interpolate(film_params, exciter.shape[-1], mode="linear")
        gamma_index, beta_index, gamma_norm, beta_norm = torch.split(
            film_params, self.n_waveshapers, 1
        )

        # print(film_params.shape, exciter.shape, gamma_index.shape, beta_index.shape, gamma_norm.shape, beta_norm.shape)

        x = self.waveshaping_index(exciter, gamma_index, beta_index)
        x = self.shaping_fn(x)
        x = self.normalising_coeff(x, gamma_norm, beta_norm)

        # return x
        return self.mixer(F.interpolate(x, scale_factor=4))
