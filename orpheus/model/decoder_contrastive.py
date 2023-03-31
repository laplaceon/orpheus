import torch
from torch import nn

class ContrastiveDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        kernel = 7,
        num_classes = 2
    ):
        super().__init__()

        self.mixer = nn.Sequential(
            nn.Conv1d(latent_dim * 2, latent_dim, kernel, padding=kernel // 2, bias=False),
            nn.GELU()
        )

        hidden_dim = latent_dim * 2

        self.fe = nn.Sequential(
            nn.GroupNorm(8, latent_dim),
            nn.Conv1d(latent_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel//2, groups=hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, latent_dim, 1)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z1, z2):
        z_p = torch.cat([z1, z2], dim=1)
        z = self.mixer(z_p)

        z = z + self.fe(z)
        
        return self.fc(self.pool(z).squeeze(2))

bs = 4
latent_size = 128
seq_len = 64

z1 = torch.rand(bs, latent_size, seq_len)
z2 = torch.rand(bs, latent_size, seq_len)

decoder_contrastive = ContrastiveDecoder(latent_size, num_classes=5)
out = decoder_contrastive(z1, z2)
print(out.shape)