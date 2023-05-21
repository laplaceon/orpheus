import torch
from torch import nn

class MoLTranslate(nn.Module):
    def __init__(
        self,
        mol_dim
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(mol_dim, mol_dim, 5, padding=2)
        # self.act = nn.GELU()
        # self.conv2 = nn.Conv2d(mol_dim, mol_dim, 5, padding=2)
    
    def forward(self, x):
        out = self.conv1(x)
        # out = self.act(x)
        # out = self.conv2(x)
        return torch.sigmoid(out)