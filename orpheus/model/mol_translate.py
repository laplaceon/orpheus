import torch
from torch import nn

class MoLTranslate(nn.Module):
    def __init__(
        self,
        scales,
        dim
    ):
        super().__init__()

        var_translate = {}

        for scale in scales:
            var_translate[str(scale)] = nn.Sequential(
                nn.Conv1d(dim, dim * 2, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(dim * 2, dim, 5, padding=2),
                nn.Sigmoid()
            )
        
        self.var_translate = nn.ModuleDict(var_translate)

    def forward(self, var_spec, y_specscale):
        # print(var_spec.shape, y_specscale)
        var_translated = self.var_translate[str(y_specscale)](var_spec)

        return var_translated