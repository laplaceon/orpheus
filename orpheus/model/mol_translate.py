import torch
from torch import nn

class MoLTranslate(nn.Module):
    def __init__(
        self,
        scales,
        dims
    ):
        super().__init__()

        translators = {}

        for scale, dim in zip(scales, dims):
            hidden_dim = dim * 3
            out_dim = dim * 2

            translators[str(scale)] = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim * 2, 7, padding=3),
                nn.GELU(),
                nn.Conv1d(hidden_dim * 2, out_dim, 7, padding=3)
            )
        
        self.translators = nn.ModuleDict(translators)

    def forward(self, scale, y_mean, y_scale, y_weighted):
        combined = torch.cat([y_mean, y_scale, y_weighted], dim=1)
        translated = self.translators[str(scale)](combined)

        y_scale_translated, y_weight_translated = translated.chunk(2, dim=1)

        # return torch.sigmoid(y_scale_translated), torch.softmax(y_weight_translated, dim=1)
        return y_scale_translated, y_weight_translated