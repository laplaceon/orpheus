from torch import nn

from .norm import RMSNorm
from .mem_attn import Attention

class AttnBlock(nn.Module):
    def __init__(
        self,
        channels,
        bucket = 2048,
        num_heads = 4,
        expansion_factor_feedforward = 1.5,
        bias_feedforward = False,
        dropout = 0
    ):
        super().__init__()

        dim_head = channels // num_heads

        self.attn = nn.Sequential(
            # Rearrange('b c l -> b l c'),
            RMSNorm(channels),
            # Rearrange('b l c -> b c l'),
            # nn.LayerNorm(channels),
            # PosEncodingC(channels),
            Attention(
                dim = channels,
                dim_head = dim_head,
                heads = num_heads,
                causal = False,
                memory_efficient = True,
                q_bucket_size = bucket // 2,
                k_bucket_size = bucket,
                dropout = dropout
            )
        )

        dim_feedforward = int(channels * expansion_factor_feedforward)

        self.ff = nn.Sequential(
            RMSNorm(channels),
            nn.Linear(channels, dim_feedforward, bias=bias_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, channels, bias=bias_feedforward)
        )

        # self.attn = AttentionLayers(channels, 1, heads=num_heads, rel_pos_bias=True, use_rms_norm=True)
    
    def forward(self, x):
        x = x.transpose(1, 2)

        x = x + self.attn(x)
        x = x + self.ff(x)

        return x.transpose(1, 2)