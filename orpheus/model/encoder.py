import torch
import torch.nn as nn

from orpheus.model.blocks import MBConv
from orpheus.model.quantizer import SQEmbedding, Jitter

class Encoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        codebook_width,
        h_dims,
        scales,
        blocks_per_stages,
        layers_per_blocks,
        se_ratio=None,
        log_param_q_init=4.60517018599
    ):
        super().__init__()

        # self.pos_emb = FixedPositionalEmbedding(1, sequence_length)

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            sequence_length = int(sequence_length / scales[i])
            encoder_stage = EncoderStage(in_channels, out_channels, 3, scales[i], sequence_length, blocks_per_stages[i], layers_per_blocks[i], se_ratio, last_stage=(i+1 == len(h_dims) - 1))
            stages.append(encoder_stage)

        self.conv = nn.Sequential(*stages)

        self.quantizer = SQEmbedding(codebook_width, h_dims[-1], nn.Parameter(torch.tensor(log_param_q_init)))
        # self.quantizer1 = SQEmbedding(codebook_width // 2, h_dims[-1], nn.Parameter(torch.tensor(log_param_q_init)))
        # self.quantizer2 = SQEmbedding(codebook_width // 2, h_dims[-1], nn.Parameter(torch.tensor(log_param_q_init)))
        # self.jitter = Jitter(0.5)

    def encode(self, x):
        return self.conv(x)

    def forward(self, x, deterministic=False, temperature=0.5):
        x = self.encode(x)

        if deterministic:
            return x, self.quantizer.encode(x)

        z, loss, perplexity = self.quantizer(x, temperature)

        return x, z, loss, perplexity
        # return x

class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        scale,
        seq_len,
        num_blocks,
        layers_per_block,
        se_ratio,
        last_stage=False
    ):
        super().__init__()

        self.downscale = nn.Conv1d(in_channels, out_channels, kernel_size=scale+1, stride=scale, padding=1)

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                EncoderBlock(
                    out_channels,
                    kernel,
                    seq_len,
                    layers_per_block,
                    se_ratio=se_ratio
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.final = MBConv(out_channels, out_channels*2, out_channels, kernel, dilation=1, se_ratio=se_ratio, activation=nn.Tanh()) if last_stage else nn.Identity()

    def forward(self, x):
        out = self.downscale(x)
        out = self.blocks(out)
        return self.final(out)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        seq_len,
        num_layers,
        dilation_factor=2,
        se_ratio=None
    ):
        super().__init__()

        conv = [MBConv(channels, channels*2, channels, kernel, se_ratio=se_ratio)]

        if num_layers is None:
            conv.append(AttentionLayer(channels, seq_len, heads=4, window_len=32))
        else:
            for i in range(num_layers-1):
                dilation = dilation_factor ** (i+1)
                conv.append(MBConv(channels, channels*2, channels, kernel, dilation=dilation, se_ratio=se_ratio))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
