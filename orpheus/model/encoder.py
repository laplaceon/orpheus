import torch
import torch.nn as nn

from .blocks import MBConv, ResBlock

class Encoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        codebook_width,
        h_dims,
        scales,
        blocks_per_stages,
        layers_per_blocks,
        se_ratio=None
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            sequence_length = int(sequence_length / scales[i])
            encoder_stage = EncoderStage(in_channels, out_channels, 3, scales[i], sequence_length, blocks_per_stages[i], layers_per_blocks[i], se_ratio, last_stage=(i+1 == len(h_dims) - 1))
            stages.append(encoder_stage)

        self.conv = nn.Sequential(*stages)

    def forward(self, x):
        return self.conv(x)

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

    def forward(self, x):
        out = self.downscale(x)
        out = self.blocks(out)
        return out

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

        conv = [ResBlock(channels, channels, kernel)]

        for i in range(num_layers):
            dilation = dilation_factor ** (i+1)
            conv.append(ResBlock(channels, channels, kernel, dilation=dilation))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)