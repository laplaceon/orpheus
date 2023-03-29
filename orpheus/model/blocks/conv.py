from torch import nn
from torch.nn import functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class CausalTransposedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='causal'):
        super(CausalTransposedConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.register_buffer('mask', self.conv.weight.data.new(*self.conv.weight.size()).zero_())
        self.create_mask()

    def create_mask(self):
        k = self.kernel_size
        self.mask[:, :, :k // 2] = 1
        if k % 2 == 0:
            self.mask[:, :, k // 2] = 0

        if self.padding == 'causal':
            self.mask[:, :, -1] = 0

    def forward(self, x):
        self.conv.weight.data *= self.mask
        output_padding = self.compute_output_padding(x)
        x = self.conv(x, output_padding=output_padding)
        return x[:, :, : -self.dilation * (self.kernel_size - 1) - 1]

    def compute_output_padding(self, x):
        if self.padding == 'causal':
            return (self.stride - x.size(-1) % self.stride) % self.stride
        else:
            return 0

# class CausalConvTranspose1d(nn.ConvTranspose1d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
#         self._padding = (kernel_size - 1) * dilation

#     def forward(self, input):
#         # Add padding to the left side of the input
#         input = nn.functional.pad(input, (self._padding, 0))
#         # Compute the convolution
#         output = super().forward(input)
#         # Remove the padded values from the output
#         return output[:, :, :-self._padding]

class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        x = F.pad(x, (self.padding, 0))
        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out