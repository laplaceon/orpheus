from torch import nn

class Encoder(nn.Module):
    def __init__(
        self,
        encoder,
        pqmf
    ):
        super().__init__()

        self.encoder = encoder
        self.pqmf = pqmf
    
    def forward(self, x):
        x_mb = self.pqmf(x)
        z = self.encoder(x_mb)
        return z

class Upscaler(nn.Module):
    def __init__(
        self,
        decoder,
        pqmf
    ):
        super().__init__()

        self.decoder = decoder
        self.pqmf = pqmf
    
    def forward(self, z):
        y_mb, _ = self.decoder(z)
        y = self.pqmf.inverse(y_mb)
        return y