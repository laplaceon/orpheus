import torch
from torch import nn
import torch.nn.functional as F

import onnxruntime

from model.rae import Orpheus

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
        y_mb = self.decoder(z)
        y = self.pqmf.inverse(y_mb)
        return y

model = Orpheus(fast_recompose=True)
model.load_state_dict(torch.load("../models/rae_8.pt"))

encoder = Encoder(model.encoder, model.pqmf)
upscaler = Upscaler(model.decoder, model.pqmf)

encoder.eval()
upscaler.eval()

x = 2 * torch.rand(4, 1, 131072) - 1
# x = 2 * torch.rand(4, 16, 262144) - 1

fl = 2 * torch.rand(4, 128, 64) - 1

# z = encoder(x)
# print(z.shape, upscaler(z).shape)

torch.onnx.export(
    encoder,
    x,
    "../models/encoder.onnx",
    input_names = ['audio_in'],
    output_names = ['latent_out'],
    dynamic_axes = {'audio_in' : {0 : 'batch_size', 2: 'audio_length'}, 'latent_out' : {0 : 'batch_size', 2: 'latent_length'}},
    verbose = False,
    do_constant_folding = False
)

torch.onnx.export(
    upscaler,
    fl,
    "../models/upscaler.onnx",
    input_names = ['latent_in'],
    output_names = ['audio_out'],
    dynamic_axes = {'latent_in' : {0 : 'batch_size', 2: 'latent_length'}, 'audio_out' : {0 : 'batch_size', 2: 'audio_length'}},
    verbose = False,
    do_constant_folding = False
)

ort_session_enc = onnxruntime.InferenceSession("../models/encoder.onnx")
ort_session_dec = onnxruntime.InferenceSession("../models/upscaler.onnx")

output_enc = ort_session_enc.run(
    None,
    {"audio_in": x.numpy()},
)

z = output_enc[0]
encoder.pqmf.polyphase = False
upscaler.pqmf.polyphase = False
z_real = encoder(x)

output_dec = ort_session_dec.run(
    None,
    {"latent_in": z},
)

out = output_dec[0]
out_real = upscaler(z_real)

print(z)
print(z_real)
print(out)
print(out_real)
print(z.shape, z_real.shape, F.mse_loss(torch.tensor(z), z_real))
print(out.shape, out_real.shape, F.mse_loss(torch.tensor(out), out_real))