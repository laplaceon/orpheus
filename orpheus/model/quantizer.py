import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            return x
        else:
            x = x.transpose(1, 2)

            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x.transpose(1, 2)

class SQEmbedding(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, log_param_q):
        super(SQEmbedding, self).__init__()

        embedding = torch.randn(n_embeddings, embedding_dim)
        embedding.normal_()
        self.embedding = nn.Parameter(embedding)

        self.log_var_q = log_param_q

    def encode(self, x):
        x = x.transpose(1, 2)

        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        log_var_q_flat = self.log_var_q.reshape(1, 1)

        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * ((embedding - x_flat) ** 2), dim=1)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized.transpose(1, 2), indices

    def forward(self, x, temperature):
        x = x.transpose(1, 2)

        M, D = self.embedding.size()
        batch_size, sample_size, channels = x.size()
        x_flat = x.reshape(-1, D)
        log_var_q_flat = self.log_var_q.reshape(1, 1)

        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * (embedding - x_flat) ** 2, dim=1)

        indices = torch.argmin(distances.float(), dim=-1)

        logits = -distances

        encodings = self._gumbel_softmax(logits, tau=temperature, dim=-1)
        quantized = torch.matmul(encodings, self.embedding)
        quantized = quantized.view_as(x)

        logits = logits.view(batch_size, sample_size, M)
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)

        precision = torch.exp(-self.log_var_q)
        loss = torch.mean(0.5 * torch.sum(precision * (x - quantized) ** 2, dim=(1, 2))
                          + torch.sum(probabilities * log_probabilities, dim=(1, 2)))

        encodings = F.one_hot(indices, M).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized.transpose(1, 2), loss, perplexity

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, dim=-1):
        eps = torch.finfo(logits.dtype).eps
        gumbels = (
            -((-(torch.rand_like(logits).clamp(min=eps, max=1 - eps).log())).log())
        )  # ~Gumbel(0,1)
        gumbels_new = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels_new.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparameterization trick.
            ret = y_soft

        return ret
