import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class SQEmbedding(nn.Module):
    def __init__(
            self, 
            n_embeddings, 
            embedding_dim, 
            param_var_q = "gaussian_1"
    ):
        super().__init__()
        self.param_var_q = param_var_q

        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.normal_()
        self.register_parameter("embedding", nn.Parameter(embedding))

    def encode(self, x, log_var_q):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        if self.param_var_q == "gaussian_1":
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == "gaussian_3":
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == "gaussian_4":
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception("Undefined param_var_q")

        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * ((embedding - x_flat) ** 2), dim=1)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x, log_var_q, temperature):
        M, D = self.embedding.size()
        batch_size, sample_size, channels = x.size()
        x_flat = x.reshape(-1, D)
        if self.param_var_q == "gaussian_1":
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == "gaussian_3":
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == "gaussian_4":
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception("Undefined param_var_q")

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

        precision = torch.exp(-log_var_q)
        loss = torch.mean(0.5 * torch.sum(precision * (x - quantized) ** 2, dim=(1, 2))
                          + torch.sum(probabilities * log_probabilities, dim=(1, 2)))

        encodings = F.one_hot(indices, M).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

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

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply