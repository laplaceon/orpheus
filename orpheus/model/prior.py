import torch
import random
from torch import nn
import torch.nn.functional as F
from power_spherical import PowerSpherical
from torch.distributions import Normal

class PsPrior(nn.Module):
    def __init__(self, latent_dim, K):
        super().__init__()

        assert latent_dim >= K, f"latent dim must be at least the number of mixture components"

        locations = torch.eye(latent_dim)
        kappas = [1., 5., 10., 50., 100.]

        self.locations = nn.ParameterList([locations[i] for i in range(K)])
        self.scales = nn.ParameterList([torch.tensor(random.choice(kappas)) for _ in range(K)])
        self.weights = nn.Parameter(torch.ones(K,))
    
    def sample(self, n):
        samples = []

        for location, scale in zip(self.locations, self.scales):
            component = PowerSpherical(loc=F.normalize(location, dim=0), scale=scale)
            samples.append(component.rsample((n,)))

        # probs = F.softmax(self.weights, dim=-1)
        probs = F.gumbel_softmax(self.weights)
        weighted_samples = torch.stack(samples).transpose(0, 2) * probs
        weighted_sum = torch.sum(weighted_samples.transpose(0, 2), dim=0)

        return weighted_sum
    
    def print(self):
        with torch.no_grad():
            for location, scale in zip(self.locations, self.scales):
                print(torch.sum(location), scale)
            print(self.weights)

class GaussianPrior(nn.Module):
    def __init__(self, latent_dim, K):
        super().__init__()

        self.locs = nn.ParameterList([torch.randn(latent_dim) for _ in range(K)])
        self.scale = nn.ParameterList([torch.rand(latent_dim) for _ in range(K)])
        self.weights = nn.Parameter(torch.ones(K,))

    def sample(self, n):
        samples = []

        for mu, scale in zip(self.locs, self.scale):
            component = Normal(2 * F.sigmoid(mu) - 1, F.sigmoid(scale))
            samples.append(component.rsample((n,)))
        
        probs = F.gumbel_softmax(self.weights)
        weighted_samples = torch.stack(samples).transpose(0, 2) * probs
        weighted_sum = torch.sum(weighted_samples.transpose(0, 2), dim=0)

        return weighted_sum
    
    def print_parameters(self):
        print(self.weights)

gmm = GaussianPrior(128, 1)