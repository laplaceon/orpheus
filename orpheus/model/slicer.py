import torch
import random
from torch import nn
import torch.nn.functional as F
from power_spherical import PowerSpherical

class MPSSlicer(nn.Module):
    def __init__(self, latent_dim, K, L):
        super().__init__()

        assert latent_dim >= K, f"latent dim must be at least the number of mixture components"

        locations = torch.eye(latent_dim)
        kappas = [1., 5., 10., 50.]

        self.locations = nn.ParameterList([locations[i] for i in range(K)])
        # self.scales = nn.ParameterList([torch.tensor(random.choice(kappas)) for _ in range(K)])
        self.scales = nn.ParameterList([torch.tensor(kappas[i % len(kappas)]) for i in range(K)])
        self.weights = nn.Parameter(torch.ones(K,))

        self.L = L
    
    def fgw_dist(self, posterior_samples, prior_samples, p=2, beta=0.1):
        # generate random projections in latent space
        projections = self.sample_slices()
        # calculate projections through the encoded samples
        posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
        prior_projections = prior_samples.matmul(projections)  # batch size x #projections
        posterior_projections = torch.sort(posterior_projections, dim=0)[0]
        prior_projections1 = torch.sort(prior_projections, dim=0)[0]
        prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
        posterior_diff = self.distance_tensor(posterior_projections, posterior_projections, p=p)
        prior_diff1 = self.distance_tensor(prior_projections1, prior_projections1, p=p)
        prior_diff2 = self.distance_tensor(prior_projections2, prior_projections2, p=p)
        # print(posterior_projections.size(), prior_projections1.size())
        # print(posterior_diff.size(), prior_diff1.size())
        w1 = torch.sum((posterior_projections - prior_projections1) ** p, dim=0)
        w2 = torch.sum((posterior_projections - prior_projections2) ** p, dim=0)
        # print(w1.size(), torch.sum(w1))
        gw1 = torch.mean(torch.mean((posterior_diff - prior_diff1) ** p, dim=0), dim=0)
        gw2 = torch.mean(torch.mean((posterior_diff - prior_diff2) ** p, dim=0), dim=0)
        # print(gw1.size(), torch.sum(gw1))
        fgw1 = (1 - beta) * w1 + beta * gw1
        fgw2 = (1 - beta) * w2 + beta * gw2
        return torch.sum(torch.min(fgw1, fgw2))

    def distance_tensor(self, pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
        """
        Returns the matrix of ||x_i-y_j||_p^p.
        :param pts_src: [R, D] matrix
        :param pts_dst: [C, D] matrix
        :param p:
        :return: [R, C, D] distance matrix
        """
        x_col = pts_src.unsqueeze(1)
        y_row = pts_dst.unsqueeze(0)
        distance = torch.abs(x_col - y_row) ** p
        return distance

    def sample_slices(self):
        samples = []

        for location, scale in zip(self.locations, self.scales):
            component = PowerSpherical(loc=F.normalize(location, dim=0), scale=scale)
            samples.append(component.rsample((self.L,)))

        # probs = F.softmax(self.weights, dim=-1)
        probs = F.gumbel_softmax(self.weights)
        weighted_samples = torch.stack(samples).transpose(0, 2) * probs
        weighted_sum = torch.sum(weighted_samples.transpose(0, 2), dim=0)

        return weighted_sum.transpose(0, 1)
    
    def print_parameters(self):
        with torch.no_grad():
            for location, scale in zip(self.locations, self.scales):
                print(torch.sum(location), scale)
            print(self.weights)