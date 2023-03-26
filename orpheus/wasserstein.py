import torch

def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
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


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def sliced_fgw_distance(posterior_samples, prior_samples, num_projections=50, p=2, beta=0.1, device='cpu'):
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = prior_samples.size(1)
    # generate random projections in latent space
    projections = torch.randn(size=(embedding_dim, num_projections)).to(device)
    # calculate projections through the encoded samples
    posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
    prior_projections = prior_samples.matmul(projections)  # batch size x #projections
    posterior_projections = torch.sort(posterior_projections, dim=0)[0]
    prior_projections1 = torch.sort(prior_projections, dim=0)[0]
    prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
    posterior_diff = distance_tensor(posterior_projections, posterior_projections, p=p)
    prior_diff1 = distance_tensor(prior_projections1, prior_projections1, p=p)
    prior_diff2 = distance_tensor(prior_projections2, prior_projections2, p=p)
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


def sliced_gw_distance(posterior_samples, prior_samples, num_projections=50, p=2, device='cpu'):
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = prior_samples.size(1)
    # generate random projections in latent space
    projections = torch.randn(size=(embedding_dim, num_projections)).to(device)
    # calculate projections through the encoded samples
    posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
    prior_projections = prior_samples.matmul(projections)  # batch size x #projections
    posterior_projections = torch.sort(posterior_projections, dim=0)[0]
    prior_projections1 = torch.sort(prior_projections, dim=0)[0]
    prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
    posterior_diff = distance_tensor(posterior_projections, posterior_projections, p=p)
    prior_diff1 = distance_tensor(prior_projections1, prior_projections1, p=p)
    prior_diff2 = distance_tensor(prior_projections2, prior_projections2, p=p)

    out1 = torch.sum(torch.sum((posterior_diff - prior_diff1) ** p, dim=0), dim=1)
    out2 = torch.sum(torch.sum((posterior_diff - prior_diff2) ** p, dim=0), dim=1)
    return torch.sum(torch.min(out1, out2))


def sliced_wasserstein_distance(encoded_samples, num_projections=50, p=2, device='cpu'):
    """
    Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    # print(encoded_samples.size())
    embedding_dim = encoded_samples.size(1)
    distribution_samples = torch.randn(size=encoded_samples.size()).to(device)
    # generate random projections in latent space
    projections = torch.randn(size=(num_projections, embedding_dim)).to(device)
    # print(projections.size())
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.sum()