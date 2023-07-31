import torch

def min_max_scale(x: torch.Tensor):
    min, _ = x.min(1, keepdim=True)
    max, _ = x.max(1, keepdim=True)

    range = max - min

    return (x - min) / (range + 1e-8)


# x1 = torch.load("../x_spec.pt")
# print(x)

# x1 = torch.tensor(((1, 2, 2), (3, 3.5, 4)))
# print(x1.shape[0], "batches")
# print(x1)
# print("min", x1.min().item(), "max", x1.max().item())
# x1_mod = min_max_scale(x1)
# print(x1_mod)
# for i, x in enumerate(x1_mod):
#     if x.min().isnan():
#         print(x1[i].min(), x1[i].max())
#         print(x.min(), x.max())
# print("min", x1_mod.min().item(), "max", x1_mod.max().item())