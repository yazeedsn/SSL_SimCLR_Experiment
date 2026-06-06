import torch 
import torch.nn.functional as F

def nt_xent(x, y, temperature=0.1):
    assert x.shape == y.shape
    assert len(x.size()) == 2

    n = x.shape[0]
    z = torch.cat([x, y])
    cs = F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)

    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    cs = cs.masked_fill(mask, float('-inf'))
    target = torch.roll(torch.arange(2 * n, device=z.device), shifts=n)

    return F.cross_entropy(cs/temperature, target, reduction="mean")