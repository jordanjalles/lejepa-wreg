"""Frozen linear probe, collapse checks and latent drift. Nothing here can update the encoder:
features come from eval mode under inference_mode, and the probe sees only those tensors."""
import torch
import torch.nn as nn
import torch.nn.functional as F

PROBE = dict(epochs=100, batch=256, lr=1e-3, wd=1e-4, seed=0)   # identical for every arm and run


@torch.inference_mode()
def extract(net, loader, device):
    """(native (n, 384), A's 512-d map output or None, A's projector output or None, labels) in eval mode."""
    was = net.training
    net.eval()
    nat, emb, prj, ys = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            native, z, e = net(x)                     # x: (n, 1, 3, H, W)
        nat.append(native[0].float()); ys.append(y.to(device))
        if e is not None:
            emb.append(e.float()); prj.append(z[0].float())
    net.train(was)
    cat = lambda t: torch.cat(t) if t else None
    return cat(nat), cat(emb), cat(prj), torch.cat(ys)


def linear_probe(ftr, ytr, fev, yev, n_classes=10):
    """Top-1 of a linear classifier on features standardized with train statistics."""
    mu, sd = ftr.mean(0), ftr.std(0).clamp_min(1e-6)
    ftr, fev = (ftr - mu) / sd, (fev - mu) / sd
    g = torch.Generator(device=ftr.device).manual_seed(PROBE['seed'])
    with torch.enable_grad():
        lin = nn.Linear(ftr.shape[1], n_classes).to(ftr.device)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g, device=ftr.device) * 0.01)
            lin.bias.zero_()
        opt = torch.optim.AdamW(lin.parameters(), lr=PROBE['lr'], weight_decay=PROBE['wd'])
        steps = PROBE['epochs'] * ((len(ftr) + PROBE['batch'] - 1) // PROBE['batch'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(PROBE['epochs']):
            perm = torch.randperm(len(ftr), generator=g, device=ftr.device)
            for i in range(0, len(ftr), PROBE['batch']):
                b = perm[i:i + PROBE['batch']]
                loss = F.cross_entropy(lin(ftr[b]), ytr[b])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return (lin(fev).argmax(1) == yev).float().mean().item()


def collapse(f):
    """Per-dimension std (mean, min), covariance effective rank exp(H(eigenvalues / sum)), and each
    embedding's own mean and std over its coordinates (target of B's per-image term: 0 and 1)."""
    sd = f.std(0)
    ev = torch.linalg.eigvalsh(torch.cov(f.T.double())).clamp_min(0)
    p = ev / ev.sum()
    erank = torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item()
    return {'std_mean': sd.mean().item(), 'std_min': sd.min().item(), 'erank': erank,
            'img_mean': f.mean(1).abs().mean().item(), 'img_std': f.std(1).mean().item()}


def drift(prev, cur):
    """Normalized displacement mean||cur - prev|| / mean||prev||, relative change of the pairwise
    distance matrix ||D_cur - D_prev||_F / ||D_prev||_F, and its scale-free version after
    dividing each matrix by its mean."""
    disp = (cur - prev).norm(dim=1).mean() / prev.norm(dim=1).mean()
    dp, dc = torch.cdist(prev, prev), torch.cdist(cur, cur)
    rel = (dc - dp).norm() / dp.norm()
    shape = (dc / dc.mean() - dp / dp.mean()).norm() / (dp / dp.mean()).norm()
    return {'disp': disp.item(), 'pdist': rel.item(), 'pdist_shape': shape.item()}
