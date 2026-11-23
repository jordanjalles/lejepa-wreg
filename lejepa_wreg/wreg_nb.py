"""VERBATIM copy of the user's Wasserstein regularizers. Do not edit: run tools/extract_wreg.py.

source: <data-root>/workspaces/Jupyter_AI/MaskedAutoencoders/ImageNet-AE-selfteach.ipynb
  cell  25  sliced_wasserstein_distance              sha256 864bd9004ec21251
  cell  26  basis_wasserstein_distance               sha256 b0ca3a71edc82f02
  cell  27  token_sliced_wasserstein_distance        sha256 cff8697c07654005
  cell  28  channel_sliced_wasserstein_distance      sha256 e74bb8eaf3a91046
"""
import torch
import torch.nn.functional as F


def sliced_wasserstein_distance(
    z,
    prior,
    n_slices=1024,
    bulk_sigma=1.0,       # smaller = stronger tail discount; None = unweighted
):
    assert z.shape == prior.shape
    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    z = z.flatten(1)                         # (B, D)
    prior = prior.detach().flatten(1)        # fixed target
    B, D = z.shape

    slices = F.normalize(
        torch.randn(D, n_slices, device=z.device, dtype=z.dtype),
        dim=0,
    )                                       # (D, S)

    z_slices = torch.sort(z @ slices, dim=0).values.float()
    p_slices = torch.sort(prior @ slices, dim=0).values.float()

    squared_error = (z_slices - p_slices).square()

    if bulk_sigma is None:
        loss = squared_error.mean()
    else:
        log_weights = -0.5 * (p_slices / bulk_sigma).square()

        # Normalize across batch ranks, separately for each slice.
        # Softmax stably computes exp(log_weights) / sum(exp(log_weights)).
        weights = torch.softmax(log_weights, dim=0)

        loss = (weights * squared_error).sum(dim=0).mean()

     
    return loss.square()


def basis_wasserstein_distance(z, prior, bulk_sigma=1.0):
    assert z.shape == prior.shape
    assert z.ndim == 3, "Expected z shaped (B, T, C)"

    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    prior = prior.detach()

    def axis_loss(source, target, dim):
        source = torch.sort(source, dim=dim).values.float()
        target = torch.sort(target, dim=dim).values.float()

        squared_error = (source - target).square()

        if bulk_sigma is None:
            return squared_error.mean()

        log_weights = -0.5 * (target / bulk_sigma).square()
        weights = torch.softmax(log_weights, dim=dim)

        return (weights * squared_error).sum(dim=dim).mean()

    loss_b = axis_loss(z, prior, dim=0)   # across images, per (t, c)
    loss_t = axis_loss(z, prior, dim=1)   # across tokens, per (b, c)
    loss_c = axis_loss(z, prior, dim=2)   # across channels, per (b, t)
    loss_e = axis_loss(
        z.flatten(1), prior.flatten(1), dim=1
    )                                    # across T*C, per image

    loss = loss_b # + 0.1 * (loss_t + loss_c + loss_e) / 3
    
 
    return loss.square()


def token_sliced_wasserstein_distance(
    z,
    prior,
    n_slices=32,
    bulk_sigma=1.0,
):
    """
    Token-space SWD.

    z/prior: (B, T, C)

    Reshapes to (B*C, T), so each channel across the token sequence
    is one sample, and random slices operate across the T token dimensions.

    This directly pressures token combinations/correlations
    toward the isotropic Gaussian prior.
    """
    assert z.shape == prior.shape
    assert z.ndim == 3, "Expected z shaped (B, T, C)"

    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    B, T, C = z.shape

    # Each channel's values across T tokens becomes one T-dimensional observation.
    z = z.permute(0, 2, 1).reshape(B * C, T)
    prior = prior.detach().permute(0, 2, 1).reshape(B * C, T)

    # Random directions through token space.
    slices = F.normalize(
        torch.randn(
            T,
            n_slices,
            device=z.device,
            dtype=z.dtype,
        ),
        dim=0,
    )  # (T, S)

    # Project each observation onto random token-space directions.
    z_slices = torch.sort(
        z @ slices,
        dim=0,
    ).values.float()

    p_slices = torch.sort(
        prior @ slices,
        dim=0,
    ).values.float()

    squared_error = (z_slices - p_slices).square()

    if bulk_sigma is None:
        loss = squared_error.mean()

    else:
        log_weights = -0.5 * (p_slices / bulk_sigma).square()

        # Normalize across the B*C observations for each slice.
        weights = torch.softmax(
            log_weights,
            dim=0,
        )

        loss = (
            weights * squared_error
        ).sum(dim=0).mean()

    return loss.square()


def channel_sliced_wasserstein_distance(
    z,
    prior,
    n_slices=32,
    bulk_sigma=1.0,
):
    """
    Channel-space SWD.

    z/prior: (B, T, C)

    Reshapes to (B*T, C), so each token embedding is one sample
    and random slices operate across the C channel dimensions.

    This directly pressures channel combinations/correlations
    toward the isotropic Gaussian prior.
    """
    assert z.shape == prior.shape
    assert z.ndim == 3, "Expected z shaped (B, T, C)"

    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    B, T, C = z.shape

    # Each token embedding becomes one C-dimensional observation.
    z = z.reshape(B * T, C)
    prior = prior.detach().reshape(B * T, C)

    # Random directions through channel space.
    slices = F.normalize(
        torch.randn(
            C,
            n_slices,
            device=z.device,
            dtype=z.dtype,
        ),
        dim=0,
    )  # (C, S)

    # Project each token embedding onto each random channel direction.
    z_slices = torch.sort(
        z @ slices,
        dim=0,
    ).values.float()

    p_slices = torch.sort(
        prior @ slices,
        dim=0,
    ).values.float()

    squared_error = (z_slices - p_slices).square()

    if bulk_sigma is None:
        loss = squared_error.mean()

    else:
        log_weights = -0.5 * (p_slices / bulk_sigma).square()

        # Normalize across the B*T observations for each slice.
        weights = torch.softmax(
            log_weights,
            dim=0,
        )

        loss = (
            weights * squared_error
        ).sum(dim=0).mean()

     
    return loss.square()
