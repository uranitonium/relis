"""Boundary evidence feature construction."""

from __future__ import annotations

import torch

from features.rolling import rolling_shift, rolling_var


def compute_phi(
    z: torch.Tensor,
    slow: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    critical_ratio: torch.Tensor | None = None,
    use_critical: bool = False,
) -> torch.Tensor:
    """Compute phi features.

    sign(0) is defined as 0 via torch.sign.
    Window uses shift by k frames (phi[t] compares t and t-k).
    """
    bsz, t, _ = z.shape
    mask_f = mask.float()

    # delta_z
    z_prev = torch.zeros_like(z)
    z_prev[:, 1:] = z[:, :-1]
    prev_mask = torch.zeros_like(mask)
    prev_mask[:, 1:] = mask[:, :-1]
    valid_pair = mask & prev_mask
    dz = torch.norm(z - z_prev, dim=-1)
    dz = dz * valid_pair.float()

    # rolling variance
    var_t = rolling_var(z, mask, k)

    feats = [dz, var_t]
    m_shift = rolling_shift(mask_f, k)
    valid_k = (mask_f > 0.5) & (m_shift > 0.5)

    slow_dim = int(slow.shape[-1])
    if slow_dim >= 1:
        d = slow[..., 0]
        d_shift = rolling_shift(d, k)
        feats.append((d - d_shift).abs() * valid_k.float())
    if slow_dim >= 2:
        i = slow[..., 1]
        i_shift = rolling_shift(i, k)
        feats.append((i - i_shift).abs() * valid_k.float())
    if slow_dim >= 3:
        c = slow[..., 2]
        c_shift = rolling_shift(c, k)
        sign_c = torch.sign(c)
        sign_c_shift = torch.sign(c_shift)
        feats.append((sign_c != sign_c_shift).float() * valid_k.float())
    if slow_dim >= 4:
        u3 = slow[..., 3]
        u3_shift = rolling_shift(u3, k)
        feats.append((u3 - u3_shift).abs() * valid_k.float())
    if slow_dim >= 5:
        u4 = slow[..., 4]
        u4_shift = rolling_shift(u4, k)
        sign_u4 = torch.sign(u4)
        sign_u4_shift = torch.sign(u4_shift)
        feats.append((sign_u4 != sign_u4_shift).float() * valid_k.float())
    if use_critical and critical_ratio is not None:
        crit = critical_ratio
        crit_shift = rolling_shift(crit, k)
        m_shift = rolling_shift(mask_f, k)
        valid_k = (mask_f > 0.5) & (m_shift > 0.5)
        dcrit = (crit - crit_shift).abs() * valid_k.float()
        sign_c = torch.sign(1.0 - crit)
        sign_c_shift = torch.sign(1.0 - crit_shift)
        flip_crit = (sign_c != sign_c_shift).float() * valid_k.float()
        feats.extend([crit, dcrit, flip_crit])
    phi = torch.stack(feats, dim=-1)
    phi = phi * mask_f.unsqueeze(-1)
    return phi
