"""
Semi-Lagrangian extrapolation operator.

This module holds a single shared implementation of the iterative
`grid_sample`-based semi-Lagrangian warp used by every stage of LUPIN
(Pavlik et al., 2025): the MF-U-Net stage (`ainpp_pb_latam.models.mfunet`)
and the AF-U-Net / joint stages (`ainpp_pb_latam.models.lupin`).

It is intentionally kept dependency-free (only torch) and side-effect-free
so it can be imported by any model package without risking circular imports
between `models.mfunet` and `models.lupin`.

Note
----
`ainpp_pb_latam.models.mfunet.forecaster.MFUNetForecaster` already contains
its own copy of this routine (added before this shared utility existed).
It is left untouched on purpose, to respect the "change existing files as
little as possible" constraint for this integration. `models.lupin` is the
only consumer of the shared version below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def semi_lagrangian_extrapolate(
    timesteps: int, precip: torch.Tensor, motion_field: torch.Tensor
) -> torch.Tensor:
    """
    Semi-Lagrangian extrapolation of `precip` along `motion_field`, iterated
    forward for `timesteps` steps via repeated `F.grid_sample` warping.

    This is the same backward-trajectory integration scheme used by the
    reference LUPIN implementation: at each step the *velocity itself* is
    resampled at the currently accumulated displacement, so the field is
    advected along curved (not just straight-line) trajectories when the
    motion field is spatially non-uniform.

    Parameters
    ----------
    timesteps:
        Number of forward extrapolation steps to produce.
    precip:
        Field to be warped, shape (B, C, H, W). C is typically 1.
    motion_field:
        Estimated motion field (u, v), shape (B, 2, H, W), in pixels per
        step over a domain normalized to [-1, 1] by `grid_sample` convention.

    Returns
    -------
    torch.Tensor
        Extrapolated field(s), shape (B, timesteps, H, W).
    """
    velocity = motion_field / (motion_field.shape[-1] / 2)

    x_values, y_values = torch.meshgrid(
        torch.arange(velocity.shape[-2], device=motion_field.device),
        torch.arange(velocity.shape[-1], device=motion_field.device),
        indexing="ij",
    )
    xy_coords = torch.stack([y_values, x_values]).to(precip.device)
    xy_coords = (xy_coords) / ((velocity.shape[-1]) / 2) - 1  # assumes square input

    precip_extrap = torch.zeros(
        (precip.shape[0], timesteps, precip.shape[2], precip.shape[3]),
        device=precip.device,
        dtype=precip.dtype,
    )
    displacement = torch.zeros(
        (velocity.shape[0], 2, velocity.shape[2], velocity.shape[3]),
        device=precip.device,
        dtype=precip.dtype,
    )
    velocity_inc = velocity.clone()

    for ti in range(timesteps):
        coords_warped = xy_coords.unsqueeze(0) + displacement
        velocity_inc = F.grid_sample(
            velocity,
            coords_warped.movedim(1, -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        displacement = displacement - velocity_inc
        coords_warped = xy_coords.unsqueeze(0) + displacement
        precip_warped = F.grid_sample(
            precip,
            coords_warped.movedim(1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        precip_extrap[:, ti : ti + 1] = precip_warped

    return precip_extrap
