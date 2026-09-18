from __future__ import annotations

import logging
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ainpp_pb_latam.models.mfunet.backbone import MFUNetBackbone

logger = logging.getLogger(__name__)

class BaseForecaster(nn.Module):
    """
    Base class for forecasters providing common utility methods.
    """

    def _apply_nonnegativity(self, x: torch.Tensor, mode: str = "relu") -> torch.Tensor:
        """
        Applies a non-linearity to enforce non-negative precipitation values.

        Args:
            x (torch.Tensor): Input tensor.
            mode (str): Activation mode ('relu', 'softplus', or 'none').

        Returns:
            torch.Tensor: Tensor with non-negative values.
        """
        if mode == "relu":
            return F.relu(x)
        if mode == "softplus":
            return F.softplus(x)
        return x

class MFUNetForecaster(BaseForecaster):

    """
    Motion Field U-Net (MF-U-Net) forecaster.

    Estimates a 2-channel motion field (u, v) from the input sequence via a U-Net
    backbone, then produces each future frame by semi-Lagrangian extrapolation
    (warping) of the last observed frame along that motion field. The process is
    repeated autoregressively for `output_timesteps` steps, sliding the input
    window forward with each newly predicted frame (following the same pattern as
    `UNetAutoRegressive`).

    Parameters
    ----------
    input_timesteps:
        Number of input time steps (Tin).
    input_channels:
        Number of channels per input frame (Cin). The precipitation field used in
        this benchmark has input_channels=1.
    output_timesteps:
        Number of future frames to forecast (Tout).
    features:
        Number of channels at each encoder/decoder level of the backbone.
    kernel_size:
        Convolution kernel size used throughout the backbone.
    bilinear:
        If True, uses bilinear upsampling in the decoder; otherwise transposed convs.
    nonnegativity:
        Non-linearity applied to the final precipitation forecast to enforce physically
        valid (non-negative) values. One of "relu", "softplus", "none".
    return_motion_field:
        If True, `forward` returns a tuple `(precip, motion_field)` instead of just
        `precip`. This is required to train with a physics-informed loss (e.g. a
        motion-field conservation term), which needs access to the estimated motion
        field in addition to the precipitation prediction. The loss function is
        responsible for unpacking this tuple; `engine.py` is unaffected either way,
        since it only ever does `criterion(pred, y)` without inspecting `pred`.
    """

    def __init__(
        self,
        input_timesteps: int,
        input_channels: int,
        output_timesteps: int,
        output_channels: int = 2,
        features: Sequence[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        bilinear: bool = True,
        nonnegativity: str = "relu",
        return_motion_field: bool = False,   
    ) -> None:
        super().__init__()
        logger.info("Initializing MFUNet forecaster.")
        self.input_timesteps = input_timesteps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_timesteps = output_timesteps
        self.features = list(features)
        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self.nonnegativity = nonnegativity
        self.return_motion_field = return_motion_field
        self._validate_cfg()
        
        logger.info(
            "Initializing MFUNetForecaster (Tin=%d, Cin=%d, Tout=%d, return_motion_field=%s).",
            input_timesteps, input_channels, output_timesteps, return_motion_field,
        )

        self.backbone = MFUNetBackbone(
            in_channels=self.input_timesteps * self.input_channels,
            out_channels=self.output_channels,
            features=self.features,
            kernel_size=self.kernel_size,
            mode="motion_field",
            bilinear=self.bilinear,
        )

    def _validate_cfg(self) -> None:
        if self.input_timesteps <= 0:
            raise ValueError("input_timesteps must be > 0.")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be > 0.")
        if self.output_timesteps <= 0:
            raise ValueError("output_timesteps must be > 0.")
        if len(self.features) < 2:
            raise ValueError("features must have length >= 2.")
        if self.nonnegativity not in ("relu", "softplus", "none"):
            raise ValueError("nonnegativity must be one of 'relu', 'softplus', 'none'.")

    @staticmethod
    def _extrapolate(
        timesteps: int, precip: torch.Tensor, motion_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Semi-Lagrangian extrapolation of `precip` along `motion_field` for `timesteps`
        forward steps, via iterative `F.grid_sample` warping (LUPIN's original scheme).

        Parameters
        ----------
        timesteps:
            Number of forward extrapolation steps to produce.
        precip:
            Field to be warped, shape (B, C, H, W).
        motion_field:
            Estimated motion field (u, v), shape (B, 2, H, W).

        Returns
        -------
        torch.Tensor
            Extrapolated field(s), shape (B, timesteps, H, W).
        """
        velocity = motion_field / (motion_field.shape[-1] / 2)

        x_values, y_values = torch.meshgrid(
            torch.arange(velocity.shape[-2]), torch.arange(velocity.shape[-1])
        )
        xy_coords = torch.stack([y_values, x_values]).to(precip.device)
        xy_coords = (xy_coords) / ((velocity.shape[-1]) / 2) - 1  # assumes square input

        precip_extrap = torch.zeros(
            (precip.shape[0], timesteps, precip.shape[2], precip.shape[3])
        ).to(precip.device)
        displacement = torch.zeros(
            (velocity.shape[0], 2, velocity.shape[2], velocity.shape[3])
        ).to(precip.device)
        velocity_inc = velocity.clone()

        for ti in range(timesteps):
            coords_warped = xy_coords.unsqueeze(0) + displacement
            velocity_inc = F.grid_sample(
                velocity, coords_warped.movedim(1, -1),
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            displacement -= velocity_inc
            coords_warped = xy_coords.unsqueeze(0) + displacement
            precip_warped = F.grid_sample(
                precip, coords_warped.movedim(1, -1),
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )
            precip_extrap[:, ti : ti + 1] = precip_warped

        return precip_extrap

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x:
            Input sequence, shape (B, Tin, C, H, W) — the benchmark's 5D dataset contract.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If `return_motion_field` is False (default): the forecast, shape
            (B, Tout, C, H, W).
            If True: a tuple `(forecast, motion_fields)`, where `motion_fields` has
            shape (B, Tout, 2, H, W) — one (u, v) field per predicted step.
        """
        b, tin, c, h, w = x.shape
        context = x.reshape(b, tin * c, h, w)

        preds = []
        motion_fields = []
        last_frame = x[:, -1]  # (B, C, H, W)

        for _ in range(self.output_timesteps):
            mf = self.backbone(context)  # (B, 2, H, W)
            next_frame = self._extrapolate(1, last_frame, mf)  # (B, 1, H, W)
            preds.append(next_frame.unsqueeze(1))
            motion_fields.append(mf.unsqueeze(1))

            last_frame = next_frame
            context = torch.cat([context[:, c:], next_frame], dim=1)  # janela deslizante

        y = torch.cat(preds, dim=1)  # (B, Tout, C, H, W)
        y = self._apply_nonnegativity(y, self.nonnegativity)

        if self.return_motion_field:
            return y, torch.cat(motion_fields, dim=1)
        return y