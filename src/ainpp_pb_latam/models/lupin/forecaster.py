from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ainpp_pb_latam._utils.lagrangian import semi_lagrangian_extrapolate
from ainpp_pb_latam.models.mfunet.backbone import MFUNetBackbone

logger = logging.getLogger(__name__)


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Return the subset of `state_dict` whose keys start with `prefix`, with
    that prefix removed. Returns an empty dict if nothing matches."""
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _load_submodule_checkpoint(
    submodule: nn.Module,
    checkpoint_path: Union[str, Path],
    submodule_key: str,
    legacy_prefixes: Sequence[str] = (),
) -> None:
    """
    Loads weights for `submodule` from a checkpoint file that may be either:

    1. A checkpoint of `submodule` on its own (bare state dict, e.g. saved by
       `EarlyStopping`/`save_epoch_checkpoint` while training that submodule
       directly as the top-level model, such as `model=lupin/mfunet`).
    2. A checkpoint of a *larger* model that contains `submodule` nested under
       `submodule_key` (e.g. loading the AF-U-Net out of a whole Stage-2
       `LUPINForecaster` checkpoint into Stage 3).

    Parameters
    ----------
    submodule:
        The `nn.Module` to load weights into (in place).
    checkpoint_path:
        Path to a `.pt` file produced by `torch.save(model.state_dict(), ...)`.
    submodule_key:
        Attribute name of `submodule` inside the *source* model that produced
        the checkpoint, e.g. "afunet_net" or "motion_field_net".
    legacy_prefixes:
        Extra key prefixes to try, for checkpoints saved by a different (but
        state-dict-compatible) class. `MFUNetForecaster` nests its backbone
        under `self.backbone`, so pass `("backbone.",)` when loading a Stage-1
        `model=lupin/mfunet` checkpoint into `motion_field_net`.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for '{submodule_key}': {checkpoint_path}"
        )

    raw_state = torch.load(checkpoint_path, map_location="cpu")

    candidates = [f"{submodule_key}."] + list(legacy_prefixes)
    last_error: Optional[Exception] = None

    for prefix in candidates:
        nested = _strip_prefix(raw_state, prefix)
        if not nested:
            continue
        try:
            submodule.load_state_dict(nested, strict=True)
            logger.info(
                "Loaded '%s' weights from %s (prefix '%s', %d tensors).",
                submodule_key, checkpoint_path, prefix, len(nested),
            )
            return
        except RuntimeError as exc:
            # Keys matched the prefix but shapes didn't (e.g. `features`
            # differs between stages) — remember and keep trying, then
            # surface a single actionable error if every candidate fails.
            last_error = exc

    # Fall back to treating the checkpoint as already being a bare state dict
    # for this exact submodule (e.g. it was itself the top-level model).
    try:
        submodule.load_state_dict(raw_state, strict=True)
        logger.info(
            "Loaded '%s' weights from %s (flat checkpoint, %d tensors).",
            submodule_key, checkpoint_path, len(raw_state),
        )
        return
    except RuntimeError as exc:
        last_error = exc

    raise ValueError(
        f"Could not load checkpoint '{checkpoint_path}' into submodule "
        f"'{submodule_key}'. Tried nested prefixes {candidates} and a flat "
        f"load. This usually means the checkpoint was produced with a "
        f"different `motion_features`/`afunet_features` (or `features`, for "
        f"an `MFUNetForecaster` checkpoint) than the ones configured for "
        f"this run — check that the two stages use matching architecture "
        f"hyperparameters. Underlying error: {last_error}"
    ) from last_error


class BaseForecaster(nn.Module):
    """Base class providing the non-negativity output head, matching the
    convention already used by `MFUNetForecaster`/`UNet*Forecaster`."""

    def _apply_nonnegativity(self, x: torch.Tensor, mode: str = "relu") -> torch.Tensor:
        if mode == "relu":
            return F.relu(x)
        if mode == "softplus":
            return F.softplus(x)
        return x


class LUPINForecaster(BaseForecaster):
    """
    LUPIN (Lagrangian U-Net for Precipitation Nowcasting) forecaster,
    Pavlik et al. (2025).

    A single class implements the two coupled sub-networks described in the
    paper and covers every training stage of the LUPIN curriculum used in
    this benchmark:

    * Stage 1 ("MF-U-Net pretraining") does **not** use this class: it
      reuses `ainpp_pb_latam.models.mfunet.forecaster.MFUNetForecaster`
      unchanged (config: `model=lupin/mfunet`), since that is exactly the
      motion-field-only sub-network LUPIN pretrains first.
    * Stage 2 ("AF-U-Net pretraining", config: `model=lupin/afunet`)
      instantiates this class with `freeze_motion_field=true` and
      `pretrained_motion_field_checkpoint=<stage-1 checkpoint>`, so only the
      AF-U-Net regression sub-network is trained.
    * Stage 3 ("joint finetuning", config: `model=lupin/final`) instantiates
      this class with both `pretrained_motion_field_checkpoint` and
      `pretrained_afunet_checkpoint` set and `freeze_motion_field=false`
      (default), finetuning both sub-networks jointly.
    * A from-scratch ablation (config: `model=lupin/direct`) instantiates
      this class with no pretrained checkpoints and no frozen submodules,
      training the full two-network model end to end in a single run,
      without the 3-stage curriculum.

    Architecture (single forward call over the full input window, matching
    this benchmark's direct multi-step contract — no external autoregressive
    driver loop is required, mirroring `MFUNetForecaster`):

        1. `motion_field_net` (a `MFUNetBackbone` in "motion_field" mode)
           estimates a dense motion field (u, v) from the stacked input
           window.
        2. That field is heavily Gaussian-blurred to obtain a smooth,
           large-scale velocity estimate, which is used to extrapolate
           ("persist") the last observed frame forward — this is the
           Lagrangian-persistence baseline `extrapolated` that AF-U-Net
           either corrects (if `apply_differencing=True`) or ignores (if
           `apply_differencing=False`).
        3. Every frame in the input window is independently warped forward
           (by a different number of steps each, so that they all land on
           the same target time) along the *unblurred* motion field. Stacking
           these co-registered frames yields the "Lagrangian coordinate"
           input tensor `x_lagrangian` used by AF-U-Net — this is what lets a
           plain convolutional network reason about storm growth/decay
           without having to also learn to compensate for advection.
        4. `afunet_net` (a `MFUNetBackbone` in "regression" mode) consumes
           `x_lagrangian` (optionally frame-differenced) and predicts either
           the residual correction to the persistence baseline
           (`apply_differencing=True`) or the absolute field directly
           (`apply_differencing=False`).
        5. The process repeats autoregressively for `output_timesteps` steps,
           sliding the input window forward with each newly predicted frame
           (same pattern as `MFUNetForecaster`/`UNetAutoRegressive`).

    Parameters
    ----------
    input_timesteps, input_channels, output_timesteps, output_channels:
        Same meaning/shape contract as every other forecaster in this
        benchmark. `output_channels` is currently expected to be 1.
    motion_features, afunet_features:
        Encoder/decoder channel widths for the motion-field sub-network and
        the AF-U-Net regression sub-network, respectively. The reference
        LUPIN implementation gives the AF-U-Net substantially more capacity
        than the motion-field network (it is doing the harder job of
        non-advective refinement), so the two are configured independently.
    kernel_size, bilinear:
        Forwarded to both `MFUNetBackbone` instances.
    nonnegativity:
        Non-linearity applied to the final precipitation forecast. One of
        "relu", "softplus", "none".
    apply_differencing:
        If True (the reference default for both the AF-U-Net-pretraining and
        joint-finetuning stages), AF-U-Net is trained/evaluated as a
        residual corrector on top of the blurred-motion persistence
        baseline, and the model's final output is
        `afunet_net(diff(x_lagrangian)) + extrapolated`. If False, AF-U-Net
        predicts the absolute field directly from the (non-differenced)
        Lagrangian-coordinate stack.
    persistence_blur_kernel_size, persistence_blur_sigma:
        Gaussian blur applied to the motion field before it is used to
        compute the persistence baseline `extrapolated` (step 2 above). The
        reference implementation uses a very large, fixed blur
        (kernel=255, sigma=127) tuned for 336x336 inputs; the defaults here
        are rescaled for this benchmark's 320x320 patches and are exposed so
        they can be retuned per-resolution. Only used when
        `apply_differencing=True` (otherwise `extrapolated` is not part of
        the output and is skipped entirely to save compute).
    freeze_motion_field:
        If True, `motion_field_net` is wrapped in `requires_grad_(False)`
        and kept in `eval()` mode at all times (see `train()` override
        below), so it neither learns nor leaks BatchNorm running statistics
        during Stage-2 training. Matches the pattern already used for the
        Stage-2 AF-U-Net pretraining in this project's LUPIN port.
    pretrained_motion_field_checkpoint:
        Optional path to a checkpoint to initialize `motion_field_net` from.
        Accepts either a checkpoint of a whole `MFUNetForecaster`
        (Stage-1 output, `model=lupin/mfunet`; keys nested under
        `backbone.`) or a checkpoint of a whole `LUPINForecaster` (keys
        nested under `motion_field_net.`) or a bare backbone state dict.
    pretrained_afunet_checkpoint:
        Optional path to a checkpoint to initialize `afunet_net` from.
        Accepts a checkpoint of a whole `LUPINForecaster`
        (Stage-2 output, `model=lupin/afunet`; keys nested under
        `afunet_net.`) or a bare backbone state dict.
    return_motion_field:
        If True, `forward` returns `(precip, motion_field)` instead of just
        `precip`, matching the 2-tuple convention already established by
        `MFUNetForecaster` (consumed by `MotionFieldConservationLoss` and by
        the tuple-unpack handling in `losses.py`/`visualization/samples.py`/
        `evaluation/evaluator.py`/`engine.py`). `motion_field` has shape
        (B, Tout, 2, H, W) — one (u, v) field per predicted step, taken
        *before* blurring. Required by `loss=lupin` / `loss=lupin_direct`
        (`LUPINJointLoss`); not needed by `loss=afunet` (plain data loss).
    """

    def __init__(
        self,
        input_timesteps: int,
        input_channels: int,
        output_timesteps: int,
        output_channels: int = 1,
        motion_features: Sequence[int] = (32, 64, 128, 256),
        afunet_features: Sequence[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        bilinear: bool = True,
        nonnegativity: str = "relu",
        apply_differencing: bool = True,
        persistence_blur_kernel_size: int = 63,
        persistence_blur_sigma: float = 15.0,
        freeze_motion_field: bool = False,
        pretrained_motion_field_checkpoint: Optional[str] = None,
        pretrained_afunet_checkpoint: Optional[str] = None,
        return_motion_field: bool = False,
    ) -> None:
        super().__init__()
        self.input_timesteps = input_timesteps
        self.input_channels = input_channels
        self.output_timesteps = output_timesteps
        self.output_channels = output_channels
        self.motion_features = list(motion_features)
        self.afunet_features = list(afunet_features)
        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self.nonnegativity = nonnegativity
        self.apply_differencing = apply_differencing
        self.persistence_blur_kernel_size = persistence_blur_kernel_size
        self.persistence_blur_sigma = persistence_blur_sigma
        self.freeze_motion_field = freeze_motion_field
        self.return_motion_field = return_motion_field
        self._validate_cfg()

        logger.info(
            "Initializing LUPINForecaster (Tin=%d, Cin=%d, Tout=%d, "
            "apply_differencing=%s, freeze_motion_field=%s).",
            input_timesteps, input_channels, output_timesteps,
            apply_differencing, freeze_motion_field,
        )

        in_channels = self.input_timesteps * self.input_channels

        self.motion_field_net = MFUNetBackbone(
            in_channels=in_channels,
            out_channels=2,  # ignored by mode="motion_field", kept explicit
            features=self.motion_features,
            kernel_size=self.kernel_size,
            mode="motion_field",
            bilinear=self.bilinear,
        )

        afunet_in_channels = in_channels - self.input_channels if apply_differencing else in_channels
        self.afunet_net = MFUNetBackbone(
            in_channels=afunet_in_channels,
            out_channels=self.output_channels,
            features=self.afunet_features,
            kernel_size=self.kernel_size,
            mode="regression",
            bilinear=self.bilinear,
        )

        if pretrained_motion_field_checkpoint:
            _load_submodule_checkpoint(
                self.motion_field_net,
                pretrained_motion_field_checkpoint,
                submodule_key="motion_field_net",
                legacy_prefixes=("backbone.",),  # Stage-1 MFUNetForecaster checkpoints
            )

        if pretrained_afunet_checkpoint:
            _load_submodule_checkpoint(
                self.afunet_net,
                pretrained_afunet_checkpoint,
                submodule_key="afunet_net",
            )

        if self.freeze_motion_field:
            self.motion_field_net.requires_grad_(False)
            logger.info("motion_field_net frozen (requires_grad=False).")

    def _validate_cfg(self) -> None:
        if self.input_timesteps <= 1:
            raise ValueError("input_timesteps must be > 1 (LUPIN needs at least 2 frames).")
        if self.input_channels != 1:
            # The shared semi-Lagrangian warp operator (`_utils.lagrangian`)
            # only supports single-channel fields, matching MFUNetForecaster.
            raise ValueError("LUPINForecaster currently requires input_channels == 1.")
        if self.output_channels != 1:
            raise ValueError("LUPINForecaster currently requires output_channels == 1.")
        if self.output_timesteps <= 0:
            raise ValueError("output_timesteps must be > 0.")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be > 0.")
        if len(self.motion_features) < 2 or len(self.afunet_features) < 2:
            raise ValueError("motion_features and afunet_features must have length >= 2.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.nonnegativity not in ("relu", "softplus", "none"):
            raise ValueError("nonnegativity must be one of 'relu', 'softplus', 'none'.")

    def train(self, mode: bool = True) -> "LUPINForecaster":
        """
        Standard `nn.Module.train()`, except `motion_field_net` is kept in
        `eval()` mode whenever `freeze_motion_field=True`.

        Without this override, `engine.run_training`'s unconditional
        `model.train()` at the top of every epoch would flip the frozen
        motion-field sub-network's BatchNorm layers back into training mode,
        silently corrupting its running statistics with AF-U-Net-stage
        batches even though its weights never receive gradients. Same
        pattern used in `MFUNetForecaster`-derived frozen-submodule setups
        elsewhere in this benchmark.
        """
        super().train(mode)
        if self.freeze_motion_field:
            self.motion_field_net.eval()
        return self

    def _lagrangian_stack(self, x: torch.Tensor, motion_field: torch.Tensor) -> torch.Tensor:
        """
        Warps every frame of the input window `x` (B, Tin, C, H, W), with
        C == 1 (see class docstring), forward to a common reference time
        (the end of the window) along `motion_field`, producing the
        co-moving "Lagrangian coordinate" stack consumed by AF-U-Net. Each
        frame i is warped forward by (Tin - i) steps under the (locally
        constant) estimated velocity.
        """
        tin = x.shape[1]
        warped_frames = []
        for i in range(tin):
            steps = tin - i
            # x[:, i] has shape (B, C=1, H, W); semi_lagrangian_extrapolate
            # returns (B, steps, H, W) for a single-channel field, so [:, -1]
            # picks out the frame warped all the way to the reference time.
            warped = semi_lagrangian_extrapolate(steps, x[:, i], motion_field)[:, -1]
            warped_frames.append(warped)
        return torch.stack(warped_frames, dim=1)  # (B, Tin, H, W)

    def _forward_step(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step forward pass: estimates the motion field from `context`
        (B, Tin*Cin, H, W) and returns `(next_frame, motion_field)`, where
        `next_frame` has shape (B, Cout, H, W).
        """
        b = context.shape[0]
        h, w = context.shape[-2:]
        # Recover the (B, Tin, C, H, W) view without squeezing C away, so the
        # channel dimension stays explicit for the warp calls below.
        x = context.view(b, self.input_timesteps, self.input_channels, h, w)

        motion_field = self.motion_field_net(context)  # (B, 2, H, W)

        if self.apply_differencing:
            blur_k = min(self.persistence_blur_kernel_size, h, w)
            blur_k = blur_k if blur_k % 2 == 1 else blur_k - 1
            blurred_mf = TF.gaussian_blur(
                motion_field, kernel_size=blur_k, sigma=self.persistence_blur_sigma
            )
            # x[:, -1] -> (B, C=1, H, W); output is already (B, 1, H, W).
            extrapolated = semi_lagrangian_extrapolate(1, x[:, -1], blurred_mf)
        else:
            extrapolated = None

        x_lagrangian = self._lagrangian_stack(x, motion_field)  # (B, Tin, H, W)

        if self.apply_differencing:
            afunet_input = torch.diff(x_lagrangian, dim=1)  # (B, Tin-1, H, W)
        else:
            afunet_input = x_lagrangian

        residual = self.afunet_net(afunet_input)  # (B, Cout, H, W)

        if self.apply_differencing:
            next_frame = residual + extrapolated
        else:
            next_frame = residual

        return next_frame, motion_field

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x:
            Input sequence, shape (B, Tin, C, H, W) — the benchmark's 5D
            dataset contract.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If `return_motion_field` is False (default): the forecast, shape
            (B, Tout, C, H, W).
            If True: a tuple `(forecast, motion_fields)`, where
            `motion_fields` has shape (B, Tout, 2, H, W).
        """
        b, tin, c, h, w = x.shape
        context = x.reshape(b, tin * c, h, w)

        preds = []
        motion_fields = []

        for _ in range(self.output_timesteps):
            next_frame, mf = self._forward_step(context)  # (B, C, H, W), (B, 2, H, W)
            preds.append(next_frame.unsqueeze(1))
            motion_fields.append(mf.unsqueeze(1))
            context = torch.cat([context[:, c:], next_frame], dim=1)  # sliding window

        y = torch.cat(preds, dim=1)  # (B, Tout, C, H, W)
        y = self._apply_nonnegativity(y, self.nonnegativity)

        if self.return_motion_field:
            return y, torch.cat(motion_fields, dim=1)
        return y
