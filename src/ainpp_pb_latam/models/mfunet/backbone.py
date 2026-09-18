from __future__ import annotations

import logging
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ainpp_pb_latam.models.mfunet.blocks import DoubleConv, DownBlock, UpBlock

logger = logging.getLogger(__name__)


class MFUNetBackbone(nn.Module):

    """
        U-Net-style encoder-decoder used as the motion-field estimator sub-network of the
        MF-U-Net (Motion Field U-Net) stage of LUPIN.
    
        A single dropout layer is applied at the bottleneck (between encoder and decoder),
        matching the original RainNet/MF-U-Net topology. The output head is selected by
        `mode`: for "motion_field", the head always emits exactly 2 channels (the u, v
        components of the estimated motion field), regardless of `out_channels` — see the
        warning below.
    
        Parameters
        ----------
        in_channels:
            Number of input channels (typically input_timesteps * input_channels, since
            frames are stacked along the channel dimension before being fed to the network).
        out_channels:
            Number of output channels. Only used by the "regression" and "segmentation"
            heads. Ignored by the "motion_field" head, which always outputs 2 channels
            (u, v). Kept as an explicit constructor argument (rather than hard-coded) so
            the class stays reusable for other modes/backbones in the benchmark.
        features:
            Number of channels at each encoder/decoder level.
        kernel_size:
            Convolution kernel size used throughout the network.
        mode:
            One of "regression", "segmentation", "motion_field". Selects the output head.
        bilinear:
            If True, uses bilinear upsampling in the decoder; otherwise uses transposed
            convolutions.
        """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        kernel_size: int = 3,
        mode: str = "motion_field",  # "regression" or "segmentation" or "motion_field"
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.mode = mode
        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self._validate_cfg()

        # Adjustment to bilinear
        features = list(features)
        if bilinear:
            features[-1] = features[-1] // 2

        self.drop = nn.Dropout(p=0.5)

        self.stem = DoubleConv(in_channels, features[0], kernel_size=kernel_size)

        # Encoder
        self.down = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down.append(DownBlock(features[i], features[i + 1], kernel_size=kernel_size))

        # Decoder
        self.up = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up.append(
                UpBlock(
                    features[i],
                    features[i - 1],
                    features[i - 1],
                    kernel_size=kernel_size,
                    bilinear=bilinear,
                )
            )

        if self.mode == "regression":
            self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
            
        elif self.mode == "segmentation":
            self.head = nn.Sequential(
                nn.Conv2d(features[0], out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            
        elif self.mode == "motion_field":
            # Padding "same" exige que o stride seja 1, o que é o padrão do Conv2d
            self.head = nn.Conv2d(features[0], 2, kernel_size=kernel_size, padding='same')
            
        else:
            raise NotImplementedError(f"Mode '{self.mode}' is not implemented.")

    def _validate_cfg(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be > 0.")
        if len(self.features) < 2:
            raise ValueError("features must have length >= 2.")
        if any(f <= 0 for f in self.features):
            raise ValueError("All feature sizes must be > 0.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not isinstance(self.bilinear, bool):
            raise ValueError("bilinear must be a boolean.")
        if self.mode not in ("regression", "segmentation", "motion_field"):
            raise NotImplementedError(f"Mode '{self.mode}' is not implemented.")
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        x = self.stem(x)
        skips.append(x)

        for down_block in self.down:
            x = down_block(x)
            skips.append(x)

        # O último elemento é o bottleneck
        bottleneck = skips.pop() 
        
        # Aplicando o dropout no bottleneck (opcional, mas comum)
        x = self.drop(bottleneck)

        for i, up_block in enumerate(self.up):
            # skips agora contem apenas as conexões residuais, lidas de trás para frente
            skip = skips[-(i + 1)]
            x = up_block(x, skip)

        # Agora o head inclui as ativações/convs corretas de acordo com o `mode`
        return self.head(x)

    