"""Clean PyTorch U-Net interface for optional learned pixel-probability segmentation."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.double_conv(x)


    class T7BlobUNet(nn.Module):
        """
        Lightweight U-Net for predicting pixel-level blob probabilities from multiscale context.
        Standard inputs:
          ch 0: DAPI nuclear stain
          ch 1: Cross-cycle top-k signal map
          ch 2: Persistence map
          ch 3: Background map
          ch 4: Saturation mask
        Total in_channels = 5.
        """
        def __init__(self, in_channels: int = 5, out_channels: int = 1, base_features: int = 16):
            super().__init__()
            self.inc = DoubleConv(in_channels, base_features)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
            
            self.up1 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
            self.conv_up1 = DoubleConv(base_features * 4, base_features * 2)
            
            self.up2 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
            self.conv_up2 = DoubleConv(base_features * 2, base_features)
            
            self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            
            u1 = self.up1(x3)
            # Pad if odd dimensions
            diff_y = x2.size()[2] - u1.size()[2]
            diff_x = x2.size()[3] - u1.size()[3]
            u1 = nn.functional.pad(u1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x_cat1 = torch.cat([x2, u1], dim=1)
            x4 = self.conv_up1(x_cat1)
            
            u2 = self.up2(x4)
            diff_y = x1.size()[2] - u2.size()[2]
            diff_x = x1.size()[3] - u2.size()[3]
            u2 = nn.functional.pad(u2, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x_cat2 = torch.cat([x1, u2], dim=1)
            x5 = self.conv_up2(x_cat2)
            
            logits = self.outc(x5)
            return logits


def predict_blob_probabilities_unet(
    dapi_image: np.ndarray,
    top_k_map: np.ndarray,
    persistence_map: np.ndarray,
    background_map: np.ndarray,
    saturation_mask: np.ndarray,
    model_path: Optional[str | Path] = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Standardized inference wrapper for the U-Net pixel-probability model.
    Constructs the 5-channel feature tensor and runs forward inference.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to run the U-Net segmentation interface.")

    H, W = top_k_map.shape
    dapi = np.squeeze(dapi_image).astype(np.float32) if dapi_image is not None else np.zeros((H, W), dtype=np.float32)

    # Normalize input channels
    c0 = dapi / max(1.0, np.percentile(dapi, 99))
    c1 = top_k_map / max(1.0, np.percentile(top_k_map, 99))
    c2 = persistence_map.astype(np.float32) / 12.0
    c3 = background_map / max(1.0, np.percentile(background_map, 99))
    c4 = saturation_mask.astype(np.float32)

    # Stack to (1, 5, H, W)
    tensor_in = np.stack([c0, c1, c2, c3, c4], axis=0)[None, ...]
    torch_in = torch.from_numpy(tensor_in).float().to(device)

    model = T7BlobUNet(in_channels=5, out_channels=1)
    if model_path is not None and Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(torch_in)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

    return prob.astype(np.float32)
