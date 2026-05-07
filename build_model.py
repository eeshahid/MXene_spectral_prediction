import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from typing import Iterable, Optional, Set


def build_model(args):
    backbone_name = getattr(args, "backbone_name", "mobilenet_v2")

    return SpectrumConvStack(
        backbone_name=backbone_name,
        output_dim=args.output_dim,
        pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_savgol=args.use_savgol,
        savgol_pos=args.savgol_pos,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        use_mcsr=args.use_mcsr,
    )


def savgol_kernel(window_length: int, polyorder: int) -> torch.Tensor:
    from scipy.linalg import pinv
    import numpy as np

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")

    half_window = (window_length - 1) // 2
    x = np.arange(-half_window, half_window + 1)
    A = np.vander(x, polyorder + 1)

    ATA_inv = pinv(A.T @ A)
    coeffs = ATA_inv @ A.T
    kernel = coeffs[0]

    return torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1)


class SavitzkyGolayLayer(nn.Module):
    def __init__(self, window_length: int, polyorder: int):
        super().__init__()
        kernel = savgol_kernel(window_length, polyorder)
        self.register_buffer("kernel", kernel)
        self.window_length = window_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.kernel, padding=self.window_length // 2)


def _replace_first_conv_with_grayscale(module: nn.Module):
    """
    Replace the first Conv2d found in a model with a 1-channel version.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
            )

            with torch.no_grad():
                if child.weight.shape[1] == 3:
                    new_conv.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                else:
                    new_conv.weight.copy_(child.weight)

                if child.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(child.bias)

            setattr(module, name, new_conv)
            return True

        replaced = _replace_first_conv_with_grayscale(child)
        if replaced:
            return True

    return False


def _build_torchvision_backbone(backbone_name: str, pretrained: bool, dropout: float, image_size: int,):
    backbone_name = backbone_name.lower()

    if backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        _replace_first_conv_with_grayscale(model.features)

        in_features = model.last_channel
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, in_features),
        )
        feature_dim = in_features
        return model, feature_dim

    raise ValueError(
        f"Unsupported backbone_name: {backbone_name}. "
        f"Supported: mobilenet_v2"
    )


class SpectrumBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenet_v2",
        image_size: int = 64,
        output_dim: int = 102,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.base_model, feature_dim = _build_torchvision_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            dropout=dropout,
            image_size=image_size,
        )

        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.fc(x)
        x = self.out_proj(x)
        x = x.unsqueeze(1)
        return x


class SpectrumConvStack(SpectrumBackbone):
    """
    Backwards-compatible replacement for mobilenet_v2_spectrum_convstack,
    but now supports multiple backbones.
    """
    def __init__(
        self,
        backbone_name: str = "mobilenet_v2",
        image_size: int = 64,
        output_dim: int = 102,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_savgol: bool = False,
        savgol_pos: str = "after_mcsr",
        savgol_window: int = 11,
        savgol_polyorder: int = 2,
        use_mcsr: bool = True,
    ):
        super().__init__(
            backbone_name=backbone_name,
            image_size=image_size,
            output_dim=output_dim,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        assert savgol_pos in ["before_mcsr", "after_mcsr"], (
            "savgol_pos must be 'before_mcsr' or 'after_mcsr'"
        )

        self.use_mcsr = use_mcsr
        self.use_savgol = use_savgol
        self.savgol_pos = savgol_pos

        if self.use_savgol:
            self.savgol_layer = SavitzkyGolayLayer(
                window_length=savgol_window,
                polyorder=savgol_polyorder,
            )

        self.conv1d1 = nn.Conv1d(1, 5, kernel_size=5, padding=2)
        self.conv1d = nn.Conv1d(5, 1, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        if self.use_savgol and self.savgol_pos == "before_mcsr":
            x = self.savgol_layer(x)

        if self.use_mcsr:
            x = self.conv1d1(x)
            x = self.conv1d(x)

        if self.use_savgol and self.savgol_pos == "after_mcsr":
            x = self.savgol_layer(x)

        x = x.squeeze(1)
        return x

