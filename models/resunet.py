import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDoubleConv(nn.Module):
    """
    Residual Double Convolution Block:
    (Conv2d -> BatchNorm -> ReLU) * 2 with residual shortcut projection.
    Helps prevent vanishing gradients when scaling to deep encoder networks.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Identity projection if channel dimensions differ
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class LargeUNet(nn.Module):
    """
    Deep, wide Residual U-Net for Semantic Segmentation.
    Encoder features: [64, 128, 256, 512, 1024] -> Bottleneck: 2048 (~123M params).
    Decoder path with transposed convolutions, skip-connection concatenation,
    and residual blocks.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256, 512, 1024],
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Path
        curr_in = in_channels
        for feature in features:
            self.downs.append(ResidualDoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = ResidualDoubleConv(features[-1], features[-1] * 2)

        # Decoder Path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ResidualDoubleConv(feature * 2, feature))

        # Final 1x1 Conv Classifier
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder forward pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for top-down concatenation
        skip_connections = skip_connections[::-1]

        # Decoder forward pass
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Transposed Conv upsampling
            skip = skip_connections[idx // 2]

            # Bilinear interpolation safeguard for arbitrary input sizes
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_x)  # Residual DoubleConv

        return self.final_conv(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running model sanity check on: {device}")

    # Use lighter feature set for quick test if running on CPU
    features = [32, 64, 128, 256, 512]
    model = LargeUNet(in_channels=3, out_channels=1, features=features).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters (features={features}): {total_params / 1e6:.2f} Million")

    dummy_input = torch.randn((2, 3, 256, 256), device=device)
    output = model(dummy_input)
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print("Model architecture verification passed!")
