import torch
import torch.nn as nn

class DisentangleEncoder(nn.Module):
    """
    Encoder Network that projects an RGB image into a 32-channel latent space,
    explicitly partitioned into:
      - 28 Content feature channels (z_c)
      - 4 Noise feature channels (z_n)
    """
    def __init__(self, in_channels=3, content_dim=28, noise_dim=4):
        super().__init__()
        self.content_dim = content_dim
        self.noise_dim = noise_dim
        total_dim = content_dim + noise_dim  # 32

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, total_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Dedicated heads for content and noise projection
        self.content_head = nn.Conv2d(total_dim, content_dim, kernel_size=3, padding=1)
        self.noise_head = nn.Conv2d(total_dim, noise_dim, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.backbone(x)
        z_c = self.content_head(features)  # [B, 28, H, W]
        z_n = self.noise_head(features)    # [B, 4, H, W]
        return z_c, z_n