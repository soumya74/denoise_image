import torch
import torch.nn as nn

class CleanDecoder(nn.Module):
    """
    Clean Image Decoder Network (D_clean):
    WEAKENED CAPACITY. Acts only as a strict translator. 
    Consumes ONLY the 28-dimensional content embedding (z_c) to reconstruct 
    the clean image (x_hat).
    """
    def __init__(self, in_channels=28, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Strictly bounds output to [0, 1] valid image space
        )

    def forward(self, z_c):
        return self.net(z_c)


class NoiseDecoder(nn.Module):
    """
    Noise Decoder Network (D_noise):
    INCREASED CAPACITY. The primary inference engine.
    Consumes ONLY the 4-dimensional noise embedding (z_n) to reconstruct 
    the explicit noise pattern (n_hat).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Added an extra layer for higher capacity noise modeling
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
            # MUST NOT have Sigmoid/ReLU here! Noise can be negative.
        )

    def forward(self, z_n):
        return self.net(z_n)