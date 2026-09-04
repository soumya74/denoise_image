import torch
import torch.nn as nn
from . import MODEL_REGISTRY

@MODEL_REGISTRY.register("baseline_cnn")
class BaselineDenoiser(nn.Module):
    """Simple 5-layer Convolutional Autoencoder Baseline"""
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)