import torch
import torch.nn as nn
from . import MODEL_REGISTRY

@MODEL_REGISTRY.register("dncnn")
class DnCNN(nn.Module):
    """DnCNN: Residual Learning for Image Denoising"""
    def __init__(self, in_channels=3, out_channels=3, num_layers=17, num_features=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # Residual Learning: Predicts noise map N(x), clean output = x - N(x)
        noise_map = self.dncnn(x)
        return x - noise_map