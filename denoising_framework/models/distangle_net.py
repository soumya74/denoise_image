import torch
import torch.nn as nn
from . import MODEL_REGISTRY
from .disentangle_encoder import DisentangleEncoder
from .disentangle_decoder import CleanDecoder, NoiseDecoder

@MODEL_REGISTRY.register("disentangle_net")
class DisentanglementNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, content_dim=28, noise_dim=4, **kwargs):
        super().__init__()
        self.encoder = DisentangleEncoder(in_channels, content_dim, noise_dim)
        self.clean_decoder = CleanDecoder(in_channels=content_dim, out_channels=out_channels)
        self.noise_decoder = NoiseDecoder(in_channels=noise_dim, out_channels=out_channels)

    def encode(self, img):
        """Extracts content and noise latent representations."""
        return self.encoder(img)

    def decode_clean(self, z_c):
        """Reconstructs clean image from content embedding."""
        return self.clean_decoder(z_c)

    def decode_noise(self, z_n):
        """Reconstructs noise pattern from noise embedding."""
        return self.noise_decoder(z_n)

    def forward(self, noisy_img, clean_img=None):
        """
        Executes both the Noisy Path and the Clean Path:
          - Noisy Image (y) -> z_c^y, z_n^y -> x_hat, n_hat -> y_hat = x_hat + n_hat
          - Clean Image (x) -> z_c^x, z_n^x -> x_hat_clean
        """
        # Noisy Path (y)
        z_c_y, z_n_y = self.encode(noisy_img)
        x_hat = self.decode_clean(z_c_y)
        n_hat = self.decode_noise(z_n_y)
        y_hat = x_hat + n_hat

        # Clean Path (x) - used during dual-path training
        if clean_img is not None:
            z_c_x, z_n_x = self.encode(clean_img)
            x_hat_clean = self.decode_clean(z_c_x)
            return {
                "x_hat": x_hat,
                "n_hat": n_hat,
                "y_hat": y_hat,
                "z_c_y": z_c_y,
                "z_n_y": z_n_y,
                "z_c_x": z_c_x,
                "z_n_x": z_n_x,
                "x_hat_clean": x_hat_clean
            }

        return x_hat