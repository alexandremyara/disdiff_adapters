import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    VAE Decoder.
    
    See VAE docstring.
    """
    
    def __init__(self, out_channels: int, latent_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            # latent_dim -> 384x8x8
            nn.Linear(latent_dim, 384*8*8),
            nn.Unflatten(1, (384, 8, 8)),
            # 384x8x8 -> 192x16x16
            nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # 192x16x16 -> 192x16x16
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 192x16x16 -> 96x32x32
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # 96x32x32 -> 96x32x32
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 96x32x32 -> 48x64x64
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # 48x64x64 -> 48x64x64
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 48x64x64 -> out_channelsx128x128
            nn.ConvTranspose2d(48, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # out_channelsx128x128 -> out_channelsx128x128
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
