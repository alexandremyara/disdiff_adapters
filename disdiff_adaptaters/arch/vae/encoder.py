import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    VAE Encoder.
    
    See VAE docstring.
    """
    
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()

        self.n_layers = 3
        self.input_channels = input_channels

        self.net = nn.Sequential(
            # in_channelsx128x128 -> 48x64x64
            nn.Conv2d(self.input_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 48x64x64 -> 48x64x64
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 48x64x64 -> 96x32x32
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 96x32x32 -> 96x32x32
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 96x32x32 -> 192x16x16
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 192x16x16 -> 192x16x16
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 192x16x16 -> 384x8x8
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 384x8x8 -> 384x8x8
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(384*8*8, latent_dim * 2),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_repr: torch.Tensor = self.net(x)
        mu, logvar = torch.chunk(latent_repr, 2, dim=1)
        return mu, logvar
