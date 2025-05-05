import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels: int, input_size: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1), nn.ELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), nn.ELU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), nn.ELU()
        )

        # calcul automatique de la taille aplatie après convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.features(dummy)
            print(out.shape)
            self.flattened_size = out.shape[1]*out.shape[2]*out.shape[3]

        self.fc = nn.Sequential(nn.Flatten(),nn.Linear(self.flattened_size, latent_dim * 2))

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar
