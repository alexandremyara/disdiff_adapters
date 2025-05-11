import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    VAE Decoder.
    
    See VAE docstring.

    ConvTranspose2D double H,W with kernel=4, stride=2, padding=1
    """
    
    def __init__(self, out_channels: int, img_size: int, latent_dim: int, out_encoder_shape: tuple[int]):
        """
        Load a decoder variational.

        Args:
            latent_dim: int, size of the input latent vector.
            out_shape: tuple[int], (C,H,W) shape of the output image
            out_encoder_shape: tuple[int], (C,H,W) shape of the last convolutional layer of the encoder.
        """

        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.out_encoder_shape = out_encoder_shape

        C,H,W = self.out_encoder_shape

        self.net = nn.Sequential(
            nn.Linear(latent_dim, C*W*H), 
            nn.Unflatten(1, (C, W, H)),
    
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), #*1
            nn.ELU(),

            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1), #*2
            nn.ELU(),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), #*1
            nn.ELU(),

            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),#*2
            nn.ELU(),

            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1), #*1
            nn.ELU(),
 
            nn.ConvTranspose2d(48, self.out_channels, kernel_size=4, stride=2, padding=1), #*2
            nn.ELU(),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1), #*1
        )

    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        
        x = self.net(z)
        if not (self.img_size > 0 and (self.img_size & (self.img_size - 1))) == 0 : 
            print(f"{self.img_size} is not a power of 2. Interpolation from this shape to {(self.out_encoder_shape, self.out_encoder_shape)}")
            x = F.interpolate(x, 
                              size=(self.img_size, self.img_size), 
                              mode='bilinear', 
                              align_corners=False)
        sigmoid = nn.Sigmoid()
        return sigmoid(x)
