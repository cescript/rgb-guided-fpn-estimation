import torch.nn as nn

class FixedPatternNoiseDecoder(nn.Module):
    def __init__(self, input_channels, scalar_and_offset, base_channels=128):
        super().__init__()
        # create a scaling coefficents
        self.scalar_and_offset = scalar_and_offset
        
        # simple denoising block
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels//2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels//2, base_channels//4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels//4, 1, 1), nn.Tanh()
        )

    def forward(self, noise_embedding):
        """
        noise_embedding
        Returns:
            fpn estimate: [B, 1, H, W]
        """
        dec_x = self.decoder(noise_embedding)
        
        # return scaled coefficients
        return self.scalar_and_offset[0] * dec_x + self.scalar_and_offset[1]