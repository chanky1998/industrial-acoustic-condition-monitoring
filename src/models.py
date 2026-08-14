import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ReLU(),
            #nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ReLU(),
            #nn.ConvTranspose2d(16,1, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.Sigmoid()

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed