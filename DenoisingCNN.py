import torch.nn as nn


class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # (64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128, 128, 128)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (256, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (256, 64, 64)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (256, 64, 64)
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # (128, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128, 32, 32)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  #(128, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),   # (3, 128, 128)
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)

        return x