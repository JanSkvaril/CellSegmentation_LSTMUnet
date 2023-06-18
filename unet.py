import torch

import torch.nn as nn
import torch.nn.functional as F


class DobleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DobleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.max_pool = nn.MaxPool2d(2)
        self.encoder1 = DobleConvolution(n_channels, 64)
        self.encoder2 = DobleConvolution(64, 128)
        self.encoder3 = DobleConvolution(128, 256)
        self.encoder4 = DobleConvolution(256, 512)
        self.encoder5 = DobleConvolution(512, 1024)

        self.up1 = DobleConvolution(1024 + 512, 512)
        self.up2 = DobleConvolution(512 + 256, 256)
        self.up3 = DobleConvolution(256 + 128, 128)
        self.up4 = DobleConvolution(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)

        # upsampling + convolution can be use instead
        self.upconv1 = nn.ConvTranspose2d(1024, 1024, 2, 2)
        self.upconv2 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.upconv3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.upconv4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.upconv5 = nn.ConvTranspose2d(64, 64, 2, 2)

    def forward(self, x):

        # encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.max_pool(x1))
        x3 = self.encoder3(self.max_pool(x2))
        x4 = self.encoder4(self.max_pool(x3))
        x5 = self.encoder5(self.max_pool(x4))

        # decoder
        x = self.upconv1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)

        x = self.upconv2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)

        x = self.upconv4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)

        x = self.conv_last(x)
        return x
