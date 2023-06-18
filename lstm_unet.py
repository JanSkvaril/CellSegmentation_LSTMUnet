import torch

import torch.nn as nn
import torch.nn.functional as F


class DobleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DobleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)


# source: https://github.com/czifan/ConvLSTM.pytorch
class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                     kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),)

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t],  # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        # (S, B, C, H, W) -> (B, S, C, H, W)
        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()


class DobleConvolutionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DobleConvolutionDown, self).__init__()
        self.lstm = ConvLSTMBlock(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),

        )

    def forward(self, x, batch_size, sequence_size):
        _, C, H, W = x.shape
        x = x.view(batch_size, sequence_size, C, H, W)
        x = self.lstm(x)
        _, _, C, H, W = x.shape
        x = x.view(batch_size*sequence_size, C, H, W)

        x = self.conv(x)
        return x


class LSTMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(LSTMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.max_pool = nn.MaxPool2d(2)
        self.encoder1 = DobleConvolutionDown(n_channels, 64)
        self.encoder2 = DobleConvolutionDown(64, 128)
        self.encoder3 = DobleConvolutionDown(128, 256)
        self.encoder4 = DobleConvolutionDown(256, 512)
        self.encoder5 = DobleConvolutionDown(512, 1024)

        self.up1 = DobleConvolution(1024 + 512, 512)
        self.up2 = DobleConvolution(512 + 256, 256)
        self.up3 = DobleConvolution(256 + 128, 128)
        self.up4 = DobleConvolution(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upconv5 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        # encoder - sequence
        x1 = self.encoder1(x, B, S)
        x2 = self.encoder2(self.max_pool(x1), B, S)
        x3 = self.encoder3(self.max_pool(x2), B, S)
        x4 = self.encoder4(self.max_pool(x3), B, S)
        x5 = self.encoder5(self.max_pool(x4), B, S)

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
      #  x = self.actv(x)
        _, C, H, W = x.shape
        x = x.view(B, S, C, H, W)
        return x
