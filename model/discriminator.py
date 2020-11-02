# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 11:13
# @Author  : Miralan
# @File    : discriminator.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm


class Residual_layer(nn.Module):
    def __init__(self, channels):
        """
        :param channels: layer channels
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad1d(2),
            weight_norm(nn.Conv1d(channels, channels, 5, 1)),
            nn.LeakyReLU(0.01),
            nn.ReflectionPad1d(2),
            weight_norm(nn.Conv1d(channels, channels, 5, 1))
        )

    def forward(self, inputs):
        x = self.layers(inputs)
        x = inputs + x
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=80, channels=256):
        """
        :param in_channels:
        :param channels:
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Conv1d(self.in_channels, self.channels, 1, 1)),
            nn.LeakyReLU(0.2),
            *[Residual_layer(self.channels) for i in range(7)],
            weight_norm(nn.Conv1d(self.channels, 1, 1, 1)),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, inputs):
        return self.discriminator(inputs)


def test():
    model = Discriminator()
    inputs = torch.randn(10, 80, 10)
    outputs = model(inputs)
    print(outputs.shape)


if __name__ == '__main__':
    test()
