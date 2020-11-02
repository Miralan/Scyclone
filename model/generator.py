# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 10:25
# @Author  : Miralan
# @File    : generator.py.py

import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm


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


class Generator(nn.Module):
    def __init__(self, in_channels, channels=256):
        """
        :param in_channels: Mel channels of input
        :param channels: layer channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.generator = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, self.channels, 1, 1)),
            nn.LeakyReLU(0.01),
            *[Residual_layer(self.channels) for i in range(7)],
            weight_norm(nn.Conv1d(self.channels, in_channels, 1, 1)),
            nn.Tanh()
        )
        self.num_params()

    def forward(self, inputs):
        return self.generator(inputs)

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)


def test():
    print("---- Start Training ----")
    print("=======================")
    print("--- Model Initializing ---")
    model = Generator(80, 256)
    print(model.state_dict().keys())
    inputs = torch.randn(10, 80, 30)
    print("--- Generating outputs ---")
    outputs = model(inputs)
    print(f'outputs.shape is {outputs.shape}')


if __name__ == '__main__':
    test()