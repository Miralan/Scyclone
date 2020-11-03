# -*- coding: utf-8 -*-
# @Time    : 2020/11/3 11:16
# @Author  : Miralan
# @File    : test.py
import hydra
import torch


def demo(a):
    print(a.batch_size)


@hydra.main(config_path="conf", config_name="config")
def test(cfg):
    print(cfg.training)
    a = torch.randn(1, 2, 3)
    b = torch.abs(a * torch.gt(a, 0))
    print(a)
    print(b)


if __name__ == '__main__':
    test()