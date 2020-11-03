# -*- coding: utf-8 -*-
# @Time    : 2020/11/3 11:16
# @Author  : Miralan
# @File    : test.py
import hydra


def demo(a):
    print(a.batch_size)


@hydra.main(config_path="conf", config_name="config")
def test(cfg):
    print(cfg.training.batch_size)
    demo(cfg.training)
    cfg.training.batch_size = cfg.training.batch_size // cfg.training.num_gpus
    print(cfg.training.batch_size)
    demo(cfg.training)


if __name__ == '__main__':
    test()