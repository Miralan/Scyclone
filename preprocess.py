# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 18:04
# @Author  : Miralan
# @File    : preprocess.py.py


import glob
import random
from pathlib import Path

training_rate = 0.9999
val_rate = 0.0001


def find_files(path, pattren="*.wav"):
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{pattren}', recursive=True):
        filenames.append(filename)
    return filenames


def create_metadata(path="datasets"):
    path = Path(path)
    names = []
    path_lists = []
    for i in path.iterdir():
        names.append(i.stem)
    for i in names:
        wav_lists = find_files(path=path.joinpath(i))
        path_lists.append(wav_lists)

    assert len(names) == len(path_lists)
    for i in range(len(names)):
        with open(f"datasets/training_{names[i]}.txt", 'w', encoding='utf-8') as w1:
            for j in path_lists[i]:
                w1.write(f"{i}\n")


if __name__ == '__main__':
    create_metadata()