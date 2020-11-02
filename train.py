# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 15:35
# @Author  : Miralan
# @File    : train.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import argparse
from itertools import chain
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
import numpy as np
from model.generator import Generator
from model.discriminator import Discriminator
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from torch.utils.data import DistributedSampler, DataLoader
from utils import scan_checkpoint, save_checkpoint, load_checkpoint
from adabelief_pytorch import AdaBelief

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True


def train(rank, a):
    if a.num_gpus > 1:
        init_process_group(backend="nccl", init_method="tcp://localhost:54321",
                           world_size=1 * a.num_gpus, rank=rank)

    torch.cuda.manual_seed(a.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    g_x2y = Generator().to(device)
    g_y2x = Generator().to(device)
    d_x = Discriminator().to(device)
    d_y = Discriminator().to(device)

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0

    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        g_x2y.load_state_dict(state_dict_g["g_x2y"])
        g_y2x.load_state_dict(state_dict_g["g_y2x"])
        d_x.load_state_dict(state_dict_do["d_x"])
        d_y.load_state_dict(state_dict_do["d_y"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    if a.num_gpus > 1:
        g_x2y = DistributedDataParallel(g_x2y, device_ids=[rank]).to(device)
        g_y2x = DistributedDataParallel(g_y2x, device_ids=[rank]).to(device)
        d_x = DistributedDataParallel(d_x, device_ids=[rank]).to(device)
        d_y = DistributedDataParallel(d_y, device_ids=[rank]).to(device)

    optim_g = AdaBelief(chain(g_x2y.parameters(), g_y2x.parameters()), a.learning_rate)
    optim_d = AdaBelief(chain(d_x.parameters(), d_y.parameters()), a.learning_rate)

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, a.segment_size, a.n_fft, a.num_mels,
                          a.hop_size, a.win_size, a.sampling_rate, a.fmin, a.fmax, n_cache_reuse=0,
                          shuffle=False if a.num_gpus > 1 else True, fmax_loss=a.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if a.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=a.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=a.batch_size,
                              pin_memory=True,
                              drop_last=True)


def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    a = parser.parse_args()

    torch.manual_seed(a.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
        a.num_gpus = torch.cuda.device_count()
        a.batch_size = int(a.batch_size / a.num_gpus)
        print('Batch size per GPU :', a.batch_size)
    else:
        pass

    if a.num_gpus > 1:
        mp.spawn(train, nprocs=a.num_gpus, args=(a,))
    else:
        train(0, a)