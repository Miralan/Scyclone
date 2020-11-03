# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 15:35
# @Author  : Miralan
# @File    : train.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import torch
import hydra
from itertools import chain
import torch.nn.functional as F
import torch.multiprocessing as mp
from model.generator import Generator
from adabelief_pytorch import AdaBelief
from model.discriminator import Discriminator
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from meldataset import MelDataset, get_dataset_filelist
from torch.utils.data import DistributedSampler, DataLoader
from utils import scan_checkpoint, save_checkpoint, load_checkpoint


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

    training_filelist1, training_filelist2 = get_dataset_filelist()

    trainset = MelDataset(training_filelist1, training_filelist2, a.segment_size, a.n_fft, a.num_mels,
                          a.hop_size, a.win_size, a.sampling_rate, a.fmin, a.fmax,
                          shuffle=False if a.num_gpus > 1 else True, device=device)

    train_sampler = DistributedSampler(trainset) if a.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=a.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=a.batch_size,
                              pin_memory=True,
                              drop_last=True)
    g_x2y.train()
    g_y2x.train()
    d_x.train()
    d_y.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if a.num_gpus > 1:
           train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            real_x, real_y = batch
            real_x, real_y = real_x.to(device), real_y.to(device)

            # Update Discriminator
            fake_y = g_x2y(real_x)
            fake_x = g_y2x(real_y)

            d_real_x = d_x(real_x)
            d_fake_x = d_x(fake_x)

            d_real_y = d_y(real_y)
            d_fake_y = d_y(fake_y)

            d_loss = torch.mean((0.5 - d_real_x) * torch.gt(0.5 - d_real_x, 0)) + \
                     torch.mean((0.5 - d_real_y) * torch.gt(0.5 - d_real_y, 0)) + \
                     torch.mean((0.5 + d_fake_x.detach()) * torch.gt(0.5 + d_fake_x.detach(), 0)) + \
                     torch.mean((0.5 + d_fake_y.detach()) * torch.gt(0.5 + d_fake_y.detach(), 0))

            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()

            # Update Generator loss
            cycle_x = g_y2x(fake_y)
            cycle_y = g_x2y(fake_x)

            indetity_x = g_y2x(real_x)
            indetity_y = g_x2y(real_y)

            d_fake_x = d_x(fake_x)
            d_fake_y = d_y(real_y)

            g_loss = torch.mean(-d_fake_x * torch.gt(-d_fake_x, 0)) + torch.mean(-d_fake_y * torch.gt(-d_fake_y, 0)) + \
                     a.lambda_cy * F.l1_loss(cycle_x, real_x) + a.lambda_cy * F.l1_loss(cycle_y, real_y) + \
                     a.lambda_id * F.l1_loss(indetity_x, real_x) + a.lambda_id * F.l1_loss(indetity_y, real_y)

            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Dis Loss. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, g_loss, d_loss, time.time() - start_b))

            # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                # save generator g_x2y and g_y2x
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'g_x2y': (g_x2y.module if a.num_gpus > 1 else g_x2y).state_dict(),
                                 'g_y2x': (g_y2x.module if a.num_gpus > 1 else g_y2x).state_dict()})

                checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                               {'d_x': (d_x.module if a.num_gpus > 1 else d_x).state_dict(),
                                'd_y': (d_y.module if a.num_gpus > 1 else d_y).state_dict(),
                                'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                'steps': steps, 'epoch': epoch})
            steps += 1


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    print('Initializing Training Process..')

    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        cfg.training.num_gpus = torch.cuda.device_count()
        cfg.training.batch_size = int(cfg.training.batch_size / cfg.training.num_gpus)
        print('Batch size per GPU :', cfg.training.batch_size)
    else:
        pass

    if cfg.training.num_gpus > 1:
        mp.spawn(train, nprocs=cfg.training.num_gpus, args=(cfg.training,))
    else:
        train(0, cfg.training)


if __name__ == '__main__':
    main()