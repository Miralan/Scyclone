# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 16:40
# @Author  : Miralan
# @File    : meldataset.py


import math
import os
import random
import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
import librosa
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path, sample_rate):
    data, sampling_rate = librosa.load(full_path, sample_rate)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def get_dataset_filelist():
    with open("datasets/training_baker.txt", 'r', encoding='utf-8') as fi:
        training_files_1 = [x.replace('\n', '').replace('\r', '') for x in fi.readlines() if len(x) > 0]

    with open("datasets/training_jqz.txt", 'r', encoding='utf-8') as fi:
        training_files_2 = [x.replace('\n', '').replace('\r', '') for x in fi.readlines() if len(x) > 0]
    return training_files_1, training_files_2


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(1024).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class MelDataset(Dataset):
    def __init__(self, training_files1, training_files2, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, shuffle=True, device=None):
        self.audiofiles1 = training_files1
        self.audiofiles2 = training_files2
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audiofiles1)
            random.shuffle(self.audiofiles2)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.device = device

    def __getitem__(self, index):
        filename1 = self.audiofiles1[index]
        filename2 = self.audiofiles2[index]
        audio1, _ = load_wav(filename1, self.sampling_rate)
        audio2, _ = load_wav(filename2, self.sampling_rate)

        audio1 = normalize(audio1) * 0.95
        audio2 = normalize(audio2) * 0.95

        audio1 = torch.FloatTensor(audio1)
        audio2 = torch.FloatTensor(audio2)
        audio1 = audio1.unsqueeze(0)
        audio2 = audio2.unsqueeze(0)

        if audio1.size(1) >= self.segment_size:
            max_audio_start = audio1.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio1 = audio1[:, audio_start:audio_start + self.segment_size]
        else:
            audio1 = torch.nn.functional.pad(audio1, (0, self.segment_size - audio1.size(1)), 'constant')

        if audio2.size(1) >= self.segment_size:
            max_audio_start = audio2.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio2 = audio2[:, audio_start:audio_start + self.segment_size]
        else:
            audio2 = torch.nn.functional.pad(audio2, (0, self.segment_size - audio2.size(1)), 'constant')

        mel1 = mel_spectrogram(audio1, self.n_fft, self.num_mels,
                              self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                              center=False)
        mel2 = mel_spectrogram(audio2, self.n_fft, self.num_mels,
                              self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                              center=False)

        return mel1.squeeze(), mel2.squeeze()

    def __len__(self):
        return min(len(self.audiofiles1), len(self.audiofiles2))