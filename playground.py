import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from torchaudio.datasets import VCTK_092
import data_loader as dl
from pixelshuffle1d import PixelShuffle1D, PixelUnshuffle1D
import audio_low_res_proccessing as alrp

import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def plot_spectogram(signal, k):
    specgram = torchaudio.transforms.Spectrogram(n_fft=2*k - 1,power=None ,center=False, normalized=True, )(signal)# (1,200,1449)
    # while signal.shape[1] / k != specgram.shape[2]:
        # specgram = specgram[:,:,0:-1]
    print('spectrogram shape:', specgram.shape)
    print('spectrogram size :', specgram.shape[1]* specgram.shape[2])
    # assert(specgram.shape[1]*specgram.shape[2] == signal.shape[1])
    # plt.figure()
    # plt.imshow(specgram.log2()[0,:,:].numpy(), aspect='auto', origin='lower')
    # plt.show()
    return specgram

data = torchaudio.load(".\data\p225\high_res\p225_030_mic2_high.wav")
print('wave shape:', data[0].shape)
# 289654
sgram = plot_spectogram(data[0], 128)
k = 128
start_time = time.perf_counter()
reconstructed = torchaudio.transforms.GriffinLim(n_fft=2*k - 1, n_iter=32,length = data[0].shape[1])(sgram)
print('Finished training, it took: ',
    (time.perf_counter() - start_time), 'seconds')
path = "reconstruction.wav"
torchaudio.save(path, reconstructed, 16000)
print('reconstructed size:', reconstructed[0].shape)
print('original size:', data[0].shape)
# Why not just make it easy and keep the complex nums?