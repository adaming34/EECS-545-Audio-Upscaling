import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.sox_effects as se

import io
import os
import math
import tarfile
import multiprocessing

import scipy
import requests
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def change_res(high_res_input, sample_rate_high_res, new_sample_rate):
    low_res_out = high_res_input
    # This can be used w/ SOX for more tuning 
    # effects = [['rate', new_sample_rate]]  # resample to 8000 Hz, we can add other effects later
    low_res_out = torchaudio.transforms.Resample(sample_rate_high_res, new_sample_rate)(high_res_input.view(1,-1))
    
    return low_res_out


def plot_spectogram(signal):
    specgram = torchaudio.transforms.Spectrogram()(signal)
    plt.figure()
    plt.imshow(specgram.log2()[0,:,:].numpy(), aspect='auto', origin='lower')
    plt.show()

def plot_spectrogram(signal, samplerate):
    specgram = torchaudio.transforms.Spectrogram()(signal)
    plt.figure()
    ex = (0, signal.shape[1] / samplerate, 0, samplerate/2)
    plt.imshow(specgram.log2()[0,:,:].numpy(), aspect='auto', origin='lower', extent=ex)
    plt.show()

def save_low_high_audio(high_res, low_res, sample_rate):
    path = "low_res.wav"
    torchaudio.save(path, low_res, sample_rate)
    path = "high_res.wav"
    torchaudio.save(path, high_res, int(sample_rate))
    

def data_to_inp_tar(data, device):
    # waveform is the ideal 
    waveform = torch.reshape(data[0], (1, -1)) # remove a dim (1,1,N) -> (1,N)

    sample_rate = data[1]
    high_res_rate = 16000
    low_res_rate = 4000

    # Downscale from 48000 to 16000 (still sounds good)
    high_res = change_res(waveform, sample_rate, high_res_rate)

    #  need to make sure the input signal is divisible by 4000 (1/12th)
    divisions = 1024
    while high_res.shape[1] % divisions != 0:
        # keep removing last element until condition is met
        high_res = high_res[:,0:(high_res.shape[1] - 1)]
    
    # Downscale from 16000 to 4000 (kinda crunchy)
    low_res = change_res(high_res, high_res_rate, low_res_rate)
    # Upscale from 4000 to 16000 to match sizes (still crunchy, but same size)
    low_res_high_rate = change_res(low_res, low_res_rate, high_res_rate)

    # assert(low_res_high_rate.shape[1] == high_res.shape[1])
   
    # telemetry for given data
    ##########################################################
    # save_low_high_audio(high_res, low_res, sample_rate)
    # plot_spectogram(low_res_high_rate)
    # plot_spectogram(high_res)
    ##########################################################

    # low_res = torch.reshape(low_res.cpu().detach(),(1,-1))
    # plot_spectrogram(low_res, 4000)

    input = low_res_high_rate.to(device)
    target = high_res.to(device)
    # print(torch.min(target))
    # print(torch.max(target))

    # input[input < 0] = -torch.log(input)
    # input[input < 0] = -torch.log(input)
    # target = torch.log(-target)

    input = torch.reshape(input, (1, 1, -1))
    target = torch.reshape(target, (1, 1, -1))
    return input, target
