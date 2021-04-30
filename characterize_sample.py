import torch
import torchaudio
import torch.nn as nn
import numpy as np
import io
import os
import sys
import math
import matplotlib.pyplot as plt

def calc_mse(original, sample):
    loss = nn.MSELoss()
    return loss(sample, original)
    

def calc_psnr(original, sample):
    return -10*math.log10(calc_mse(original, sample))

def stft(sample, **kwargs):
    return torchaudio.transforms.Spectrogram(**kwargs)(sample)

def calc_lsd(original, sample):
    errs = stft(sample).log2()[0].numpy() - stft(original).log2()[0].numpy()
    return ((errs**2).mean(axis=1)**0.5).mean(axis=0)

def spectrogram(sample, title='', sr=16000):
    specgram = stft(sample)
    plt.figure()
    ex = (0, sample.shape[1] / sr, 0, sr/2)
    plt.imshow(specgram.log2()[0,:,:].numpy(), aspect='auto', origin='lower', extent=ex)
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.title(title)
    plt.show()

def main():
    gt_file = sys.argv[1]
    sample_files = sys.argv[2:]
    
    gt, sr = torchaudio.load(gt_file)
    
    print("          \tpsnr\tlsd")
    for filename in sample_files:
        sample, samp_sr = torchaudio.load(filename)
        if samp_sr != sr:
            print(f"{filename} sample-rate ({samp_sr}) doesn't match ground truth sample rate ({sr})")
            continue
        
        length = min(sample.shape[1], gt.shape[1])
        psnr = calc_psnr(gt[:, :length], sample[:, :length])
        lsd = calc_lsd(gt[:, :length], sample[:, :length])
        
        print(f"{filename}:\t{psnr:.3f}\t{lsd:.3f}")
        
        spectrogram(sample[:, :length], title=filename, sr=sr)


if __name__ == '__main__':
    main()
