import torch
import torchaudio
import torch.nn as nn
import numpy as np
import io
import os
import sys
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

def calc_mse(original, sample):
    loss = nn.MSELoss()
    return loss(sample, original)
    

def calc_psnr(original, sample):
    return -10*math.log10(calc_mse(original, sample))

def stft(sample, **kwargs):
    return torchaudio.transforms.Spectrogram(**kwargs)(sample)

def calc_lsd(original, sample):
    sample_ft = stft(sample)[0]
    orig_ft = stft(original)[0]
    
    mask = orig_ft > 0
    mask = np.all(mask.numpy(), axis=0)
    sample_ft = sample_ft[:, mask]
    orig_ft = orig_ft[:, mask]
    
    errs = 2*(sample_ft.log10()).numpy() - 2*(orig_ft.log10()).numpy()
    return ((errs**2).mean(axis=0)**0.5).mean(axis=0)

viridis = cm.get_cmap('viridis', 256)
colors = viridis(np.linspace(0, 1, 256))
colors[0, :] = np.array([0. , 0., 0., 0.])
cmap = ListedColormap(colors)

def spectrogram(sample, fig, ax, title='', sr=16000):
    specgram = stft(sample)
    ex = (0, sample.shape[1] / sr, 0, sr/2)
    ret = ax.imshow(10*specgram.log10()[0,:,:].numpy(), aspect='auto', origin='lower', extent=ex, cmap=cmap, vmin=-60, vmax=20)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    ax.set_title(title)
    
    return ret

def main():
    gt_file = sys.argv[1]
    sample_files = sys.argv[2:]
    
    gt, sr = torchaudio.load(gt_file)
    
    rows = len(sample_files) // 2 + 1
    
    fig, axs = plt.subplots(2, rows, sharex=True, sharey=True, constrained_layout=True)
    
    flat_axs = axs.flat
    
    pcm = spectrogram(gt, fig, flat_axs[0], title='Ground Truth', sr=sr)
    
    print("          \tpsnr\tlsd")
    for filename, ax in zip(sample_files, flat_axs[1:]):
        sample, samp_sr = torchaudio.load(filename)
        if samp_sr != sr:
            print(f"{filename} sample-rate ({samp_sr}) doesn't match ground truth sample rate ({sr})")
            continue
        
        length = min(sample.shape[1], gt.shape[1])
        psnr = calc_psnr(gt[:, :length], sample[:, :length])
        lsd = calc_lsd(gt[:, :length], sample[:, :length])
        
        print(f"{filename}:\t{psnr:.3f}\t{lsd:.3f}")
        
        name = ' '.join(filename.split('/')[-1].split('.')[:-1])
        
        pcm = spectrogram(sample[:, :length], fig, ax, title=name, sr=sr)
    
    fig.colorbar(pcm, ax=axs[..., 1], label='Intensity (db)')
    plt.show()

if __name__ == '__main__':
    main()
