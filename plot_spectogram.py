import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from matplotlib.pyplot import figure
import numpy as np

def plot_spectrogram(path):
    sample_rate, samples = wavfile.read(path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    figure(figsize=(8, 6), dpi=120)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    # plt.imshow(spectrogram)
    plt.xlim((times[0], times[-1]))
    plt.ylim((0, 8000))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def main():
    PATH = "./data/p225/high_res/p225_030_mic2_high.wav"
    # PATH = "./data/p225/low_res/p225_030_mic2_low.wav"
    # PATH = "./low_res_upscaled.wav"
    plot_spectrogram(PATH)

if __name__ == '__main__':
    main()