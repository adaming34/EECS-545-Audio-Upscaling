import torch
import torchaudio
import matplotlib.pyplot as plt

# input: waveform tensor
# returns: stft split into k buckets, should maintain same length as input *1 (may need to split into two real layers if complex num is an issue)
def wav2stft(wav, k):
    stft  = torch.stft(input=wav, n_fft=2*k - 1, hop_length=k, return_complex=True)
    # stft = torch.view_as_real(stft)
    return stft

# input: stft (complex)
# returns: original waveform of length l, assuming 16000 sample rate, and k buckets in stft
def stft2wav(stft, k, l):
    istft = torch.istft(input=stft, n_fft=2*k - 1, hop_length=k, length=l)
    return istft

#  Just a test ot make sure the functions works
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data = torchaudio.load(".\data\p225\high_res\p225_030_mic2_high.wav")
    signal = data[0].to(device)

    k = 128 # num of buckets
    l = signal.shape[1] # length of the origninal waveform

    stft = wav2stft(signal, k)
    wav = stft2wav(stft, k, l)

    spectrogram = torch.log(torch.abs(stft.cpu())**2)
    torchaudio.save("stft_original.wav", signal.cpu(), data[1])
    torchaudio.save("stft_reconstruction.wav", wav.cpu(), data[1])

    plt.pcolormesh(spectrogram[0])
    plt.ylabel('Frequency (buckets) [Hz]')
    plt.xlabel('Time (buckets) [sec]')
    plt.show()

if __name__ == '__main__':
    main()