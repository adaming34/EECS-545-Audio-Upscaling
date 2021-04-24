import torch
import torchaudio
import matplotlib.pyplot as plt

# input: waveform tensor
# returns: stft split into k buckets, should maintain same length as input *1 (may need to split into two real layers if complex num is an issue)
def wav2stft(wav, k):
    stft  = torch.stft(input=wav, n_fft=2*k - 1, hop_length=k, return_complex=True)
    # stft = torch.log(stft)
    # stft = torch.view_as_real(stft)
    # stft = stft.permute(0,3,1,2)
    # print(stft[:,0,:,:])
    # stft[:,0,:,:] = torch.log(abs(stft[:,0,:,:])) #take log of real to make more readable, later incorperate MEL
    # stft_imag = stft[:,1,:,:]

    return stft

# input: stft (complex)
# returns: original waveform of length l, assuming 16000 sample rate, and k buckets in stft
def stft2wav(stft, k, l):
   
    istft = torch.istft(input=stft, n_fft=2*k - 1, hop_length=k, length=l)
    return istft

def griff(stft, k, l):
    wav = torchaudio.transforms.GriffinLim(n_fft=2*k - 1, hop_length=k, length=l, rand_init=True, n_iter=32)(stft)
    return wav

def fft(signal):
    return torch.fft.fft(signal)

def ifft(fft, l):
    return torch.fft.ifft(fft, n=l)


#  Just a test ot make sure the functions works
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data = torchaudio.load(".\data\p225\high_res\p225_030_mic2_high.wav")
    signal = data[0].to(device)

    k = 128 # num of buckets
    l = signal.shape[1] # length of the origninal waveform

    stft = wav2stft(signal, k)

    fft = torch.fft.fft(signal, norm="ortho")
    ifft = torch.fft.ifft(fft, n=l, norm="ortho")
    r_ifft = stft = torch.view_as_real(ifft)[:,:, 0]

    

    spectrogram = torch.log(torch.abs(stft)).cpu()
    mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2*k - 1, hop_length=k, n_mels=128, f_max=8000)(signal.cpu())
    specgram = torchaudio.transforms.Spectrogram(n_fft=2*k - 1, hop_length=k)(signal.cpu())





    # plt.pcolormesh(torch.log(specgram[0]))
    # plt.ylabel('Frequency (buckets) [Hz]')
    # plt.xlabel('Time (buckets) [sec]')
    # plt.show()

    wav = griff(specgram, k, l)

    torchaudio.save("stft_original.wav", signal.cpu(), data[1])
    torchaudio.save("stft_reconstruction.wav", r_ifft.cpu(), data[1])



if __name__ == '__main__':
    main()