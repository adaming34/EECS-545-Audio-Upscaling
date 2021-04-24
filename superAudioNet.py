import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

from torchaudio.datasets import VCTK_092
import data_loader as dl
from pixelshuffle1d import PixelShuffle1D, PixelUnshuffle1D
import audio_low_res_proccessing as alrp
import stft_conversions as sc

import os




###########################################################

class Net(nn.Module):
    """
    docstring
    """

    def __init__(self):
        super(Net, self).__init__()
        n_filters = np.intc(np.array([128, 384, 512, 512, 512, 512, 512, 512]) / 4)
        self.n_filters = n_filters
        n_filtersizes = np.array([7, 7, 7, 7, 7, 7, 7, 7, 7])
        self.n_filtersizes = n_filtersizes
        n_padding = np.intc((n_filtersizes - 1) * 0.5)
        scale_factor = 2

        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.down1 = nn.Conv1d(1, n_filters[0], n_filtersizes[0], padding = n_padding[0], stride=2)
        self.down2 = nn.Conv1d(n_filters[0], n_filters[1], n_filtersizes[1], padding = n_padding[1], stride=2)
        self.down3 = nn.Conv1d(n_filters[1], n_filters[2], n_filtersizes[2], padding = n_padding[2], stride=2)
        self.down4 = nn.Conv1d(n_filters[2], n_filters[3], n_filtersizes[3], padding = n_padding[3], stride=2)
        self.down5 = nn.Conv1d(n_filters[3], n_filters[4], n_filtersizes[4], padding = n_padding[4], stride=2)
        self.down6 = nn.Conv1d(n_filters[4], n_filters[5], n_filtersizes[5], padding = n_padding[5], stride=2)
        self.down7 = nn.Conv1d(n_filters[5], n_filters[6], n_filtersizes[6], padding = n_padding[6], stride=2)
        self.bottle = nn.Conv1d(n_filters[6], n_filters[7], n_filtersizes[7], padding = n_padding[7], stride=2)

        self.dropout = nn.Dropout(0.25)
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.pixel_upsample = PixelShuffle1D(scale_factor)

        self.downSampling = []
        self.up1 = nn.Conv1d(n_filters[7], n_filters[6]*2, n_filtersizes[7], padding = n_padding[7])
        self.up2 = nn.Conv1d(n_filters[6]*1, n_filters[5]*2, n_filtersizes[6], padding = n_padding[6])
        self.up3 = nn.Conv1d(n_filters[5]*1, n_filters[4]*2, n_filtersizes[5], padding = n_padding[5])
        self.up4 = nn.Conv1d(n_filters[4]*1, n_filters[3]*2, n_filtersizes[4], padding = n_padding[4])
        self.up5 = nn.Conv1d(n_filters[3]*1, n_filters[2]*2, n_filtersizes[3], padding = n_padding[3])
        self.up6 = nn.Conv1d(n_filters[2]*1, n_filters[1]*2, n_filtersizes[2], padding = n_padding[2])
        self.up7 = nn.Conv1d(n_filters[1]*1, n_filters[0]*2, n_filtersizes[1], padding = n_padding[1])
        self.up8 = nn.Conv1d(n_filters[0]*1, 2, n_filtersizes[0], padding = n_padding[0])


        # frequency stuff    
        n_stft_filters = np.intc(np.array([128, 256, 512, 512, 512, 512]) / 4)
        n_stft_filtersizes = np.array([7, 7, 7, 7, 7, 7])
        n_stft_padding = np.intc((n_stft_filtersizes - 1) * 0.5)
        self.stft_residuals = []

        self.stft_down1 =  ComplexConv2d(1                , n_stft_filters[0]  , (1,n_stft_filtersizes[0]), padding = (0,n_stft_padding[0]), stride=2)
        self.stft_down2 =  ComplexConv2d(n_stft_filters[0], n_stft_filters[1]  , (1,n_stft_filtersizes[1]), padding = (0,n_stft_padding[1]), stride=2)
        self.stft_down3 =  ComplexConv2d(n_stft_filters[1], n_stft_filters[2]  , (1,n_stft_filtersizes[2]), padding = (0,n_stft_padding[2]), stride=2)
        self.stft_down4 =  ComplexConv2d(n_stft_filters[2], n_stft_filters[3]  , (1,n_stft_filtersizes[3]), padding = (0,n_stft_padding[3]), stride=2)
        self.stft_down5 =  ComplexConv2d(n_stft_filters[3], n_stft_filters[4]  , (1,n_stft_filtersizes[4]), padding = (0,n_stft_padding[4]), stride=2)
        self.stft_down6 =  ComplexConv2d(n_stft_filters[4], n_stft_filters[5]  , (1,n_stft_filtersizes[5]), padding = (0,n_stft_padding[5]), stride=2)
        
        self.stft_up1 =    ComplexConv2d(n_stft_filters[5], n_stft_filters[4]*2, (1,n_stft_filtersizes[5]), padding = (0,n_stft_padding[5]))
        self.stft_up2 =    ComplexConv2d(n_stft_filters[4], n_stft_filters[3]*2, (1,n_stft_filtersizes[4]), padding = (0,n_stft_padding[4]))
        self.stft_up3 =    ComplexConv2d(n_stft_filters[3], n_stft_filters[2]*2, (1,n_stft_filtersizes[3]), padding = (0,n_stft_padding[3]))
        self.stft_up4 =    ComplexConv2d(n_stft_filters[2], n_stft_filters[1]*2, (1,n_stft_filtersizes[2]), padding = (0,n_stft_padding[2]))
        self.stft_up5 =    ComplexConv2d(n_stft_filters[1], n_stft_filters[0]*2, (1,n_stft_filtersizes[1]), padding = (0,n_stft_padding[1]))
        self.stft_up6 =    ComplexConv2d(n_stft_filters[0], 2                  , (1,n_stft_filtersizes[0]), padding = (0,n_stft_padding[0]))

        self.pixel_upsample_2d = nn.PixelShuffle(scale_factor)




    def forward_downsample(self, x):
        '''
        Insert caption
        '''
        
        self.downSampling.append(x)
        x = self.leakyRelu(self.down1(x))
        self.downSampling.append(x)
        x = self.leakyRelu(self.down2(x))
        self.downSampling.append(x)
        x = self.leakyRelu(self.down3(x))
        self.downSampling.append(x)
        x = self.leakyRelu(self.down4(x))
        self.downSampling.append(x)
        x = self.leakyRelu(self.down5(x))
        self.downSampling.append(x)
        x = self.leakyRelu(self.down6(x))
        self.downSampling.append(x)
        x = self.leakyRelu(self.down7(x))
        self.downSampling.append(x)

        return x

    def forward_bottleneck(self, x):
        '''
        Insert caption
        '''
        x = self.bottle(x)
        x = self.dropout(x)
        x = self.leakyRelu(x)
        
        return x

    def forward_upsample(self, x):
        '''
        Insert caption
        '''
        x = self.pixel_upsample(F.relu(self.dropout(self.up1(x)))) + self.downSampling[7]
        x = self.pixel_upsample(F.relu(self.dropout(self.up2(x)))) + self.downSampling[6]
        x = self.pixel_upsample(F.relu(self.dropout(self.up3(x)))) + self.downSampling[5]
        x = self.pixel_upsample(F.relu(self.dropout(self.up4(x)))) + self.downSampling[4]
        x = self.pixel_upsample(F.relu(self.dropout(self.up5(x)))) + self.downSampling[3]
        x = self.pixel_upsample(F.relu(self.dropout(self.up6(x)))) + self.downSampling[2]
        x = self.pixel_upsample(F.relu(self.dropout(self.up7(x)))) + self.downSampling[1]
        x = self.pixel_upsample((self.up8(x))) + self.downSampling[0]
        # if you include this then your output is the exact same as input. commented out you get 0 as answer (exploding/vanishing gradient?)
        self.downSampling.clear()
       
        return x

    def forward_stft(self, y):
        '''
        '''
        self.stft_residuals.append(y)
        y = complex_relu(self.stft_down1(y))
        self.stft_residuals.append(y)
        y = complex_relu(self.stft_down2(y))
        self.stft_residuals.append(y)
        y = complex_relu(self.stft_down3(y))
        self.stft_residuals.append(y)
        y = complex_relu(self.stft_down4(y))
        # self.stft_residuals.append(y)
        # y = complex_relu(self.stft_down5(y))
        # self.stft_residuals.append(y)
        # y = complex_relu(self.stft_down6(y))
        # self.stft_residuals.append(y)

        # y = self.pixel_upsample(complex_relu(self.stft_up1(y)).squeeze(2)).unsqueeze(2) + self.stft_residuals[5]
        # y = self.pixel_upsample(complex_relu(self.stft_up2(y)).squeeze(2)).unsqueeze(2) + self.stft_residuals[4]
        y = self.pixel_upsample(complex_relu(self.stft_up3(y)).squeeze(2)).unsqueeze(2) + self.stft_residuals[3]
        y = self.pixel_upsample(complex_relu(self.stft_up4(y)).squeeze(2)).unsqueeze(2) + self.stft_residuals[2]
        y = self.pixel_upsample(complex_relu(self.stft_up5(y)).squeeze(2)).unsqueeze(2) + self.stft_residuals[1]
        y = self.pixel_upsample(self.stft_up6(y).squeeze(2)).unsqueeze(2)               + self.stft_residuals[0]
        self.stft_residuals.clear()

        return y

    # x = waveform ,y is the stft
    def forward(self, x):
        '''
        Insert caption
        '''

        # y = sc.wav2stft(x[0,0], 128)
        # # y = (y)
        # y = y.unsqueeze(0)
        # y = y.unsqueeze(0)
        # y = self.forward_stft(y)
        # y = sc.stft2wav(y[0,0], 128, x.shape[2])
        # y = y.unsqueeze(0)
        # y = y.unsqueeze(0)

        # s = sc.fft(x)
        # s = s.unsqueeze(0)
        # s = self.forward_stft(s)
        # s = sc.ifft(s, x.shape[2])
        # s = torch.view_as_real(s)[0,:,:,:,0]    

        x = self.forward_downsample(x)
        x = self.forward_bottleneck(x)
        x = self.forward_upsample(x)

        x = x#+s # FuSiOn Layer

        return x

def main():
    # Load Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUNNING ON:", device)
    vctk_p255 = dl.VCTKCroppedDataSet(root_dir='data/p225')
    # Split into training and validation subsets
    train_size = int(0.85 * len(vctk_p255))
    valid_size = len(vctk_p255) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(vctk_p255, [train_size, valid_size])
    # VCTK_data = torchaudio.datasets.VCTK_092('./', download=False) # old non-custom dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)


    
    # Train
    net = Net()
    net.to(device)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data)
            nn.init.orthogonal(m.bias.data)

    # net.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0) # not sure if this helps

    ###########################################################
    PATH = './superAudioNet.pth'
    start_time = time.perf_counter()
    NUM_OF_EPOCHS = 225
    loss_results = np.zeros((2,NUM_OF_EPOCHS))
    for k in range(NUM_OF_EPOCHS):
        training_loss = 0.0
        valid_loss = 0.0
        
        # GO THROUGH ALL TRAINING DATA
        net.train()
        itr_train = 0
        for data in train_loader:
            input = data["low_res"].to(device)
            target = data["high_res"].to(device)

            optimizer.zero_grad()

            output = net(input)
            loss = criterion(output, target)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
 
            itr_train = itr_train + 1

            del loss, output
        
        # GO THROUGH ALL VALIDATION
        net.eval()
        itr_valid = 0
        for data in valid_loader:
            input = data["low_res"].to(device)
            target = data["high_res"].to(device)

            with torch.no_grad():
                output = net(input)
                loss = criterion(output, target)
                valid_loss += loss.item()

            itr_valid = itr_valid + 1

            del loss, output

        # print('------------------------------------------')
        print('EPOCH:', k+1)
        print('TRAINING LOSS:  ', round(training_loss/itr_train, 10))
        print('VALIDATION LOSS:', round(valid_loss/itr_valid, 10))
        print('------------------------------------------')
        loss_results[0,k] = training_loss/itr_train
        loss_results[1,k] = valid_loss/itr_valid

        torch.save(net.state_dict(), PATH)
        k = k + 1


    print('Finished training, it took: ',
        (time.perf_counter() - start_time), 'seconds')

    np.savetxt("loss.csv", loss_results, delimiter=",")
    # Test!
    start_time = time.perf_counter()
    net.eval()

    with torch.no_grad():
        
        data = torchaudio.load("./VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_030_mic2.flac")
        low_res_input, actual_high_res = alrp.data_to_inp_tar(data, device)

        high_res_output = net(low_res_input)
        actual_high_res = torch.reshape(actual_high_res.cpu().detach(),(1,-1))
        high_res_output = torch.reshape(high_res_output.cpu().detach(),(1,-1))
        low_res_input = torch.reshape(low_res_input.cpu().detach(),(1,-1))
        alrp.save_low_high_audio(high_res_output, low_res_input, 16000)
        alrp.plot_spectrogram(high_res_output, 16000)
        alrp.plot_spectrogram(low_res_input, 16000)
        alrp.plot_spectrogram(actual_high_res, 16000)

    print('Finished testing, it took: ',
        (time.perf_counter() - start_time), 'seconds')


if __name__ == '__main__':
    main()
    
