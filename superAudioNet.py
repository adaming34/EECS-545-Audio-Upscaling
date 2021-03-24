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
import audio_low_res_proccessing as alrp

import os




###########################################################


class Net(nn.Module):
    """
    docstring
    """

    def __init__(self):
        super(Net, self).__init__()
        filt = 48
        size = 51

        n_filters = np.intc(np.array([128, 384, 512, 512, 512, 512, 512, 512]) / 16)
        n_filtersizes = np.array([65, 33, 17,  9,  9,  9,  9, 9, 9])
        n_padding = np.intc((n_filtersizes - 1) * 0.5)

        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.down1 = nn.Conv1d(1, n_filters[0], n_filtersizes[0], padding = n_padding[0])
        self.down2 = nn.Conv1d(n_filters[0], n_filters[1], n_filtersizes[1], padding = n_padding[1])
        self.down3 = nn.Conv1d(n_filters[1], n_filters[2], n_filtersizes[2], padding = n_padding[2])
        self.down4 = nn.Conv1d(n_filters[2], n_filters[3], n_filtersizes[3], padding = n_padding[3])
        self.down5 = nn.Conv1d(n_filters[3], n_filters[4], n_filtersizes[4], padding = n_padding[4])
        self.down6 = nn.Conv1d(n_filters[4], n_filters[5], n_filtersizes[5], padding = n_padding[5])
        self.down7 = nn.Conv1d(n_filters[5], n_filters[6], n_filtersizes[6], padding = n_padding[6])
        self.down8 = nn.Conv1d(n_filters[6], n_filters[7], n_filtersizes[7], padding = n_padding[7])

        self.dropout = nn.Dropout(0.5)
        self.leakyRelu = nn.LeakyReLU(0.2)

        self.downSampling = [] #np.array([])
        self.up1 = nn.Conv1d(n_filters[7]*1, n_filters[6]*1, n_filtersizes[7], padding = n_padding[7])
        self.up2 = nn.Conv1d(n_filters[6]*1, n_filters[5]*1, n_filtersizes[6], padding = n_padding[6])
        self.up3 = nn.Conv1d(n_filters[5]*1, n_filters[4]*1, n_filtersizes[5], padding = n_padding[5])
        self.up4 = nn.Conv1d(n_filters[4]*1, n_filters[3]*1, n_filtersizes[4], padding = n_padding[4])
        self.up5 = nn.Conv1d(n_filters[3]*1, n_filters[2]*1, n_filtersizes[3], padding = n_padding[3])
        self.up6 = nn.Conv1d(n_filters[2]*1, n_filters[1]*1, n_filtersizes[2], padding = n_padding[2])
        self.up7 = nn.Conv1d(n_filters[1]*1, n_filters[0]*1, n_filtersizes[1], padding = n_padding[1])
        self.up8 = nn.Conv1d(n_filters[0]*1, 1, n_filtersizes[0], padding = n_padding[0])
        
        

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
        x = self.leakyRelu(self.down8(x))
        

        return x

    def forward_bottleneck(self, x):
        '''
        Insert caption
        '''
        x = self.down8(x)
        x = self.dropout(x)
        x = self.leakyRelu(x)
        
        return x

    def forward_upsample(self, x):
        '''
        Insert caption
        '''
        x = F.relu(self.dropout(self.up1(x))) + self.downSampling[7]
        x = F.relu(self.dropout(self.up2(x))) + self.downSampling[6]
        x = F.relu(self.dropout(self.up3(x))) + self.downSampling[5]
        x = F.relu(self.dropout(self.up4(x))) + self.downSampling[4]
        x = F.relu(self.dropout(self.up5(x))) + self.downSampling[3]
        x = F.relu(self.dropout(self.up6(x))) + self.downSampling[2]
        x = F.relu(self.dropout(self.up7(x))) + self.downSampling[1]
        x = F.relu(self.dropout(self.up8(x)))# + self.downSampling[0] 
        # if you include this then your output is the exact same as input. commented out you get 0 as answer (exploding/vanishing gradient?)
        self.downSampling.clear()
        return x

    def forward(self, x):
        '''
        Insert caption
        '''
        x = self.forward_downsample(x)
        x = self.forward_bottleneck(x)
        x = self.forward_upsample(x)

        return x





def main():
    # Load Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    VCTK_data = torchaudio.datasets.VCTK_092('./', download=False)
    data_loader = torch.utils.data.DataLoader(VCTK_data, batch_size=1, shuffle=False) #change shuffle to true when actually running
    
    # Train
    net = Net()
    net.to(device)
    criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    ###########################################################

    start_time = time.perf_counter()

    for k in range(5):
        running_loss = 0.0
        
        itr = 0
        for data in data_loader:
            if data[3][0] != 'p225':
                break

            inp, target = alrp.data_to_inp_tar(data, device)

            optimizer.zero_grad()

            output = net(inp)
            loss = criterion(output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if itr % 10 == 0:
                print('At iteration ', itr, ',loss: ', loss.item(), end = '\r')
            itr = itr + 1

            del loss, output

        print('\n--------------------------------\nEPOCH:', k, ', TOTAL LOSS:', running_loss, '\n--------------------------------\n')
        # print(net.down1.weight.T)
        k = k + 1


    print('Finished training, it took: ',
        (time.perf_counter() - start_time), 'seconds')

    PATH = './superAudioNet.pth'
    torch.save(net.state_dict(), PATH)

    # Test!
    start_time = time.perf_counter()

    with torch.no_grad():
        net.eval()
        data = torchaudio.load("./VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_030_mic1.flac")
        low_res_input, _ = alrp.data_to_inp_tar(data, device)


        high_res_output = net(low_res_input)
        high_res_output = torch.reshape(high_res_output.cpu().detach(),(1,-1))
        low_res_input = torch.reshape(low_res_input.cpu().detach(),(1,-1))
        alrp.save_low_high_audio(high_res_output, low_res_input, 16000)
        alrp.plot_spectogram(high_res_output)
        alrp.plot_spectogram(low_res_input)

    print('Finished testing, it took: ',
        (time.perf_counter() - start_time), 'seconds')


if __name__ == '__main__':
    main()
    
