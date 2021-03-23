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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


VCTK_data = torchaudio.datasets.VCTK_092('./', download=False)

data_loader = torch.utils.data.DataLoader(VCTK_data, batch_size=1, shuffle=False) #change shuffle to true when actually running

###########################################################


class Net(nn.Module):
    """
    docstring
    """

    def __init__(self):
        super(Net, self).__init__()
        filt = 48
        size = 51

        n_filters = np.intc(np.array([128, 384, 512, 512, 512, 512, 512, 512]) / 8)
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

        self.dropout = nn.Dropout(0.1)
        self.leakyRelu = nn.LeakyReLU(0.2)

        self.downSampling = [] #np.array([])
        self.up1 = nn.Conv1d(n_filters[7]*1, n_filters[6]*2, n_filtersizes[7], padding = n_padding[7])
        self.up2 = nn.Conv1d(n_filters[6]*2, n_filters[5]*2, n_filtersizes[6], padding = n_padding[6])
        self.up3 = nn.Conv1d(n_filters[5]*2, n_filters[4]*2, n_filtersizes[5], padding = n_padding[5])
        self.up4 = nn.Conv1d(n_filters[4]*2, n_filters[3]*2, n_filtersizes[4], padding = n_padding[4])
        self.up5 = nn.Conv1d(n_filters[3]*2, n_filters[2]*2, n_filtersizes[3], padding = n_padding[3])
        self.up6 = nn.Conv1d(n_filters[2]*2, n_filters[1]*2, n_filtersizes[2], padding = n_padding[2])
        self.up7 = nn.Conv1d(n_filters[1]*2, n_filters[0]*2, n_filtersizes[1], padding = n_padding[1])
        self.up8 = nn.Conv1d(n_filters[0]*2, 1, n_filtersizes[0], padding = n_padding[0])
        
        

    def forward_downsample(self, x):
        '''
        Insert caption
        '''
        x = self.leakyRelu(self.down1(x))
        # self.downSampling.append(x)
        x = self.leakyRelu(self.down2(x))
        # self.downSampling.append(x)
        x = self.leakyRelu(self.down3(x))
        # self.downSampling.append(x)
        x = self.leakyRelu(self.down4(x))
        # self.downSampling.append(x)
        x = self.leakyRelu(self.down5(x))
        # self.downSampling.append(x)
        x = self.leakyRelu(self.down6(x))
        # self.downSampling.append(x)
        x = self.leakyRelu(self.down7(x))
        # self.downSampling.append(x)
        x = self.leakyRelu(self.down8(x))
        # self.downSampling.append(x)

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
        x = F.relu(self.dropout(self.up1(x)))
        x = F.relu(self.dropout(self.up2(x)))
        x = F.relu(self.dropout(self.up3(x)))
        x = F.relu(self.dropout(self.up4(x)))
        x = F.relu(self.dropout(self.up5(x)))
        x = F.relu(self.dropout(self.up6(x)))
        x = F.relu(self.dropout(self.up7(x)))
        x = F.relu(self.dropout(self.up8(x)))
        return x

    def forward(self, x):
        '''
        Insert caption
        '''
        x = self.forward_downsample(x)
        x = self.forward_bottleneck(x)
        x = self.forward_upsample(x)

        return x


net = Net()
net.to(device)
criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.005)

###########################################################

start_time = time.perf_counter()
for k in range(301):
    running_loss = 0.0
    itr = 0
    for data in data_loader:
        input, target = alrp.data_to_inp_tar(data, device)

        optimizer.zero_grad()

        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # break # REMOVE BEFORE FLIGHT
        del loss, output
        if itr % 1 == 0:
            print('At iteration ', itr, ',running loss: ', running_loss)
        itr = itr + 1



    k = k + 1


print('Finished training, it took: ',
      (time.perf_counter() - start_time), 'seconds')

PATH = './superAudioNet.pth'
torch.save(net.state_dict(), PATH)

# Test!
start_time = time.perf_counter()

with torch.no_grad():
    tar = targets[trainingCase, :, :, :].to(device).unsqueeze(0)
    outs = net(inputs[trainingCase, :, :, :].to(device).unsqueeze(0))
    loss = criterion(outs, tar)
    print('Test data loss: ', loss.item())

print('Finished testing, it took: ',
      (time.perf_counter() - start_time), 'seconds')


def plotAnalysis(anaNum):
    """
    docstring
    """
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    # fig.suptitle('Horizontally stacked subplots')
    # ax1.imshow(inputs[trainingCase,0,:,:], cmap="gray")
    # ax1.axis('off'), ax2.axis('off'), ax3.axis('off'), ax4.axis('off'), ax5.axis('off'), ax6.axis('off')
    # ax2.imshow(tar[0,0,:,:].cpu().detach().numpy() , cmap="gray")
    # ax3.imshow(outs[0,0,:,:].cpu().detach().numpy() , cmap="gray")
    # ax5.imshow(tar[0,1,:,:].cpu().detach().numpy() , cmap="gray")
    # ax6.imshow(outs[0,1,:,:].cpu().detach().numpy() , cmap="gray")
    # plt.show()
    return


plotAnalysis(0)
print('done')
