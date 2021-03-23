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
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SubsetSC(VCTK_092):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + \
                load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


train_set = SubsetSC("training")
validation_set = SubsetSC("validation")
test_set = SubsetSC("testing")

# inputs = torch.from_numpy(inputs).float()
# targets = torch.from_numpy(targets).float()

# print(targets.shape)
###########################################################


class Net(nn.Module):
    """
    docstring
    """

    def __init__(self):
        super(Net, self).__init__()
        filt = 48
        size = 51

        n_filters = [128, 384, 512, 512, 512, 512, 512, 512]
        n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]

        self.down1 = nn.Conv1d(1, n_filters[0], n_filtersizes[0])
        self.down2 = nn.Conv1d(n_filters[0], n_filters[1], n_filtersizes[1])
        self.down3 = nn.Conv1d(n_filters[1], n_filters[2], n_filtersizes[2])
        self.down4 = nn.Conv1d(n_filters[2], n_filters[3], n_filtersizes[3])
        self.down5 = nn.Conv1d(n_filters[3], n_filters[4], n_filtersizes[4])
        self.down6 = nn.Conv1d(n_filters[4], n_filters[5], n_filtersizes[5])
        self.down7 = nn.Conv1d(n_filters[5], n_filters[6], n_filtersizes[6])
        self.down8 = nn.Conv1d(n_filters[6], n_filters[7], n_filtersizes[7])

        self.dropout = nn.Dropout(0.375)
        self.leakyRelu = nn.LeakyReLU(0.2)

        self.downSampling = [] #np.array([])

        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.up1 = nn.Conv1d(n_filters[7]*2, n_filters[6]*2, n_filtersizes[7])
        self.up2 = nn.Conv1d(n_filters[6]*2, n_filters[5]*2, n_filtersizes[6])
        self.up3 = nn.Conv1d(n_filters[5]*2, n_filters[4]*2, n_filtersizes[5])
        self.up4 = nn.Conv1d(n_filters[4]*2, n_filters[3]*2, n_filtersizes[3])
        self.up5 = nn.Conv1d(n_filters[3]*2, n_filters[2]*2, n_filtersizes[2])
        self.up6 = nn.Conv1d(n_filters[2]*2, n_filters[1]*2, n_filtersizes[1])
        self.up7 = nn.Conv1d(n_filters[1]*2, 1, n_filtersizes[0])
        

    def forward_downsample(self, x):
        '''
        Insert caption
        '''
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
        self.downSampling.append(x)

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
        return x

    def forward(self, x):
        '''
        Insert caption
        '''
        x = forward_downsample(self, x)
        x = forward_bottleneck(self, x)
        x = forward_upsample(self, x)

        return x


net = Net()
net.to(device)
criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.00005)

###########################################################

start_time = time.perf_counter()
for k in range(301):
    running_loss = 0.0
    for i in [1, 2, 3, 0]:
        inp, tar = inputs[i, :, :, :].to(
            device), targets[i, :, :, :].to(device)
        inp = inp.unsqueeze(0)
        tar = tar.unsqueeze(0)
        optimizer.zero_grad()

        outs = net(inp)
        loss = criterion(outs, tar)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if k % 50 == 0:
        print('At iteration ', k, ',running loss: ', running_loss)


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
