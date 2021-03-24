import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import audio_low_res_proccessing as alrp
import superAudioNet as asn

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def test_output_audio(low_res_input, sample_rate, PATH, device):
    model = asn.Net()
    model.to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    with torch.no_grad():
        high_res_output = model(low_res_input)
        high_res_output = high_res_output.cpu().detach()
        low_res_input = low_res_input.cpu().detach()
        alrp.save_low_high_audio(torch.reshape(high_res_output,(1,-1)), torch.reshape(low_res_input,(1,-1)), sample_rate)

data = torchaudio.load("./VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_030_mic1.flac")
low_res_input, _ = alrp.data_to_inp_tar(data, device)
PATH = './superAudioNet.pth'
test_output_audio(low_res_input, 16000, PATH, device)