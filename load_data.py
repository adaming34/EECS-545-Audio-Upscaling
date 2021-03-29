import torch
import torchaudio
import torch.nn as nn
import numpy as np
import io
import os
import math

# returns a 'resized' waveform at the new sample rate of shape (1,N)
def change_sample_rate(input_wave, input_sample_rate, output_sample_rate):
    # This can be used w/ SOX for more tuning 
    # effects = [['rate', new_sample_rate]]  # resample to 8000 Hz, we can add other effects later
    output_wave = torchaudio.transforms.Resample(input_sample_rate, output_sample_rate)(input_wave.view(1,-1))
    return output_wave

# returns two tensors of size (1,1,N) based on the input tensor at the sample rates inputted
def create_high_low_res(ideal_tensor, ideal_sample_rate, high_res_sample_rate, low_res_sample_rate):
    # remove a dim (1,1,N) -> (1,N)
    ideal_tensor = torch.reshape(ideal_tensor[0], (1, -1)) 

    # Downscale from ideal to high_res (still sounds good)
    high_res = change_sample_rate(ideal_tensor, ideal_sample_rate, high_res_sample_rate)

    #  need to make sure the input signal is divisible by a large power of 2 (to make sure we can keep cutting in half in model)
    divisions = 1024 # maybe we can go lower
    while high_res.shape[1] % divisions != 0:
        # keep removing last element until condition is met
        high_res = high_res[:,0:(high_res.shape[1] - 1)]
    
    # Downscale from high_res to low_res (kinda crunchy)
    low_res = change_sample_rate(high_res, high_res_sample_rate, low_res_sample_rate)
    # Upscale from low_res to high_res to match sizes (still crunchy, but same size)
    # this adds more computational cost, so worth changing in the future
    low_res_high_rate = change_sample_rate(low_res, low_res_sample_rate, high_res_sample_rate)
    # just make sure they are the same size
    assert(low_res_high_rate.shape[1] == high_res.shape[1])
    # reshape back to default tesnor shape
    low_res_high_rate = torch.reshape(low_res_high_rate, (1, 1, -1))
    high_res = torch.reshape(high_res, (1, 1, -1))

    return high_res, low_res_high_rate

def create_data_set(input_folder_path, output_folder_path, mic_str, high_sample_rate, low_sample_rate):
    # Get list of files in dir (will get issues if other dir in this dir)
    files = os.listdir(input_folder_path)
    for file in files:
        # choose which mic we are using
        if mic_str in file:
            # load in the uncompressed audio flac file
            tensor, sample_rate = torchaudio.load(input_folder_path + '/'+ file)
            # get the new waveform tensors at specified sample rates
            high_res, low_res = create_high_low_res(tensor, sample_rate, high_sample_rate, low_sample_rate)
            # remove a dim (1,1,N) -> (1,N)
            high_res = torch.reshape(high_res[0], (1, -1))
            low_res = torch.reshape(low_res[0], (1, -1)) 
            # save the the low res + high res file in the specified location
            low_file_path = output_folder_path + '/low_res/' + file.split('.')[0] + '_low.wav'
            high_file_path = output_folder_path + '/high_res/' + file.split('.')[0] + '_high.wav'
            torchaudio.save(low_file_path, low_res, high_sample_rate) # resampled to match sampling rate
            torchaudio.save(high_file_path, high_res, high_sample_rate)

# Need to pad data to allow minibatches (all samples need to be the same size)
def pad_data(output_folder_path):
    low_path = output_folder_path + '/low_res'
    high_path = output_folder_path + '/high_res'
    low_res_files = os.listdir(low_path)
    high_res_files = os.listdir(high_path)
    # find the largest sound clip
    max_length = -1
    for low_file in low_res_files:
        t, _ = torchaudio.load(low_path + '/' + low_file)
        if t.shape[1] > max_length:
            max_length = t.shape[1]
    
    # pad w/ 0's, could use nn.mirror possibly
    for low_file in low_res_files:
        t, s = torchaudio.load(low_path + '/' + low_file)
        grow = max_length - t.shape[1]
        padding = nn.ConstantPad1d((grow,0), 0)
        t = padding(t)
        torchaudio.save(low_path + '/' + low_file, t, s)

    # repeat for the high res clips
    for high_file in high_res_files:
        t, s = torchaudio.load(high_path + '/' + high_file)
        grow = max_length - t.shape[1]
        padding = nn.ConstantPad1d((grow,0), 0)
        t = padding(t)
        torchaudio.save(high_path + '/' + high_file, t, s)

    # print("longest clip:", max_length)

def main():
    ################### INPUTS ###################
    input_folder_path = "./VCTK-Corpus-0.92/wav48_silence_trimmed/p225"
    output_folder_path = "./data/p225" # only using this specific voice for now
    mic_str = "mic2" # alternative "mic1"
    high_resolution_sample_rate = 16000
    low_resolution_sample_rate = 4000
    ################### INPUTS ###################

    # make direcotry if it doesn't exist
    if not os.path.exists(output_folder_path + '/low_res'):
        os.makedirs(output_folder_path + '/low_res')

    if not os.path.exists(output_folder_path + '/high_res'):
        os.makedirs(output_folder_path + '/high_res')

    create_data_set(input_folder_path, output_folder_path, mic_str, high_resolution_sample_rate, low_resolution_sample_rate)
    # pad_data(output_folder_path)

if __name__ == '__main__':
    main()