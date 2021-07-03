import pandas as pd
import numpy as np
import os, librosa, glob, shutil, torch
from tqdm import tqdm
from hparams import *


text_dir = './archive/transcript.v.1.4.txt'

metadata = pd.read_csv(text_dir, dtype='object', sep='|', header=None)
wav_dir = metadata[0].values

out_dir = './data'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + '/mel', exist_ok=True)
os.makedirs(out_dir + '/audio', exist_ok=True)

for idx, fn in enumerate(tqdm(wav_dir)):
    file_dir = './archive/kss/' + fn
    wav, _ = librosa.load(file_dir, sr=sample_rate)
    
    wav = torch.from_numpy(wav)
    wav = wav.unsqueeze(0).unsqueeze(0)
    wav = torch.nn.functional.pad(wav, ((n_fft - hop_length) // 2, (n_fft - hop_length) // 2), mode='reflect')
    wav = wav.squeeze(0).squeeze(0)
    spec = torch.stft(wav, n_fft, hop_length, win_length, center=False, window=torch.hann_window(win_length))
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_dim)
    mel_filter = torch.from_numpy(mel_filter)
    mel_spec = torch.matmul(mel_filter, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    mel_spec = mel_spec.numpy()

    mel_name = 'kss-mel-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/mel', mel_name), mel_spec, allow_pickle=False)

    audio_name = 'kss-audio-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/audio', audio_name), wav, allow_pickle=False)

os.makedirs(valid_dir, exist_ok=True)
os.makedirs(valid_dir + '/mel', exist_ok=True)
os.makedirs(valid_dir + '/audio', exist_ok=True)
mel_list = sorted(glob.glob(os.path.join(out_dir + '/mel', '*.npy')))
wav_list = sorted(glob.glob(os.path.join(out_dir + '/audio', '*.npy')))
for i in range(valid_n):
    shutil.move(mel_list[i], valid_dir + '/mel')
    shutil.move(wav_list[i], valid_dir + '/audio')
