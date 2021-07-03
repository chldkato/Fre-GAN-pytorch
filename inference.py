import os, argparse, glob, librosa, librosa.display, torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from gan import Generator
from hparams import *


def main(args):
    vocoder = Generator()
    vocoder = vocoder.cuda()
    ckpt = torch.load(args.checkpoint)
    vocoder.load_state_dict(ckpt['G'])
    testset = glob.glob(os.path.join(args.test_dir, '*.wav'))
    for i, test_path in enumerate(tqdm(testset)):
        mel, spectrogram = process_audio(test_path)
        g_audio = vocoder(mel.cuda())
        g_audio = g_audio.squeeze().cpu()
        audio = g_audio * 32768
        g_spec = torch.stft(audio, n_fft, hop_length, win_length, center=False, window=torch.hann_window(win_length))
        g_spec = torch.sqrt(g_spec.pow(2).sum(-1) + 1e-9)
        audio = audio.detach().numpy()
        g_spec = g_spec.detach().numpy()
        sf.write(os.path.join(args.save_dir, 'generated-{}.wav'.format(i)),
                 audio.astype('int16'),
                 sample_rate)
        plot_stft(spectrogram, g_spec, i)


def process_audio(wav_path):
    wav, sr = librosa.load(wav_path, sr=sample_rate)
    wav = torch.from_numpy(wav)
    spec = torch.stft(wav, n_fft, hop_length, win_length, center=False, window=torch.hann_window(win_length))
    specs = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_dim)
    mel_filter = torch.from_numpy(mel_filter)
    mel_spec = torch.matmul(mel_filter, specs)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec.unsqueeze(0), specs


def plot_stft(spectrogram, g_spec, idx):
    plt.figure(figsize=(12, 8))

    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='log', hop_length=256)
    plt.title('original audio spectrogram')

    g_spec = librosa.amplitude_to_db(np.abs(g_spec), ref=np.max)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(g_spec, x_axis='time', y_axis='log', hop_length=256)
    plt.title('generated audio spectrogram')

    plt.tight_layout()
    fn = 'spectrogram-%d.png' % idx
    plt.savefig(args.save_dir + '/' + fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', '-t', default='./test')
    parser.add_argument('--checkpoint', '-p', required=True)
    parser.add_argument('--save_dir', '-s', default='./output')
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    main(args)
