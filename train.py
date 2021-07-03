import os, argparse, traceback, glob, librosa, random, itertools, time, torch
import numpy as np
import soundfile as sf
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from gan import *
from hparams import *


class MelDataset(Dataset):
    def __init__(self, mel_list, audio_list):
        self.mel_list = mel_list
        self.audio_list = audio_list

    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        mel = np.load(self.mel_list[idx])
        mel = torch.from_numpy(mel).float()
        start = random.randint(0, mel.size(1) - seq_len - 1)
        mel = mel[:, start : start + seq_len]

        wav = np.load(self.audio_list[idx])
        wav = torch.from_numpy(wav).float()
        start *= hop_length
        wav = wav[start : start + seq_len * hop_length]

        return mel, wav.unsqueeze(0)


def train(args):
    base_dir = 'data'
    mel_list = sorted(glob.glob(os.path.join(base_dir + '/mel', '*.npy')))
    audio_list = sorted(glob.glob(os.path.join(base_dir + '/audio', '*.npy')))
    trainset = MelDataset(mel_list, audio_list)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)

    test_mel = sorted(glob.glob(os.path.join(valid_dir + '/mel', '*.npy')))
    testset = []
    for d in test_mel:
        mel = np.load(d)
        mel = torch.from_numpy(mel).float()
        mel = mel.unsqueeze(0)
        testset.append(mel)

    G = Generator().cuda()
    mpd = MultiPeriodDiscriminator().cuda()
    msd = MultiScaleDiscriminator().cuda()

    optim_g = AdamW(G.parameters(), learning_rate, betas=[b1, b2])
    optim_d = AdamW(itertools.chain(msd.parameters(), mpd.parameters()), learning_rate, betas=[b1, b2])

    step, epochs = 0, -1
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        G.load_state_dict(ckpt['G'])
        optim_g.load_state_dict(ckpt['optim_g'])
        mpd.load_state_dict(ckpt['mpd'])
        msd.load_state_dict(ckpt['msd'])
        optim_d.load_state_dict(ckpt['optim_d'])
        step = ckpt['step'],
        epochs = ckpt['epoch']
        step = step[0]
        print('Load Status: Step %d' % (step))
        
    scheduler_g = ExponentialLR(optim_g, gamma=lr_decay, last_epoch=epochs)
    scheduler_d = ExponentialLR(optim_d, gamma=lr_decay, last_epoch=epochs)

    torch.backends.cudnn.benchmark = True

    start = time.time()
    try:
        for epoch in itertools.count(epochs):
            for (mel, audio) in train_loader:
                x = mel.cuda()
                y = audio.cuda()
                
                y_g_hat = G(x)

                y_pad = torch.nn.functional.pad(y_g_hat,
                                                ((n_fft - hop_length) // 2, (n_fft - hop_length) // 2),
                                                mode='reflect')
                spec = torch.stft(y_pad.squeeze(1), n_fft, hop_length, win_length, center=False,
                                  window=torch.hann_window(win_length).cuda())               
                spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
                mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_dim)
                mel_filter = torch.from_numpy(mel_filter).float().cuda()
                mel_spec = torch.matmul(mel_filter, spec)
                mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
                
                # Discriminator
                optim_d.zero_grad()
                
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                d_loss = loss_disc_s + loss_disc_f
            
                d_loss.backward()
                optim_d.step()

                # Generator
                optim_g.zero_grad()

                loss_mel = F.l1_loss(x, mel_spec) * 45
                
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                
                g_loss = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                g_loss.backward()
                optim_g.step()

                step += 1
                if step % log_step == 0:
                    print('step: {}, D_loss: {:.3f}, G_loss: {:.3f}, {:.3f} sec/step'.format(
                        step, d_loss, g_loss, (time.time() - start) / log_step))
                    start = time.time()

                if step % checkpoint_step == 0:
                    save_dir = './ckpt/' + args.name
                    with torch.no_grad():
                        for i, mel_test in enumerate(testset):
                            g_audio = G(mel_test.cuda())
                            g_audio = g_audio.squeeze().cpu()
                            audio = (g_audio.numpy() * 32768)
                            sf.write(os.path.join(save_dir, 'generated-{}-{}.wav'.format(step, i)),
                                     audio.astype('int16'),
                                     sample_rate)

                    print("Saving checkpoint")
                    torch.save({
                        'G': G.state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'mpd': mpd.state_dict(),
                        'msd': msd.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, os.path.join(save_dir, 'ckpt-{}.pt'.format(step)))
            
            scheduler_g.step()
            scheduler_d.step()

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-p', default=None)
    parser.add_argument('--name', '-n', required=True)
    args = parser.parse_args()
    save_dir = os.path.join('./ckpt', args.name)
    os.makedirs(save_dir, exist_ok=True)
    train(args)
