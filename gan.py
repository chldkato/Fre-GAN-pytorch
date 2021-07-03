import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module, ModuleList, Sequential, Conv1d, ConvTranspose1d, Upsample, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm
from pytorch_wavelets import DWT1D
from hparams import *


class ResBlock(Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.convs1 = ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
            weight_norm(Conv1d(channels, channels, kernel_size, dilation=dilation[3],
                               padding=get_padding(kernel_size, dilation[3])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x
    

class Generator(Module):
    def __init__(self):
        super().__init__()
        resblock = ResBlock
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
         
        self.conv_pre = weight_norm(Conv1d(80, 512, 7, 1, padding=3)) 
        
        self.ups, self.nn, self.condition = ModuleList(), ModuleList(), ModuleList()
        for i, (k, u) in enumerate(zip(upsample_kernel_sizes, upsample_rates)):
            self.ups.append(weight_norm(
                ConvTranspose1d(512 // (2 ** i), 512 // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))
            
            if i < self.num_upsamples - 1:
                cond_dim = 80 if i == 0 else 512 // (2 ** i)
                self.condition.append(weight_norm(
                    ConvTranspose1d(cond_dim, 512 // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))
            
            if i > 1:
                self.nn.append(Sequential(
                    Upsample(scale_factor=u),
                    weight_norm(Conv1d(512 // (2 ** i), 512 // (2 ** (i + 1)), 1))
                ))
        
        self.resblocks = ModuleList()
        for i in range(len(self.ups)):
            ch = 512 // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d))
        
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
     
    
    def forward(self, x):
        cond = x
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            if i > 0:
                x += cond
            
            if i < self.num_upsamples - 1:
                cond = self.condition[i](cond)
                
            x = F.leaky_relu(x, 0.1)    
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            if i > 0:
                if i == 1:
                    upsample = self.nn[i - 1](x)
                else:
                    upsample += x
                    if i < 4:
                        upsample = self.nn[i - 1](upsample)

        x = F.leaky_relu(upsample)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class DiscriminatorP(Module):
    def __init__(self, period, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs1 = ModuleList([
            norm_f(Conv2d(1, 32, (5, 1), (2, 1), padding=(2, 0))),       # (16, 32, 2048, 2)
            norm_f(Conv2d(32, 128, (5, 1), (2, 1), padding=(2, 0))),     # (16, 128, 1024, 2)
            norm_f(Conv2d(128, 512, (5, 1), (2, 1), padding=(2, 0))),    # (16, 512, 512, 2)
            norm_f(Conv2d(512, 1024, (5, 1), (4, 1), padding=(2, 0))),   # (16, 1024, 128, 2)
            norm_f(Conv2d(1024, 1024, (10, 1), (8, 1), padding=(2, 0)))  # (16, 1024, 16, 2)
        ])
        
        self.convs2 = ModuleList([
            norm_f(Conv1d(2, 1, 1)),
            norm_f(Conv1d(4, 1, 1)),
            norm_f(Conv1d(8, 1, 1))
        ])
        
        self.convs3 = ModuleList([
            norm_f(Conv2d(1, 32, 1)),
            norm_f(Conv2d(1, 128, 1)),
            norm_f(Conv2d(1, 512, 1))
        ])    
    
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), padding=(1, 0)))
        
        self.dwt = DWT1D(wave='db1', mode='symmetric')

        
    def forward(self, x, idx):
        fmap = []
        wavelet = [x]          
        x = reshape_p(x, self.period)
        for i, l in enumerate(self.convs1):
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
            
            if i < 3:
                nw = []
                for w in wavelet:
                    cA, cD = self.dwt(w)
                    nw.append(cA)
                    nw.append(cD[0])
                wavelet = nw
                nw = torch.cat(nw, dim=1)
                nw = self.convs2[i](nw)
                nx = reshape_p(nw, self.period)
                nx = self.convs3[i](nx.cuda())
                x += nx
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
            
        return x, fmap


class MultiPeriodDiscriminator(Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, i)
            y_d_g, fmap_g = d(y_hat, i)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs1 = ModuleList([
            norm_f(Conv1d(1, 128, 41, 2, padding=20)),                 # (16, 128, 4096)
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),     # (16, 128, 2048)
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),    # (16, 256, 1024)
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),    # (16, 512, 256)
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),   # (16, 1024, 64)
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),  # (16, 1024, 64)
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),               # (16, 1024, 64)
        ])
        
        self.convs2 = ModuleList([
            norm_f(Conv1d(2, 1, 1)),
            norm_f(Conv1d(4, 1, 1)) 
        ])
        
        self.convs3 = ModuleList([
            norm_f(Conv1d(1, 128, 1)),
            norm_f(Conv1d(1, 128, 1))
        ])
        
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))
        
        self.dwt = DWT1D(wave='db1', mode='symmetric')

    def forward(self, x):
        fmap = []
        wavelet = [x]
        for i, l in enumerate(self.convs1):
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

            if i < 2:
                nw = []
                for w in wavelet:
                    cA, cD = self.dwt(w)
                    nw.append(cA)
                    nw.append(cD[0])
                wavelet = nw
                nw = torch.cat(nw, dim=1)
                nw = self.convs2[i](nw)
                nx = self.convs3[i](nw)
                x += nx
                
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        
        self.dwt = ModuleList([
            DWT1D(wave='db1', mode='symmetric'),
            DWT1D(wave='db1', mode='symmetric')
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        ny, ny_hat = [y], [y_hat]
        for i, d in enumerate(self.discriminators):
            if i != 0:
                nw = []
                for w in ny:
                    cA, cD = self.dwt[i - 1](w)
                    nw.append(cA)
                    nw.append(cD[0])
                ny = nw
                y = torch.cat(ny, dim=-1)
                
                nw_hat = []
                for w in ny_hat:
                    cA, cD = self.dwt[i - 1](w)
                    nw_hat.append(cA)
                    nw_hat.append(cD[0])
                ny_hat = nw_hat
                y_hat = torch.cat(ny_hat, dim=-1)
                
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
        
        
def reshape_p(x, period):
    b, c, t = x.shape
    if t % period != 0:
        n_pad = period - (t % period)
        x = F.pad(x, (0, n_pad), "reflect")
        t = t + n_pad
    return x.view(b, c, t // period, period)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2
     