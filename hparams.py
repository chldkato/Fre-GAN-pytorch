sample_rate = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
mel_dim = 80

upsample_rates = [8, 4, 2, 2, 2]
upsample_kernel_sizes = [16, 8, 4, 4, 4]
resblock_kernel_sizes = [3, 7, 11]
resblock_dilation_sizes = [[1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7]]

batch_size = 16
checkpoint_step = 2000
log_step = 1
learning_rate = 0.0001
b1 = 0.8
b2 = 0.999
lr_decay = 0.999
lambda_feat = 10
seq_len = 32

checkpoint_dir = './ckpt'
valid_dir = './valid'
valid_n = 1