from __future__ import division
from past.utils import old_div
split = True
sel_sz = old_div(512,2)
scale = 2
rescale = 2
numscale = 3
pool_scale = 4
psz = sel_sz/(scale**(numscale-1))/rescale
cachedir = '/home/mayank/work/tensorflow/cache/'
labelfile = '/home/mayank/Dropbox/AdamVideos/M118_M119_M122_M127_M130_M173_M174_M147_20150615.mat'
viddir = '/home/mayank/Dropbox/AdamVideos/multiPoint'
ptn = 'M1[1-2][0-9]'
trainfilename = 'train_lmdb'
valfilename = 'val_lmdb'

valdatafilename = 'valdata'
valratio = 0.3

dist2pos = 5
label_blur_rad = 8.
numNegPerPos = 4

learning_rate = 0.0001
training_iters = 20000
batch_size = 64
display_step = 30

# Network Parameters
n_classes = 1 # 
dropout = 0.5 # Dropout, probability to keep units
nfilt = 128
nfcfile = 512

display_step = 100
map_size = 100000*psz**2*3

outname = 'pawMulti_r2_s3'
save_step = 4000
numTest = 400
imsz = (260,352)