import os
import tensorflow
import hdf5storage
import h5py
from scipy import io as sio


base_dir = '/home/mayank/Dropbox (HHMI)/temp'
data_file = os.path.join(base_dir, 'trnDataSH_Apr18.mat')
info_file = os.path.join(base_dir, 'trnSplits_20180418T173507.mat')

A = sio.loadmat(info_file)
B = h5py.File(data_file)





