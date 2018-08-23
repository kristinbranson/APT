##
import  tensorflow as tf
import PoseUNetAttention
reload(PoseUNetAttention)
from poseConfig import aliceConfig as conf
import PoseTools

tf.reset_default_graph()
self = PoseUNetAttention.PoseUNetAttention(conf)
self.train_unet(False,0)


## create the d
import  tensorflow as tf
import PoseUNetAttention
reload(PoseUNetAttention)
from poseConfig import aliceConfig as conf
import PoseTools

tf.reset_default_graph()
self = PoseUNetAttention.PoseUNetAttention(conf)
self.create_tfrecord_trx(True)

##
import  tensorflow as tf
import PoseUNetAttention
reload(PoseUNetAttention)
from poseConfig import aliceConfig as conf
import PoseTools

tf.reset_default_graph()
self = PoseUNetAttention.PoseUNetAttention(conf,unet_name='pose_unet_128_bs8')
sess = self.init_net(0,True)
xx = self.fd[self.ph['scores']]
xx[0,:,:] = xx[0,0,:]
ss = sess.run(self.debug_layers, self.fd)

##
import PoseUNetAttention
import tensorflow
import pickle
import multiResData
import os

from poseConfig import aliceConfig as conf

self = PoseUNetAttention.PoseUNetAttention(conf,unet_name='pose_unet_128_bs8')

self.create_tfrecord_trx(True,distort=True)


##


import PoseUNetAttention
import tensorflow
import pickle
from multiResData import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from poseConfig import aliceConfig as conf

conf.batch_size = 1

self = PoseUNetAttention.PoseUNetAttention(conf,unet_name='pose_unet_128_bs8')
tfile = os.path.join(conf.cachedir, conf.trainfilename + '_att.tfrecords')
im, locs, layer_out, exp_id, t, trx_ndx, max_scores = PoseUNetAttention.read_and_decode_without_session(tfile,conf,1,10)

exp_id = exp_id[0]
t = t[0]
trx_ndx = trx_ndx[0]
#
sess = self.dep_nets.setup_net(0, True)
unet = self.dep_nets

is_val, local_dirs, _ = load_val_data(conf)
L = h5py.File(conf.labelfile, 'r')
trx_files = get_trx_files(L, local_dirs)
cap = movies.Movie(local_dirs[exp_id])
T = sio.loadmat(trx_files[trx_ndx])['trx'][0]
cur_trx = T[trx_ndx]
##

hist = np.random.randint(self.att_hist)
cur_fnum = t - hist - 1

if not check_fnum(cur_fnum, cap, 'movie', 0):
    print('Value out of range')


framein, curloc = get_patch_trx(cap, cur_trx, cur_fnum, conf.imsz[0], np.zeros([conf.n_classes,2]))
framein = framein[:, :, 0:conf.imgDim]

cur_xs, _ = PoseTools.preprocess_ims(framein[np.newaxis, ...], curloc[np.newaxis, ...], self.conf, distort=False, scale=self.conf.unet_rescale)

unet.fd[unet.ph['phase_train']] = False
unet.fd[unet.ph['keep_prob']] = 1

unet.fd[unet.ph['x']] = cur_xs

sel_layers = [-self.dep_nets.n_conv * x for x in self.att_layers]
preds = [unet.all_layers[s] for s in sel_layers]

oo = sess.run(preds, unet.fd)
layer_out = np.reshape(layer_out[0],[128,12,12,128])
kk = oo[0][0,...] - layer_out[hist,...]
print(np.abs(kk.flatten()).max())



