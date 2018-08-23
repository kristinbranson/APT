from poseConfig import aliceConfig as conf
import tensorflow as tf
tf.reset_default_graph()
import multiResData
conf.trange = 5
conf.cachedir += '_dataset'
# conf.dl_steps = 100000
# conf.cos_steps = 4
# import PoseUMDN_dataset
# self = PoseUMDN_dataset.PoseUMDN(conf,name='pose_umdn_test')

import PoseUNet_dataset

self = PoseUNet_dataset.PoseUNet(conf,name='pose_unet_fusion')

self.train_unet()

##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
args =  '-name pend -cache /home/mayank/temp -type unet /home/mayank/work/poseTF/data/apt/pend_1_stripped_preProcDataCache_scale4_NumChans1_v73.lbl train -use_cache'
args = args.split()
import APT_interface as apt
apt.main(args)


##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from poseConfig import aliceConfig as conf
import tensorflow as tf
import multiResData
conf.trange = 5
conf.cachedir += '_dataset'
import mdn_keras
mdn_keras.training(conf)

##
from poseConfig import aliceConfig as conf
import tensorflow as tf
tf.reset_default_graph()
import multiResData
conf.trange = 5
conf.cachedir += '_dataset'
# conf.dl_steps = 100000
# conf.cos_steps = 4
# import PoseUMDN_dataset
# self = PoseUMDN_dataset.PoseUMDN(conf,name='pose_umdn_test')

import PoseUNet_dataset
self = PoseUNet_dataset.PoseUNet(conf,name='pose_unet_orig_layers')
self.train_unet()
tf.reset_default_graph()
V = self.classify_val()
np.percentile(V[0],[90,95,98,99],axis=0)


##


import APT_interface_mdn as apat
conf = apat.create_conf('/home/mayank/Dropbox (HHMI)/temp/20180807T130922_v73.lbl',0,'alice',cache_dir='/home/mayank/temp')
##
args =  '-name a -model_file cache/alice_dataset/aliceFly_pose_umdn_cosine-20000 -cache cache/alice_dataset -type unet data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl track -mov /home/mayank/work/FlySpaceTime/test_umdn_classification/movie.ufmf -trx /home/mayank/work/FlySpaceTime/test_umdn_classification/registered_trx.mat -start_frame 5000 -end_frame 5500 -out /home/mayank/work/FlySpaceTime/test_umdn_classification/umdn_out.trk -hmaps '
args = args.split()
import APT_interface_mdn as apt
apt.main(args)


##


import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from poseConfig import aliceConfig as conf
import tensorflow as tf
import multiResData
conf.trange = 15
conf.cachedir += '_dataset'
conf.batch_size = 16
import PoseUMDN_dataset
self = PoseUMDN_dataset.PoseUMDN(conf,name='pose_umdn_bsz16')
V = self.classify_val()
np.percentile(V[0],[90,95,98,99],axis=0)


##

import PoseUMDN_dataset
from stephenHeadConfig import conf as conf
conf.rescale = 2
conf.n_steps = 3
conf.cachedir = '/home/mayank/work/poseTF/cache/stephen_dataset'
self = PoseUMDN_dataset.PoseUMDN(conf,name='pose_umdn_joint')
self.train_umdn(False)



##

##
import PoseUNet_dataset
from poseConfig import aliceConfig as conf
conf.cachedir += '_dataset'
reload(PoseUNet_dataset)
self = PoseUNet_dataset.PoseUNet(conf)
A = self.classify_val(at_step=20000)

## create deeplabcut db for alice

import  os
import imageio
from poseConfig import aliceConfig as conf
import  multiResData
conf.cachedir += '/deeplabcut'

def deepcut_outfn(data, outdir, count, fis, save_data):
    # pass count as array to pass it by reference.
    if conf.imgDim == 1:
        im = data[0][:, :, 0]
    else:
        im = data[0]
    img_name = os.path.join(outdir, 'img_{:06d}.png'.format(count[0]))
    imageio.imwrite(img_name, im)
    locs = data[1]
    bparts = conf.n_classes
    for b in range(bparts):
        fis[b].write('{}\t{}\t{}\n'.format(count[0], locs[b, 0], locs[b, 1]))
    mod_locs = np.insert(np.array(locs), 0, range(bparts), axis=1)
    save_data.append([img_name, im.shape, mod_locs])
    count[0] += 1


bparts = ['part_{}'.format(i) for i in range(conf.n_classes)]
train_count = [0]
train_dir = os.path.join(conf.cachedir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
train_fis = [open(os.path.join(train_dir, b + '.csv'), 'w') for b in bparts]
train_data = []
val_count = [0]
val_dir = os.path.join(conf.cachedir, 'val')
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
val_fis = [open(os.path.join(val_dir, b + '.csv'), 'w') for b in bparts]
val_data = []
for ndx in range(conf.n_classes):
    train_fis[ndx].write('\tX\tY\n')
    val_fis[ndx].write('\tX\tY\n')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

def train_out_fn(data):
    deepcut_outfn(data, train_dir, train_count, train_fis, train_data)

def val_out_fn(data):
    deepcut_outfn(data, val_dir, val_count, val_fis, val_data)


# collect the images and labels in arrays
out_fns = [train_out_fn, val_out_fn]
in_db_file_train = '/home/mayank/work/poseTF/cache/alice/train_TF.tfrecords'
in_db_file_val = '/home/mayank/work/poseTF/cache/alice/val_TF.tfrecords'

T = multiResData.read_and_decode_without_session(in_db_file_train,conf,())
V = multiResData.read_and_decode_without_session(in_db_file_val,conf,())

for ndx in range(len(T[0])):
    train_out_fn([T[0][ndx], T[1][ndx], T[2][ndx]])

for ndx in range(len(V[0])):
    val_out_fn([V[0][ndx], V[1][ndx], V[2][ndx]])

[f.close() for f in train_fis]
[f.close() for f in val_fis]
import pickle
with open(os.path.join(conf.cachedir, 'train_data.p'), 'w') as f:
    pickle.dump(train_data, f, protocol=2)
with open(os.path.join(conf.cachedir, 'val_data.p'), 'w') as f:
    pickle.dump(val_data, f, protocol=2)

##

import PoseUNet_dataset
reload(PoseUNet_dataset)
from PoseUNet_dataset import PoseUNet
from poseConfig import aliceConfig as conf
import  tensorflow as tf
conf.cachedir += '_dataset'
self = PoseUNet(conf,name='pose_unet_residual')
conf.unet_steps = 20000
self.train_unet(False)
tf.reset_default_graph()
A = self.classify_val(at_step=20000)

##
import tensorflow as tf
from poseConfig import aliceConfig as conf
import PoseTools

db_file = '/home/mayank/Dropbox (HHMI)/temp/alice/full_train_TF.tfrecords'

def _parse_function(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                  'width': tf.FixedLenFeature([], dtype=tf.int64),
                  'depth': tf.FixedLenFeature([], dtype=tf.int64),
                  'trx_ndx': tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
                  'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                  'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                  'ts': tf.FixedLenFeature([], dtype=tf.float32),
                  'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                  })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    trx_ndx = tf.cast(features['trx_ndx'], tf.int64)
    image = tf.reshape(image, conf.imsz + (conf.imgDim,))

    locs = tf.cast(features['locs'], tf.float64)
    exp_ndx = tf.cast(features['expndx'], tf.float64)
    ts = tf.cast(features['ts'], tf.float64)  # tf.constant([0]); #
    info = tf.stack([exp_ndx, ts, tf.cast(trx_ndx,tf.float64)])
    return image, locs, info

dataset = tf.data.TFRecordDataset(db_file)
dataset = dataset.map(_parse_function)
#

extra = []
def preproc_func(ims_in, locs_in, info_in, extra):
    ims = ims_in
    locs = locs_in
    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, True, conf.unet_rescale)
    return ims, locs, info_in

tpre= lambda ims, locs, info: preproc_func(ims,locs,info,extra)

py_map = lambda ims, locs, info: tuple(tf.py_func(
    tpre, [ims, locs, info], [tf.float64, tf.float64, tf.float64]))


dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(8)
dataset = dataset.map(py_map)
dataset = dataset.repeat()

##
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.InteractiveSession()

ff = sess.run(next_element)

##
aa =tf.placeholder(tf.bool)
im = tf.cond(aa,lambda:tf.identity(next_element[0]),lambda:tf.identity(next_element[0]))
kk = sess.run(im, feed_dict={aa:False})

##

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = '-name leap_default -view 0 -cache cache/leap_compare -type leap data/leap/leap_data.lbl train -use_defaults -skip_db'.split()
import APT_interface as apt
apt.main(args)

##
model_file = '/home/mayank/Dropbox (HHMI)/temp/alice/leap/final_model.h5'
lbl_file = '/home/mayank/work/poseTF/data/leap/leap_data.lbl'
cache_dir = '/home/mayank/work/poseTF/cache/leap_db'

import sys
import socket
import  numpy as np
import os

import APT_interface as apt
view = 0
conf = apt.create_conf(lbl_file,0,'leap_db','leap',cache_dir)
apt.create_leap_db(conf, False)

data_path = os.path.join(cache_dir, 'leap_train.h5')
cmd = 'python leap/training_MK.py {}'.format(data_path)
print('RUN: {}'.format(cmd))


##
import APT_interface as apt
import os
import h5py
import logging
reload(apt)

lbl_file = '/home/mayank/work/poseTF/data/stephen/sh_cacheddata_20180717T095200.lbl'

log = logging.getLogger()  # root logger
log.setLevel(logging.ERROR)

cmd = '-view 1 -name sh_cache -cache /home/mayank/work/poseTF/cache/stephen/cache_test {} train -use_cache'.format(lbl_file)

apt.main(cmd.split())

##
import socket
import APT_interface as apt
import os
import shutil
import h5py
import logging
reload(apt)

lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

split_file = '/home/mayank/work/poseTF/cache/apt_interface/multitarget_bubble_view0/test_leap/splitdata.json'

log = logging.getLogger()  # root logger
log.setLevel(logging.ERROR)

import deepcut.train
conf = apt.create_conf(lbl_file,0,'test_openpose_delete')
conf.splitType = 'predefined'
apt.create_tfrecord(conf, True, split_file=split_file)
from poseConfig import config as args
args.skip_db = True
apt.train_openpose(conf,args)

##
import deepcut.train
import  tensorflow as tf
tf.reset_default_graph
conf.batch_size = 1
pred_fn, model_file = deepcut.train.get_pred_fn(conf)
rfn, n= deepcut.train.get_read_fn(conf,'/home/mayank/work/poseTF/cache/apt_interface/multitarget_bubble_view0/test_deepcut/val_data.p')
A = apt.classify_db(conf, rfn, pred_fn, n)

##
import socket
import APT_interface as apt
import os
import shutil
import h5py
import logging

lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

conf = apt.create_conf(lbl_file,view=0,name='test_openpose')
graph =  [ [1,2],[1,3],[2,5],[3,4],[1,6],[6,7],[6,8],[6,10],[8,9],[10,11],[5,12],[9,13],[6,14],[6,15],[11,16],[4,17]]
graph = [[g1-1, g2-1] for g1, g2 in graph]
conf.op_affinity_graph = graph

from poseConfig import config as args
# APT_interface.create_leap_db(conf,True)

# conf.batch_size = 32
# conf.rrange = 15
# conf.dl_steps = 2500

log = logging.getLogger()  # root logger
log.setLevel(logging.INFO)

args.skip_db = True
apt.train_openpose(conf, args)


##

import  h5py
import APT_interface as apt
lbl_file = '/home/mayank/work/poseTF/data/stephen/sh_trn4523_gt080618_made20180627_stripped.lbl'
from stephenHeadConfig import sideconf as conf
conf.labelfile = lbl_file

conf.cachedir = '/home/mayank/work/poseTF/cache/stephen'
from poseConfig import config as args
args.skip_db = False
apt.train_unet(conf,args)
##
import socket
import APT_interface
import os
import shutil
import h5py

lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

conf = APT_interface.create_conf(lbl_file,view=0,name='test_leap')
# APT_interface.create_leap_db(conf,True)

conf.batch_size = 32
conf.rrange = 15
# conf.dl_steps = 2500

db_path = [os.path.join(conf.cachedir, 'leap_train.h5')]
db_path.append(os.path.join(conf.cachedir, 'leap_val.h5'))
import leap.training
reload(leap.training)
from leap.training import train_apt
train_apt(db_path,conf,'test_leap')

## test leap db creation
import APT_interface
lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

conf = APT_interface.create_conf(lbl_file,view=0,name='stacked_hourglass')
APT_interface.create_leap_db(conf,True)

##


import socket
import APT_interface
import os
import shutil
import h5py

if socket.gethostname() == 'mayankWS':
    in_lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'
    lbl_file = '/home/mayank/work/poseTF/data/apt/alice_test_apt.lbl'
else:
    in_lbl_file = ''
    lbl_file = None


shutil.copyfile(in_lbl_file,lbl_file)
H = h5py.File(lbl_file,'r+')

H[H['trackerData'][1,0]]['sPrm'].keys()


