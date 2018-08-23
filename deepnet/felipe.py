import pandas as pd
import json

json_file= pd.read_json('cache/felipe/annotations.json')
im_dir = 'cache/felipe/Data'
ims = json_file.keys()

n_classes = 5
im = ims[0]
part_keys = None

def get_xy(ann):
    return np.array([ann['cx'],ann['cy']])

all_lbls = []
for im in ims:
    anns = json_file[im]
    n_anns = len(anns)
    assert n_anns==17, 'incorrect number of animals'
    lbls = np.zeros([n_anns,n_classes,2])
    lbls[:] = np.nan
    if not part_keys:
        part_keys = anns[0].keys()
    for ndx in range(n_anns):
        cur_ann = anns[ndx]
        for pndx,p in enumerate(part_keys):
            if isinstance(cur_ann,dict):
                lbls[ndx,pndx,:] = get_xy(cur_ann[p])

    all_lbls.append(lbls)

all_lbls = np.array(all_lbls)

## view the labels
scale = 1 # 0.25
from scipy import misc
import os
r_im = np.random.randint(len(ims))
sel_im = os.path.join('cache/felipe/Data',ims[r_im])
im = misc.imread(sel_im)
cur_l = all_lbls[r_im]

sel_pts = np.array([1,3,4])
plt.figure()
plt.imshow(misc.imresize(im,scale))
plt.scatter(cur_l[:,sel_pts,0].flatten()*scale,cur_l[:,sel_pts,1].flatten()*scale)

## create tf records file

from poseConfig import felipeConfig as conf
import tensorflow as tf
from multiResData import int64_feature, bytes_feature, float_feature
import os
from scipy import misc

split = True
isval = np.random.choice(len(ims),int(len(ims)*conf.valratio))

if split:
    trainfilename = os.path.join(conf.cachedir, conf.trainfilename)
    valfilename = os.path.join(conf.cachedir, conf.valfilename)

    env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
    valenv = tf.python_io.TFRecordWriter(valfilename + '.tfrecords')
else:
    trainfilename = os.path.join(conf.cachedir, conf.fulltrainfilename)
    env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
    valenv = None

pts = all_lbls
count = 0
valcount = 0
psz = conf.sel_sz

h_ndx = part_keys.index('head')
t_ndx = part_keys.index('Tail')
hsz = conf.imsz[0]/2
rescale = 0.5
for ndx, dir_name in enumerate(ims):

    expname = 'felipe_exp'
    curpts = all_lbls[ndx,...]*rescale
    # frames correspond to animals in a frame now.
    frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
    curdir = 'cache/felipe/Data'
    curenv = valenv if ndx in isval and split else env

    im = misc.imread(os.path.join('cache/felipe/Data', dir_name))
    im = misc.imresize(im,rescale)
    for fnum in frames:
        framein = im.copy()
        framein = np.pad(framein,[[hsz,hsz],[hsz,hsz],[0,0]],'constant')
        cc = (curpts[fnum,h_ndx,:] + curpts[fnum,t_ndx,:])/2
        cc = cc.astype('int')
        framein = framein[cc[1]:cc[1]+hsz*2,cc[0]:cc[0]+hsz*2,:]
        curloc = curpts[fnum, :, :].copy()
        curloc[:, 0] = curloc[:, 0] - cc[0] + hsz
        curloc[:, 1] = curloc[:, 1] - cc[1] + hsz

        rows = framein.shape[0]
        cols = framein.shape[1]
        if np.ndim(framein) > 2:
            depth = framein.shape[2]
        else:
            depth = 1

        image_raw = framein.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(rows),
            'width': int64_feature(cols),
            'depth': int64_feature(depth),
            'locs': float_feature(curloc.flatten()),
            'expndx': float_feature(ndx),
            'ts': float_feature(fnum),
            'image_raw': bytes_feature(image_raw)}))
        curenv.write(example.SerializeToString())

        if ndx in isval and split:
            valcount += 1
        else:
            count += 1

    print('Done %d of %d movies, count:%d val:%d' % (ndx, len(ims), count, valcount))
env.close()  # close the database
if split:
    valenv.close()
print('%d,%d number of pos examples added to the db and valdb' % (count, valcount))

## view the database
from poseConfig import felipeConfig as conf
import PoseTools

oim,dim,locs = PoseTools.gen_distorted_images(conf)
fig,ax = plt.subplots(1,2)
sel = np.random.randint(conf.batch_size)
ax[0].imshow(np.transpose(oim[sel,:,:,:],[1,2,0])); ax[0].scatter(locs[sel,:,0],locs[sel,:,1])
pp = dim[sel,...].astype('uint8')
ax[1].imshow(pp); ax[1].scatter(locs[sel,:,0],locs[sel,:,1])

## multi res network
from poseConfig import felipeConfig as conf
import PoseTrain

self = PoseTrain.PoseTrain(conf)
self.baseTrain(restore=False,trainType=0,trainPhase=True)

## val results
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from poseConfig import felipeConfig as conf
import PoseTrain
import PoseTools

self = PoseTrain.PoseTrain(conf)
val_dist, val_ims, val_preds, val_predlocs, val_locs = self.classify_val_base()

kk = np.percentile(val_dist,[80,90,95],axis=0)
pp = np.transpose(val_ims[0,...].astype('uint8'),[1,2,0])
PoseTools.create_result_image(pp,val_locs[0,...],kk)


## U-Net.

from poseConfig import felipeConfig as conf
import PoseUNet
conf.mdn_min_sigma = 10.
conf.mdn_max_sigma = 15.
self = PoseUNet.PoseUNet(conf,name='pose_unet')
self.train_unet(restore=False,train_type=0)

## U-Net results

from poseConfig import felipeConfig as conf
import PoseUNet
import os
import PoseTools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
conf.mdn_min_sigma = 10.
conf.mdn_max_sigma = 15.

self = PoseUNet.PoseUNet(conf,name='pose_unet')

val_dist, val_ims, val_preds, val_predlocs, val_locs = self.classify_val()

kk = np.percentile(val_dist,[80,90,95],axis=0)
pp = val_ims[0,...].astype('uint8')
PoseTools.create_result_image(pp,val_locs[0,...],kk)

##
from poseConfig import felipeConfig as conf
import PoseUMDN
conf.mdn_min_sigma = 10.
conf.mdn_max_sigma = 15.
self = PoseUMDN.PoseUMDN(conf,name='pose_umdn')
self.train_umdn(restore=True,train_type=0, joint=True)

## U-MDN results

from poseConfig import felipeConfig as conf
import PoseUMDN
import os
import PoseTools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
conf.mdn_min_sigma = 10.
conf.mdn_max_sigma = 15.

self = PoseUMDN.PoseUMDN(conf,name='pose_umdn')

val_dist, val_ims, val_preds, val_predlocs, val_locs = self.classify_val()

kk = np.percentile(val_dist,[80,90,95],axis=0)
pp = val_ims[0,...].astype('uint8')
PoseTools.create_result_image(pp,val_locs[0,...],kk)


## for multi animal..

## create tf records file

import pandas as pd
import json

json_file= pd.read_json('cache/felipe/annotations.json')
im_dir = 'cache/felipe/Data'
ims = json_file.keys()

n_classes = 5
im = ims[0]
part_keys = None

def get_xy(ann):
    return np.array([ann['cx'],ann['cy']])

all_lbls = []
for im in ims:
    anns = json_file[im]
    n_anns = len(anns)
    assert n_anns==17, 'incorrect number of animals'
    lbls = np.zeros([n_anns,n_classes,2])
    lbls[:] = np.nan
    if not part_keys:
        part_keys = anns[0].keys()
    for ndx in range(n_anns):
        cur_ann = anns[ndx]
        for pndx,p in enumerate(part_keys):
            if isinstance(cur_ann,dict):
                lbls[ndx,pndx,:] = get_xy(cur_ann[p])

    all_lbls.append(lbls)

all_lbls = np.array(all_lbls)

max_n = all_lbls.shape[1]

from poseConfig import felipe_config_multi as conf
import tensorflow as tf
from multiResData import int64_feature, bytes_feature, float_feature
import os
from scipy import misc

split = True
isval = np.random.choice(len(ims),int(len(ims)*conf.valratio))

if split:
    trainfilename = os.path.join(conf.cachedir, conf.trainfilename)
    valfilename = os.path.join(conf.cachedir, conf.valfilename)

    env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
    valenv = tf.python_io.TFRecordWriter(valfilename + '.tfrecords')
else:
    trainfilename = os.path.join(conf.cachedir, conf.fulltrainfilename)
    env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
    valenv = None

pts = all_lbls
count = 0
valcount = 0
psz = conf.sel_sz

h_ndx = part_keys.index('head')
t_ndx = part_keys.index('Tail')
hsz = 320
rescale = 0.25
n_ims = 2 # split the image horizontally.
# tail-head distance is roughly 80. pad enough so that the
# animal as a whole is one of the two halfs
pad = 60
for ndx, dir_name in enumerate(ims):

    expname = 'felipe_exp'
    curpts = all_lbls[ndx,...]*rescale
    # frames correspond to animals in a frame now.
    frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
    curdir = 'cache/felipe/Data'
    curenv = valenv if ndx in isval and split else env

    im = misc.imread(os.path.join('cache/felipe/Data', dir_name))
    im = misc.imresize(im,rescale)

    for fnum in range(n_ims):
        cc = (curpts[:,h_ndx,:] + curpts[:,t_ndx,:])/2
        curloc = curpts[:, :, :].copy()
        temploc = curloc.copy()
        curloc[:] = np.nan

        if fnum is 0: # left side
            framein = im.copy()[:,:hsz,:]
            framein = np.pad(framein,
                             [[0,0],[0,pad],[0,0]],'constant')
            sel_animal = np.invert((cc[:, 0] > hsz) | np.isnan(cc[:, 0]))
            cur_n = np.count_nonzero(sel_animal)
            curloc[:cur_n,:,:] = temploc[sel_animal, :, :]
        else: # right side
            framein = im.copy()[:,hsz:,:]
            framein = np.pad(framein,
                             [[0,0],[0,pad],[0,0]],'constant')
            sel_animal = np.invert((cc[:,0] <= hsz) | np.isnan(cc[:,0]))
            cur_n = np.count_nonzero(sel_animal)
            curloc[:cur_n,:,:] = temploc[sel_animal, :, :]
            curloc[:,:,0] -= hsz

        curloc = np.round(curloc)[:,conf.selpts,...]
        rows = framein.shape[0]
        cols = framein.shape[1]
        if np.ndim(framein) > 2:
            depth = framein.shape[2]
        else:
            depth = 1

        image_raw = framein.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(rows),
            'width': int64_feature(cols),
            'depth': int64_feature(depth),
            'locs': float_feature(curloc.flatten()),
            'n_animals': int64_feature(cur_n),
            'expndx': float_feature(ndx),
            'ts': float_feature(fnum),
            'image_raw': bytes_feature(image_raw)}))
        curenv.write(example.SerializeToString())

        if ndx in isval and split:
            valcount += 1
        else:
            count += 1

    print('Done %d of %d movies, count:%d val:%d' % (ndx, len(ims), count, valcount))
env.close()  # close the database
if split:
    valenv.close()
print('%d,%d number of pos examples added to the db and valdb' % (count, valcount))


## U-Net multi

from poseConfig import felipe_config_multi as conf
from PoseUNet import PoseUNetMulti
self = PoseUNetMulti(conf)
self.train_unet(restore=True,train_type=0)

##

from poseConfig import felipe_config_multi as conf
from PoseUNet import PoseUNetMulti
import os
import PoseTools
os.environ['CUDA_VISIBLE_DEVICES'] = ''

self = PoseUNetMulti(conf)

val_dist, val_ims, val_preds, val_locs = self.classify_val_m()


## test data for multi Unet

from poseConfig import felipe_config_multi as conf
from PoseUNet import PoseUNetMulti
import  tensorflow as tf
import math
self = PoseUNetMulti(conf)

train_type= 0
restore = False
td_fields = ['dist','loss']
pred_in_key = 'y'

self.init_train(train_type)
self.pred = self.create_network()
def loss(pred_in, pred_out):
    return tf.nn.l2_loss(pred_in-pred_out)
self.cost = loss(self.pred, self.ph[pred_in_key])
self.create_optimizer()
self.create_saver()
num_val_rep = self.conf.numTest / self.conf.batch_size + 1

sess = tf.InteractiveSession()
self.create_cursors(sess)
step = 0
ex_count = step * self.conf.batch_size
cur_lr = 0.0001 * self.conf.gamma ** math.floor(ex_count/ self.conf.step_size)
self.fd[self.ph['learning_rate']] = cur_lr
self.fd_train()
self.update_fd(self.DBType.Train, sess, True)


## U-MDN multi

from poseConfig import felipe_config_multi as conf
import PoseUMDN
conf.mdn_min_sigma = 8.
conf.mdn_max_sigma = 12.
self = PoseUMDN.PoseUMDNMulti(conf,name='pose_umdn')
self.train_umdn(restore=False,train_type=0, joint=True)

## U-MDN multi results

from poseConfig import felipe_config_multi as conf
import PoseUMDN
import os
import PoseTools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
conf.mdn_min_sigma = 10.
conf.mdn_max_sigma = 15.

self = PoseUMDN.PoseUMDNMulti(conf,name='pose_umdn')

val_dist, val_ims, val_preds, val_predlocs, val_locs, \
val_out = self.classify_val()
val_means, val_std, val_wts = val_out

##

pp = np.zeros(val_dist.shape)
qq = np.zeros(val_predlocs.shape)
for ndx in range(val_means.shape[0]):
    sel_ex = val_wts[ndx,:] > 0
    jj = val_means[ndx,...][np.newaxis,sel_ex,:,:] - \
         val_locs[ndx,...][:,np.newaxis,:,:]
    dd = np.sqrt(np.sum(jj**2,axis=-1)).min(axis=1)
    pp[ndx,:] = dd
pp[val_locs[...,0]<-1000] = np.nan

ll = np.where(pp[:,:,:].sum(axis=2)>20)
sel = -1
##
plt.close('all')
f1 = plt.figure()
# f2 = plt.figure()
tr = 1.
for sel in range(len(ll[0])):
    # sel += 1;
    im = ll[0][sel]; an = ll[1][sel]
    # sel += 1; im = sel; an = 0
    # plt.close('all')
    plt.figure(f1.number); plt.clf()
    plt.imshow(val_ims[im,...])
    plt.scatter(val_locs[im,an,:,0], val_locs[im,an,:,1])
    sel_ex = np.where(val_wts[im,:] > 0)[0]
    for ndx in sel_ex:
        plt.plot(val_means[im,ndx,:,0], val_means[im,ndx,:,1],
                 linewidth=val_wts[im,ndx]/2)
    # plt.figure(f2.number); plt.clf()
    # plt.imshow(val_preds[im,...,0])
    # plt.scatter(val_locs[im,an,0,0], val_locs[im,an,0,1])
    # plt.scatter(val_means[im,:,0,0],val_means[im,:,0,1],
    #             s=np.clip(qq[im,:]*1000,0,20), marker='o')
    # plt.savefig('/home/mayank/work/poseTF/results/felipe/MDN_Multi_{}.png'.format(sel))
    plt.pause(2)
