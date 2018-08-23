## for debug..
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import PoseUNet
from poseConfig import aliceConfig as conf
import h5py
import multiResData
import tensorflow as tf
import numpy as np
import PoseTools
import pickle
from scipy import  io as sio
import math

self = PoseUNet.PoseUNet(conf, name='pose_unet_128_bs8')
_,dirs,_ = multiResData.load_val_data(conf)
for ndx in range(len(dirs)):
    dirs[ndx] = dirs[ndx].replace('$dataroot','/home/mayank/work/FlySpaceTime')

trx_file = dirs[2].replace('movie.ufmf', 'registered_trx.mat')
self.create_pred_movie_trx(dirs[2],'/home/mayank/temp/aliceOut_JHS_fly6_0_300.avi', trx_file, 6, max_frames = 300, start_at = 0, trace=True)




##

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import PoseUNet
from poseConfig import aliceConfig as conf
import h5py
import multiResData
import tensorflow as tf
import numpy as np
import PoseTools

self = PoseUNet.PoseUNet(conf, name='pose_unet_128_bs8')

val_dist, val_ims, val_preds, val_predlocs, val_locs, val_info = self.classify_val(0, -1)

#
L = h5py.File(conf.labelfile, 'r')
curpts = multiResData.trx_pts(L,0)

_,dirs,_ = multiResData.load_val_data(conf)
for ndx in range(len(dirs)):
    dirs[ndx] = dirs[ndx].replace('$dataroot','/home/mayank/work/FlySpaceTime')

#

tf.reset_default_graph()
sess = self.init_net(0,True)
selpt = 13
sel = np.where(val_dist[:,selpt]>10)[0]
# sel = [546, 553, 567, 572, 653, 654, 671, 672, 681, 721, 722, 749]

ims = []; preds= []; pred_locs = []
for ndx in range(len(sel)):
    exp,t  = val_info[sel[ndx],:].astype('int')
    start_at = max(0,t-100)
    trx_file = dirs[exp].replace('movie.ufmf','registered_trx.mat')
    cur_pl, cur_i, cur_p = self.classify_movie_trx(dirs[exp], trx_file, sess, end_frame=t + 1, start_frame=start_at, return_ims=True)
    preds.append(cur_p)
    ims.append(cur_i)
    pred_locs.append(cur_pl)


##
ex = np.random.choice(len(ims))
exp, t = val_info[sel[ex], :].astype('int')
curpts = multiResData.trx_pts(L,exp)
t_ndx = np.where(np.invert(np.isnan(curpts[:,t,0,0])))[0]
if not t_ndx.size == 1:
    print('many ts')
t_ndx = t_ndx[0]

pp = preds[ex][:,t_ndx,:,:,selpt].max(axis=(1,2))
conf_ndx = np.where(pp>0.5)[0][-1]

f,ax = plt.subplots(1,4)
ax = ax.flatten()

ax[0].imshow(ims[ex][conf_ndx,t_ndx,:,:,0],'gray')
ax[0].scatter(pred_locs[ex][conf_ndx,t_ndx,:,0],pred_locs[ex][conf_ndx,t_ndx,:,1])
ax[0].set_title('{}'.format(conf_ndx))

ax[1].imshow(ims[ex][-1,t_ndx,:,:,0],'gray')
ax[1].scatter(pred_locs[ex][-1,t_ndx,:,0],pred_locs[ex][-1,t_ndx,:,1])

ax[2].imshow(val_ims[sel[ex],:,:,0],'gray')
ax[2].scatter(val_predlocs[sel[ex],:,0],val_predlocs[sel[ex],:,1])

ax[2].scatter(val_locs[sel[ex],:,0],val_locs[sel[ex],:,1], marker='+')


ax[3].plot(pp)

##

zz = np.zeros(self.fd[self.ph['x']].shape)
zz[1,...] = val_ims[sel[ex],...]
zz[2,...] = ims[ex][-1,t_ndx,...]
self.fd[self.ph['x']] = zz
pp = sess.run(self.pred, self.fd)

f,ax = plt.subplots(1,2)
ax = ax.flatten()
ax[0].imshow(pp[1,:,:,16])
ax[1].imshow(pp[2,:,:,16])

##

trx_file = dirs[2].replace('movie.ufmf', 'registered_trx.mat')
self.create_pred_movie_trx(dirs[2],'/home/mayank/temp/aliceOut_JHS_fly6_0_300.avi', trx_file, 6, max_frames = 300, start_at = 0, trace=True)

##


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import PoseUNet
from poseConfig import aliceConfig as conf
import h5py
import multiResData
import tensorflow as tf
import numpy as np
import PoseTools
import pickle
from scipy import  io as sio
import math

self = PoseUNet.PoseUNet(conf, name='pose_unet_128_bs8')
# _,dirs,_ = multiResData.loadValdata(conf)
# for ndx in range(len(dirs)):
#     dirs[ndx] = dirs[ndx].replace('$dataroot','/home/mayank/work/FlySpaceTime')

dirs = ['/home/mayank/work/FlySpaceTime/cx_GMR_SS00077_CsChr_RigD_20150930T134055/movie.ufmf','/home/mayank/work/FlySpaceTime/cx_GMR_SS00168_CsChr_RigD_20150909T111218/movie.ufmf']

tf.reset_default_graph()
sess = self.init_net(0, True)
chunk_size = 5000
for cur_dir in dirs:

    trx_file = cur_dir.replace('movie.ufmf','registered_trx.mat')
    T = sio.loadmat(trx_file)['trx'][0]
    n_trx = len(T)

    end_frames = np.array([x['endframe'][0, 0] for x in T])
    n_chunks = int(math.ceil(float(end_frames.max())/chunk_size))

    for cur_c in range(n_chunks):
        print('+++++ Chunk:{} +++++'.format(cur_c))
        cur_pl= self.classify_movie_trx(cur_dir, trx_file, sess, start_frame=cur_c * chunk_size, end_frame=(cur_c + 1) * chunk_size, return_ims=False)
        out_file = cur_dir.replace('movie.ufmf','unet_res_{}.pl'.format(cur_c))
        with open(out_file,'wb') as f:
            pickle.dump(cur_pl,f)



##

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import PoseUNet
from poseConfig import aliceConfig as conf
import h5py
import multiResData
import tensorflow as tf
import numpy as np
import PoseTools
import pickle
from scipy import  io as sio
import math

from datetime import datetime, timedelta

def datetime2matlabdn(dt):
    ord = dt.toordinal()
    mdn = dt + timedelta(days = 366)
    frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac

# _,dirs,_ = multiResData.loadValdata(conf)
# for ndx in range(len(dirs)):
#     dirs[ndx] = dirs[ndx].replace('$dataroot','/home/mayank/work/FlySpaceTime')


dirs = ['/home/mayank/work/FlySpaceTime/cx_GMR_SS00077_CsChr_RigD_20150930T134055/movie.ufmf','/home/mayank/work/FlySpaceTime/cx_GMR_SS00168_CsChr_RigD_20150909T111218/movie.ufmf']


chunk_size = 5000
for cur_dir in reversed(dirs):

    trx_file = cur_dir.replace('movie.ufmf','registered_trx.mat')
    T = sio.loadmat(trx_file)['trx'][0]
    n_trx = len(T)

    end_frames = np.array([x['endframe'][0, 0] for x in T])
    n_frames = end_frames.max()
    n_chunks = int(math.ceil(float(end_frames.max())/chunk_size))
    nTrx = T.shape[0]
    trk = np.zeros([conf.n_classes, 2, n_frames, nTrx])
    trkTs = np.ones([conf.n_classes,  n_frames, nTrx]) * datetime2matlabdn(datetime.now())
    trkTag = np.zeros([conf.n_classes,  n_frames, nTrx]).astype(np.object)
    trkFrm = np.arange(n_frames)+1
    trkiTgt = np.arange(n_trx)+1
    trkiPt = np.arange(conf.n_classes)+1

    print('+++++ Exp:{} +++++'.format(cur_dir ))
    for cur_c in range(n_chunks):
        out_file = cur_dir.replace('movie.ufmf','unet_res_{}.pl'.format(cur_c))
        with open(out_file,'rb') as f:
            cur_pl = pickle.load(f)
        st = cur_c*chunk_size
        en = st+cur_pl.shape[0]

        trk[:,:,st:en,:] = cur_pl.transpose([2,3,0,1])

    out_file = cur_dir.replace('movie.ufmf','unet_results.trk')
    sio.savemat(out_file, {'pTrk':trk,'pTrkTS':trkTs, 'pTrkTag':trkTag,'pTrkFrm':trkFrm, 'pTrkiTgt':trkiTgt, 'pTrkiPt':trkiPt },appendmat=False)
