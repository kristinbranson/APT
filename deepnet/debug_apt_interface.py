



import APT_interface as apt
cmd = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl -name apt_expt -cache /nrs/branson/mayank/apt_cache -conf_params batch_size 8  dlc_augment False  decay_steps 20000  save_step 5000  rrange 10  dl_steps 60000  trange 5  mdn_use_unet_loss True -train_name dlc_noaug train -skip_db -use_cache'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
apt.main(cmd.split())

##
import cv2
cap = cv2.VideoCapture('/nrs/branson/longterm/files_for_working_with_apt/20160214T111910_1_hour_segment_02.mjpg')


##

import PoseTools
import os
import glob
import APT_interface as apt
import apt_expts
reload(apt_expts)
import PoseUNet_resnet
reload(PoseUNet_resnet)
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

db_file = '/nrs/branson/mayank/apt_cache/sh_trn4523_gt080618_made20180627_cacheddata/mdn/view_1/sh_compare/val_TF.tfrecords'
bsz = 4
lr_mul = 1
name = 'bsz_{}_lr_{}'.format(bsz,int(lr_mul*10))
cdir = os.path.dirname(db_file)
tfile = os.path.join(cdir,'sh_trn4523_gt080618_made20180627_cacheddata_{}_traindata'.format(name))

A = PoseTools.pickle_load(tfile)
conf = A[1]

files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*.index").format(name))
files.sort(key=os.path.getmtime)
aa = [int(re.search('-(\d*).index',f).groups(0)[0]) for f in files]
aa = [b-a for a,b in zip(aa[:-1],aa[1:])]
if any([a<0 for a in aa]):
    bb = int(np.where(np.array(aa)<0)[0])+1
    files = files[bb:]
files = [f.replace('.index','') for f in files]
files = files[-1:]

mdn_out = apt_expts.classify_db_all(conf,db_file,files,'mdn',name=name)


##
cmd = '-name 20190129T180959 -view 1 -cache /home/mayank/temp/apt_cache -err_file /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T180959/trk/movie_trn20190129T180959_iter20000_20190208T141629.err -model_files /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T180959/deepnet-20000 -type mdn /home/mayank/temp/apt_cache/multitarget_bubble/20190129T180959_20190129T181147.lbl track -mov /home/mayank/work/FlySpaceTime/cx_GMR_SS00038_CsChr_RigB_20150729T150617/movie.ufmf -out /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T180959/trk/movie_trn20190129T180959_iter20000_20190208T141629.trk -start_frame 8496 -end_frame 8696 -trx /home/mayank/work/FlySpaceTime/cx_GMR_SS00038_CsChr_RigB_20150729T150617/registered_trx.mat -trx_ids 3'
##
# debug postprocessing
import APT_interface as apt
import numpy as np

lbl_file = '/home/mayank/temp/apt_cache/multitarget_bubble/20190207T121622_20190207T121731.lbl'
conf = apt.create_conf(lbl_file,0,'20190207T121622','/home/mayank/temp/apt_cache','mdn')

import multiResData
A = multiResData.read_and_decode_without_session('/home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190207T121622/train_TF.tfrecords',conf,())
ims = np.array(A[0])
locs = np.array(A[1])
import PoseTools
reload(PoseTools)
a,b = PoseTools.randomly_affine(ims[:10,...],locs[:10,...],conf)

##
# debug postprocessing
import APT_interface as apt
import RNN_postprocess

lbl_file = '/home/mayank/temp/apt_cache/multitarget_bubble/20190207T121622_20190207T121731.lbl'
conf = apt.create_conf(lbl_file,0,'20190207T121622','/home/mayank/temp/apt_cache','mdn')
self = RNN_postprocess.RNN_pp(conf,'deepnet',
                              name = 'rnn_pp',
                              data_name='rnn_pp_groups_augfix')
self.rnn_pp_hist = 8
self.train_rep = 3
self.create_db(split_file = '/home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T153403/splitdata.json')


##
import APT_interface as apt
apt.main(cmd.split())

##
lbl_file  = '/home/mayank/temp/apt_cache/multitarget_bubble/20190131T181525_20190131T181623.lbl'
import APT_interface as apt
import os
import tensorflow as tf
import multiResData
apt.test_preproc(lbl_file)
##
lbl_file  = '/home/mayank/temp/apt_cache/multitarget_bubble/20190129T180959_20190129T181147.lbl'
import APT_interface as apt
import os
import tensorflow as tf
import multiResData
conf = apt.create_conf(lbl_file,0,'compare_cache','/home/mayank/temp/apt_cache','mdn')

conf.trainfilename = 'normal.tfrecords'
n_envs = multiResData.create_envs(conf,False)
conf.trainfilename = 'cached.tfrecords'
c_envs = multiResData.create_envs(conf,False)

n_out_fns = [lambda data: n_envs[0].write(apt.tf_serialize(data)),
           lambda data: n_envs[1].write(apt.tf_serialize(data))]
c_out_fns = [lambda data: c_envs[0].write(apt.tf_serialize(data)),
           lambda data: c_envs[1].write(apt.tf_serialize(data))]

splits = apt.db_from_cached_lbl(conf, c_out_fns, False, None, False)
splits = apt.db_from_lbl(conf, n_out_fns, False, None, False)
c_envs[0].close()
n_envs[0].close()

c_file_name = os.path.join(conf.cachedir,'cached.tfrecords')
n_file_name = os.path.join(conf.cachedir,'normal.tfrecords')
A = []
A.append(multiResData.read_and_decode_without_session(c_file_name,conf,()))
A.append(multiResData.read_and_decode_without_session(n_file_name,conf,()))

ims1= np.array(A[0][0]).astype('float')
ims2 = np.array(A[1][0]).astype('float')
locs1 = np.array(A[0][1])
locs2 = np.array(A[1][1])

ndx = np.random.choice(ims1.shape[0])
f,ax = plt.subplots(1,2,sharex=True,sharey=True)
ax = ax.flatten()
ax[0].imshow(ims1[ndx,:,:,0],'gray',vmin=0,vmax=255)
ax[1].imshow(ims2[ndx,:,:,0],'gray',vmin=0,vmax=255)
ax[0].scatter(locs1[ndx,:,0],locs1[ndx,:,1])
ax[1].scatter(locs2[ndx,:,0],locs2[ndx,:,1])


##
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cmd = '-name 20190129T144258 -view 1 -cache /home/mayank/temp/apt_cache -err_file /home/mayank/temp/apt_cache/multitarget_bubble/20190129T144258_20190129T144311.err -type mdn /home/mayank/temp/apt_cache/multitarget_bubble/20190129T144258_20190129T144311.lbl train -use_cache'
apt.main(cmd.split())

##

import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cmd = '-name 20190114T160046 -view 1 -cache /home/mayank/temp/apt_cache -err_file /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190114T160046/trk/movie_trn20190114T160046_20190114T181805.err -type mdn /home/mayank/temp/apt_cache/multitarget_bubble/20190114T160046_20190114T160137.lbl track -mov /home/mayank/work/FlySpaceTime/cx_GMR_SS00038_CsChr_RigB_20150729T150617/movie.ufmf -out /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190114T160046/trk/movie_trn20190114T160046_20190114T181805.trk -start_frame 50 -end_frame 220 -trx /home/mayank/work/FlySpaceTime/cx_GMR_SS00038_CsChr_RigB_20150729T150617/registered_trx.mat -trx_ids 3'

apt.main(cmd.split())

##
import APT_interface as apt
reload(apt)
import multiResData
import h5py
# lbl_file = '/home/mayank/temp/test_conversion/20190114T111122_20190114T111124.lbl'
# db_file = '/home/mayank/temp/test_conversion/mdn/view_0/20190114T111122/train_TF.tfrecords'
lbl_file = '/home/mayank/temp/apt_cache/multitarget_bubble/20190114T151632_20190114T151735.lbl'
db_file = '/home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190114T151632/train_TF.tfrecords'
L = h5py.File(lbl_file,'r')
conf = apt.create_conf(lbl_file,0,'test','/home/mayank/temp','mdn')
A = multiResData.read_and_decode_without_session(db_file,conf,())
orig_locs = []
labeled_locs = []
for ndx in range(0,len(A[0]),500):
    cur_locs = A[1][ndx]
    info = A[2][ndx]
    trx_file = apt.read_string(L[L['trxFilesAll'][0,info[0]]])
    cur_trx,_ = apt.get_cur_trx(trx_file,info[2])
    cur_orig = apt.convert_to_orig(cur_locs,conf,info[1],cur_trx,None)
    orig_locs.append(cur_orig)
    pts = apt.trx_pts(L,info[0])
    labeled_locs.append(pts[info[2],info[1],:,:].T)

orig_locs = np.array(orig_locs)
labeled_locs = np.array(labeled_locs)


##
##
import APT_interface as apt

lbl_file = '/home/mayank/temp/apt_cache/multitarget_bubble/20190111T185319_20190111T185419.lbl'
conf = apt.create_conf(lbl_file,0,'test','/home/mayank/temp/apt_cache1','mdn')
apt.create_tfrecord(conf,True)

##
import numpy as np
import multiResData
reload(multiResData)
import easydict
conf = easydict.EasyDict()
isz = 6 + np.random.choice(2)
conf.imsz = [isz,isz]
conf.img_dim = 1
ims = np.zeros([18,18,1])
st = 6
en = 9 + np.random.choice(2)
ims[st:en,st:en,:] = 1
locs = np.array([[st,st,en-1,en-1,7],[st,en-1,st,en-1,7]])
locs = locs.T
angle = np.random.choice(180) * np.pi / 180
ni,nl = multiResData.crop_patch_trx(conf,ims,7,7,angle,locs)
f,ax = plt.subplots(1,2)
ax[0].imshow(ims[:,:,0])
ax[0].scatter(locs[:,0],locs[:,1])
ax[1].imshow(ni[:,:,0])
ax[1].scatter(nl[:,0],nl[:,1])
ax[1].set_title('{},{}'.format(isz%2,en%2))


##
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cmd_str = '-cache /home/mayank/temp -name xv_test -type mdn /home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl classify -out_file /home/mayank/temp/aa'
cc = cmd_str.split()
apt.main(cc)

##
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cmd_str = '-name alice_compare -cache /home/mayank/work/APT/deepnet/cache -type mdn /home/mayank/work/APT/deepnet/data/multitarget_bubble_expandedbehavior_20180425_modified4.lbl track -start_frame 5000 -end_frame 5500 -trx /home/mayank/work/FlyBowl/pBDPGAL4U_TrpA_Rig2Plate14BowlD_20110617T143743/registered_trx.mat -mov /home/mayank/work/FlyBowl/pBDPGAL4U_TrpA_Rig2Plate14BowlD_20110617T143743/movie.ufmf -trx_ids 3 8 -out /home/mayank/temp/a.trk'

cc = cmd_str.split()
apt.main(cc)

##
import APT_interface as apt
cmd_str = '-name stephen_20181029 -conf_params mdn_groups ((0),(1,2,3,4)) -cache /tmp -type mdn /home/mayank/work/APT/deepnet/data/sh_trn4879_gtcomplete.lbl train -use_cache -skip_db'

cc = cmd_str.split()
apt.main(cc)
## stephen without image mean normalization
import APT_interface as apt
import os
import PoseUNet_resnet as PoseUNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
lbl_file = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl'
view = 1
conf = apt.create_conf(lbl_file, view, 'conf','/tmp',net_type='umdn')
conf.cachedir = '/nrs/branson/mayank/apt_cache/stephen_view{}'.format(view)
conf.normalize_img_mean = False
self = PoseUNet.PoseUMDN_resnet(conf,name='no_mean_norm')
self.train_umdn()
V = self.classify_val()

##

import APT_interface as apt
import os
import PoseUNet_resnet as PoseUNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
lbl_file = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl'
view = 1
conf = apt.create_conf(lbl_file, view, 'conf','/tmp',net_type='umdn')
conf.cachedir = '/nrs/branson/mayank/apt_cache/stephen_view{}'.format(view)
conf.normalize_img_mean = False
self = PoseUNet.PoseUMDN_resnet(conf,'no_mean_norm')
V = self.classify_val()
res = np.array([
    [8.15356254,  7.79341274,  8.01287003,  8.61840345,  8.13417424],
    [ 9.65344996,  9.5212058 ,  9.99045115, 10.12551694,  9.58502402],
   [11.86516147, 12.44826803, 12.82939408, 12.48889447, 12.14856348],
    [13.38951142, 15.10225055, 14.2305725 , 15.00483587, 14.43672831]])


## Incorrect img normalize code
import APT_interface as apt
import os
import PoseUNet_resnet as PoseUNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
lbl_file = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl'
view = 1
conf = apt.create_conf(lbl_file, view, 'conf','/tmp',net_type='umdn')
conf.cachedir = '/nrs/branson/mayank/apt_cache/stephen_view{}'.format(view)
self = PoseUNet.PoseUMDN_resnet(conf)
V = self.classify_val()
res = np.array([
    [ 7.24539496,  7.8049516 ,  7.97217146,  8.44032115,  7.69838612],
    [ 8.76899118,  9.63168685,  9.55280912, 10.26671805,  9.54993247],
    [10.91299409, 12.02790342, 11.79002365, 13.02997551, 11.82780871],
    [12.50440241, 15.64563049, 13.13194025, 14.71218933, 13.96273946]])

unet_pred = V[6][-1]
mdn_pred = V[3]
locs = V[4]
xx = V[5][3]*self.offset
ii = np.argmax(V[5][2],axis=1)
mdn_conf = np.zeros([xx.shape[0],5])
for ndx in range(V[5][0].shape[0]):
    for pt in range(conf.n_classes):
        mdn_conf[ndx,pt] = xx[ndx,ii[ndx,0],pt]

dd = np.sqrt(np.sum((locs-mdn_pred)**2,axis=-1))
dd_unet = np.sqrt(np.sum((locs-unet_pred)**2,axis=-1))
dd_unet_mdn = np.sqrt(np.sum((mdn_pred-unet_pred)**2,axis=-1))
unet_conf = np.max(V[6][0],axis=(1,2))
pos = dd > self.min_dist;#self.min_dist
pt = 1
from sklearn.metrics import roc_curve, auc
fpr_mdn,tnr_mdn,_ = roc_curve(pos[:,pt],mdn_conf[:,pt])
fpr_unet, tnr_unet,_ = roc_curve(pos[:,pt], dd_unet_mdn[:,pt])
fpr_unetc, tnr_unetc,_ = roc_curve(pos[:,pt], -unet_conf[:,pt])
# dd_comb = np.maximum(dd_unet_mdn,mdn_conf)
dd_comb = dd_unet_mdn+mdn_conf

fpr_comb, tnr_comb,_ = roc_curve(pos[:,pt], dd_comb[:,pt])

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(dd[:,pt],mdn_conf[:,pt])
plt.figure()
plt.scatter(dd[:,pt],dd_unet_mdn[:,pt])

plt.figure()
plt.plot(fpr_mdn,tnr_mdn)
plt.plot(fpr_unet,tnr_unet)
plt.plot(fpr_unetc,tnr_unetc)
plt.plot(fpr_comb,tnr_comb)
plt.legend(['mdn','unet','unetc','comb'])

##
from matplotlib import pyplot as plt
tr = 12
kk = np.where(np.any(V[0]>tr,axis=1))[0]
pt = np.random.choice(kk)
jj = np.where(V[0][pt,:]>tr)[0][0]
plt.imshow(V[2][pt,:,:,jj])
plt.scatter(V[4][pt,jj,0],V[4][pt,jj,1])
plt.figure()
plt.imshow(V[1][pt,:,:,0],'gray')
plt.scatter(V[4][pt,jj,0],V[4][pt,jj,1])
##
##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval';
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_dataset as PoseUNet
conf.normalize_img_mean = True
self = PoseUNet.PoseUNet(conf,'mean_img',pad_input=False)
self.no_pad = False
self.train_unet()
V = self.classify_val()


##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval';
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_dataset as PoseUNet
self = PoseUNet.PoseUNet(conf,'normal',pad_input=False)
self.no_pad = False
V = self.classify_val()

res = np.array([[
         1.33725852,  1.44260777,  1.2197094 ,  1.4054476 ,  1.35461989,
         1.56339064,  1.26828579,  1.67202601,  1.64307798,  1.58602942,
         1.53797884,  2.02196571,  1.69124296,  3.61847242,  3.67470543,
         1.33736349,  1.59257857],
       [ 1.50369884,  1.63343346,  1.37017441,  1.56989693,  1.52073472,
         1.81230858,  1.43857005,  1.91531989,  1.96209875,  1.80547304,
         1.86310851,  2.93751308,  2.18633115,  6.91663258,  7.43064016,
         1.7052951 ,  2.31254946],
       [ 1.66861117,  1.87583231,  1.5277497 ,  1.77298601,  1.7090494 ,
         2.1332188 ,  1.66154602,  2.33803084,  2.43522788,  2.09856196,
         2.37024879,  5.07045206,  4.64947697, 12.4103475 , 12.49274121,
         4.09860036,  3.98436972],
       [ 1.83226236,  1.99950524,  1.62278993,  1.91806128,  1.83863565,
         2.36829227,  1.83341173,  2.70041566,  3.03500595,  2.40945369,
         2.76952887,  7.3060946 ,  8.71812219, 17.10891411, 15.22070564,
         8.70113416,  6.7436511 ]])

##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval';
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_dataset as PoseUNet
self = PoseUNet.PoseUNet(conf,'pad_input',pad_input=True)
self.no_pad = True
V = self.classify_val()
res = np.array([[ 
        0.95649637,  1.02741287,  1.04891375,  1.07227733,  1.02023282,
         1.27983229,  1.09385869,  1.31877817,  1.3403253 ,  1.30063015,
         1.41057154,  1.72905072,  1.45960683,  3.40282266,  3.17321152,
         1.40711469,  1.53104846],
       [ 1.09724486,  1.16849824,  1.22082977,  1.24287679,  1.19170516,
         1.53320109,  1.26721911,  1.54900245,  1.69752963,  1.54619462,
         1.75219689,  2.72170986,  1.98324972,  7.71005632,  6.90737272,
         1.82825058,  2.44256888],
       [ 1.31465742,  1.35346597,  1.49190471,  1.49177832,  1.40930846,
         1.85047257,  1.56990009,  1.97619232,  2.48437686,  1.94906626,
         2.35587152,  5.23579364,  5.49393322, 14.4646526 , 12.48910452,
         5.18257147,  4.67947102],
       [ 1.49047977,  1.58661416,  1.74333241,  1.74709639,  1.71730347,
         2.18539484,  1.87386084,  2.8115908 ,  3.51928015,  2.47797139,
         3.02428713, 10.29741283, 12.1697288 , 20.66926424, 16.88128325,
        11.85019535,  8.162599  ]])

##
from poseConfig import aliceConfig as conf
conf.cachedir += '_bigsize'; conf.imsz = (370,370)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_dataset as PoseUNet
p_sz, a_sz = PoseUNet.find_pad_sz(4,conf.imsz[0])
print a_sz
self = PoseUNet.PoseUNet(conf,'unet_no_pad',pad_input=False)
self.no_pad = True
V = self.classify_val()
res = np.array([[ 
        0.98649726,  1.02755724,  1.05933487,  1.0318891 ,  1.05936735,
         1.32980098,  1.15310487,  1.30030844,  1.34996524,  1.35013757,
         1.43107478,  1.78587489,  1.51182891,  3.71149039,  3.65348891,
         1.33653132,  1.55634646],
       [ 1.16142442,  1.18833696,  1.24923555,  1.19032531,  1.22991107,
         1.55732873,  1.35345994,  1.59213747,  1.72097007,  1.60372583,
         1.78842193,  2.8360998 ,  2.22584712,  7.71927776,  7.67398873,
         1.80188066,  2.58162556],
       [ 1.411857  ,  1.41011399,  1.4909414 ,  1.50987044,  1.46764162,
         1.8779709 ,  1.70232433,  2.16371472,  2.8266522 ,  2.03626738,
         2.43268328,  6.47444709,  7.745492  , 16.25207438, 14.27371572,
         6.12199992,  5.13091404],
       [ 1.88424917,  1.69646305,  1.75829603,  1.79497454,  1.89181109,
         2.29069112,  2.58236851,  4.82837597,  5.54433209,  3.4245605 ,
         3.63215097, 16.06407953, 18.46977448, 29.2672358 , 27.22291975,
        13.97790465,  8.60922717]])

##
from poseConfig import aliceConfig as conf
conf.cachedir += '_bigsize'; conf.imsz = (370,370)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_dataset as PoseUNet
p_sz, a_sz = PoseUNet.find_pad_sz(4,conf.imsz[0])
print a_sz
self = PoseUNet.PoseUNet(conf,'unet_normal',pad_input=False)
self.no_pad = False
V = self.classify_val()
res = np.array([[ 
         1.01337715,  1.05705156,  1.07776031,  1.09172904,  1.06536099,
         1.48785068,  1.27296158,  1.33163499,  1.6128113 ,  1.46002975,
         1.44665614,  1.8028599 ,  1.53172588,  3.88912751,  4.49638049,
         1.40090689,  1.66136731],
       [ 1.17340684,  1.21460893,  1.24967976,  1.27470344,  1.25700266,
         1.78026737,  1.51757002,  1.58810376,  2.08679414,  1.71911213,
         1.82005424,  2.75833757,  2.20933533,  7.74779025,  8.49043286,
         1.88379963,  2.56583515],
       [ 1.41253535,  1.43810346,  1.48506689,  1.53555795,  1.54355258,
         2.14507233,  1.83928879,  2.03955129,  2.92512937,  2.1207873 ,
         2.46551749,  5.7303777 ,  6.61282364, 14.16446044, 15.11038458,
         5.97520839,  4.9157388 ],
       [ 1.9945976 ,  1.81462598,  1.84469387,  1.87837574,  1.82111803,
         2.62341737,  2.39273338,  2.84012614,  4.22403285,  2.76444815,
         3.63215097, 13.98818852, 14.41327968, 26.60785536, 22.10706024,
        13.97790465, 10.49898462]])


##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_dataset as PoseUNet
p_sz, a_sz = PoseUNet.find_pad_sz(4,conf.imsz[0])
print a_sz
self = PoseUNet.PoseUNet(conf,'test_pad',pad_input=False)
self.no_pad = True
self.train_unet()
V = self.classify_val()

##
import APT_interface as apt
cmd_str = '-name stephen_20181029 -cache /groups/branson/home/kabram/bransonlab/stephen_copy/apt_cache/ /groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl train -use_cache'
cmd_str = '-view 1 -type mdn -name stephen_20181101 -cache /groups/branson/home/kabram/bransonlab/stephen_copy/apt_cache/ /groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl train -use_cache -skip_db'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
apt.main(cmd_str.split())

##
cmd_str = '-name alice_unet_dataset -cache /nrs/branson/mayank/temp/apt_cache /nrs/branson/mayank/apt_cache/alice_model_20181011/multitarget_bubble_expandedbehavior_20180425_modified3.lbl track -mov {0}/movie.ufmf -trx {0}/registered_trx.mat -out /tmp/movie_unet_20181011.trk -trx_ids 9 -start_frame 7539 -end_frame 7550'
mov = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00038_CsChr_RigB_20150729T150617'
args = cmd_str.format(mov).split()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import APT_interface as apt
apt.main(args)

##
import trackStephenHead_KB as ts
cmd_str = '-s /groups/branson/bransonlab/mayank/stephen_copy/fly1302/View1Vids.txt -f /groups/branson/bransonlab/mayank/stephen_copy/fly1302/View2Vids.txt -d /groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv -o /nrs/branson/mayank/temp/stephenOut'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ts.main(cmd_str.split())
##
import APT_interface as apt
cmd_str = '-name stephen_20181029 -cache /groups/branson/home/kabram/temp/delete -out_dir /groups/branson/home/kabram/temp -view 1 /groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl track -mov /groups/huston/hustonlab/flp-chrimson_experiments/fly_219_to_228_28_10_15_SS00325_x_norpAcsChrimsonFlp11/fly219/fly219_trial9/C001H001S0001/C001H001S0001_c.avi -out /groups/branson/home/kabram/temp/sh_test.trk -crop_loc 1 230 1 350       '
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
apt.main(cmd_str.split())

##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'test_pad')#,no_pad=True)
self.pred_dist = True
conf.use_unet_loss = True
conf.pretrained_weights = '/groups/branson/bransonlab/mayank/PoseTF/data/pretrained/resnet_tf_v2/20180601_resnet_v2_imagenet_checkpoint/model.ckpt-258931'
self.train_umdn()
V = self.classify_val()


##
from poseConfig import aliceConfig as conf
conf.cachedir += '_bigsize'
conf.imsz = (370,370)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'test_bigsize',no_pad=True)
self.pred_dist = False
conf.use_unet_loss = False
conf.pretrained_weights = '/groups/branson/bransonlab/mayank/PoseTF/data/pretrained/resnet_tf_v2/20180601_resnet_v2_imagenet_checkpoint/model.ckpt-258931'
self.train_umdn()
V = self.classify_val()

res = np.array([[ 1.18327626,  1.15724428,  1.20505805,  1.08790263,  1.11557909,
         1.45859782,  1.62034894,  1.39129368,  1.85965736,  1.43013668,
         1.79865341,  2.96021304,  2.41855288,  4.71243198,  5.26124405,
         2.19385436,  2.7959588 ],
       [ 1.3767262 ,  1.33358876,  1.39085419,  1.25266083,  1.28215576,
         1.68902865,  1.91232526,  1.65692984,  2.26331428,  1.69121696,
         2.14496275,  4.10558733,  3.6119709 ,  6.88167053,  7.48156162,
         3.0936714 ,  3.97832132],
       [ 1.60399063,  1.53403108,  1.63004902,  1.47647513,  1.4889976 ,
         1.99646117,  2.25486105,  2.0183618 ,  3.03954479,  2.04269245,
         2.737381  ,  5.99884427,  5.85614553,  9.93859059, 10.32393578,
         5.30966831,  5.9665582 ],
       [ 1.81175005,  1.65701498,  1.84175742,  1.65496362,  1.61401973,
         2.31068424,  2.49783233,  2.38824771,  3.74834668,  2.38614989,
         3.33242332,  7.70357524,  7.72372822, 11.74492753, 11.98443179,
         7.62420938,  8.10080463]])


##
# with image normalization
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
conf.normalize_img_mean = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'img_mean')
self.pred_dist = True
self.no_pad = False
conf.use_unet_loss = True
conf.pretrained_weights = '/groups/branson/bransonlab/mayank/PoseTF/data/pretrained/resnet_tf_v2/20180601_resnet_v2_imagenet_checkpoint/model.ckpt-258931'
self.train_umdn()
V = self.classify_val()

res = np.array([[
         1.58521412,  1.48815935,  1.45851919,  1.44834792,  1.37390946,
         1.8788664 ,  3.73171258,  1.72103746,  2.41810415,  1.80480459,
         2.28317475,  3.72904744,  3.44024931,  7.74428839,  6.04168117,
         2.76626114,  3.43857159],
       [ 1.85102995,  1.70563589,  1.68911544,  1.65062221,  1.57813998,
         2.20772525,  4.16370384,  2.03675058,  3.02349929,  2.09745323,
         2.78786808,  4.95289499,  4.56364654, 10.09315267,  8.41309695,
         3.91772582,  4.66935327],
       [ 2.2363397 ,  1.9506147 ,  1.96752048,  1.92388896,  1.82183436,
         2.50450121,  4.77445928,  2.57665294,  4.00480706,  2.51823133,
         3.62576981,  7.0623059 ,  6.90236872, 13.7972485 , 11.39102586,
         6.54784416,  6.92665553],
       [ 2.61349925,  2.21672822,  2.35895399,  2.09854907,  2.02583241,
         2.71236362,  5.06641738,  3.0526406 ,  4.73309124,  2.90487022,
         4.25193007,  8.89275189,  9.67358293, 17.10405132, 12.92693627,
         8.30638507,  9.13116725]])
##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
conf.use_unet_loss = True
conf.pretrained_weights = '/groups/branson/bransonlab/mayank/PoseTF/data/pretrained/resnet_tf_v2/20180601_resnet_v2_imagenet_checkpoint/model.ckpt-258931'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'joint_deconv_wt_decay')
self.pred_dist = True
self.train_umdn()

##
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
conf = apt.create_conf('/nrs/branson/mayank/apt_cache/alice_model_20181011/multitarget_bubble_expandedbehavior_20180425_modified3.lbl',0,'temp',cache_dir='/nrs/branson/mayank/apt_cache/alice_model_20181011')
db_file = '/groups/branson/bransonlab/mayank/apt_expts/alice/withinAssay_mnc_view0.tfrecords'
V = apt.classify_db_all('unet',conf,db_file)
dd = np.sqrt(np.sum((V[0]-V[1])**2,axis=-1))
np.percentile(dd,[90,95,98],axis=0)
import multiResData
A = multiResData.read_and_decode_without_session(db_file,conf,())
ex = 25
plt.imshow(A[0][ex,:,:,0],'gray')
plt.scatter(A[1][ex,:,0],A[1][ex,:,1])
plt.scatter(V[0][ex,:,0],V[0][ex,:,1])

ex = 25
plt.imshow(A[0][ex][:,:,0],'gray')
plt.scatter(A[1][ex][:,0],A[1][ex][:,1])
plt.scatter(V[0][ex,:,0],V[0][ex,:,1])

##
cmd_str = '-model /groups/branson/bransonlab/apt/tmp/postproc/cache/sh_trn4523_gt080618_made20180627_cacheddata_view0/20180924T171526/sh_trn4523_gt080618_made20180627_cacheddata_view0_pose_unet-2000 -view 1 /groups/branson/bransonlab/apt/tmp/postproc/cache/20180924T200537.lbl track -mov /groups/huston/hustonlab/flp-chrimson_experiments/fly_640_to_645_SS02323_24_5_17/fly641_intensity_4/C001H001S0004/C001H001S0004_.avi -out /tmp/sh_trk.trk -start_frame 200 -hmaps -crop_loc 33 262 35 384'
args = cmd_str.split()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import APT_interface as apt
apt.main(args)
##

##
from poseConfig import aliceConfig as conf

conf.cachedir += '_moreeval'
import RNN_postprocess

self = RNN_postprocess.RNN_pp(conf, 'joint_deconv',
                              name='test',
                              data_name='pp_data_joint_deconv')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
self.conf.display_step = 500
self.conf.save_step = 20000
self.conf.save_td_step = 500
self.train()
V = self.classify_val()

##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
conf.use_unet_loss = True
conf.pretrained_weights = '/groups/branson/bransonlab/mayank/PoseTF/data/pretrained/resnet_tf_v2/20180601_resnet_v2_imagenet_checkpoint/model.ckpt-258931'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'joint_deconv_dist_pred')
self.pred_dist = True
self.train_umdn()

res = np.array([[
         1.54084525,  1.62379318,  1.3783159 ,  1.49279011,  1.39449199,
         1.72221675,  1.89682875,  1.62341823,  2.04673058,  1.65487335,
         1.94486546,  3.04943163,  2.43700348,  4.73240183,  5.35499167,
         2.35998346,  2.73526744],
       [ 1.74630131,  1.85429174,  1.54907082,  1.70381653,  1.57535291,
         1.98919827,  2.13145079,  1.87813749,  2.4231829 ,  1.87284427,
         2.33709487,  4.14463842,  3.43682919,  6.51247556,  7.50756899,
         3.06960952,  3.64118686],
       [ 1.96937294,  2.06981242,  1.81262413,  1.949028  ,  1.7854574 ,
         2.29420957,  2.42605065,  2.17318074,  3.11284902,  2.22724712,
         2.8449839 ,  5.57011565,  5.70503642,  9.67688295, 10.0948947 ,
         4.76154918,  5.33240949],
       [ 2.20182296,  2.25321972,  1.96811797,  2.11017522,  1.94010261,
         2.52629141,  2.62292896,  2.44078065,  3.97345295,  2.5294883 ,
         3.19292776,  7.21698654,  7.89742484, 11.80681899, 12.21654218,
         7.14757342,  7.12179407]])

## check loc transformations.

import h5py
import APT_interface as apt
import multiResData
from poseConfig import aliceConfig as conf
import movies
from scipy import io as sio

on_gt = False

lbl = h5py.File(conf.labelfile, 'r')
local_dirs, _ = multiResData.find_local_dirs(conf, False)
ndx = 0
dir_name = local_dirs[ndx]
cap = movies.Movie(dir_name)
trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)
trx = sio.loadmat(trx_files[ndx])['trx'][0]
n_trx = len(trx)
trx_ndx = 0
frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx, False)

sel_pts = conf.selpts
cur_trx = trx[trx_ndx]
fnum = frames[0]
cur_pts = multiResData.trx_pts(lbl, ndx, on_gt)
frame_in, cur_loc = multiResData.get_patch(
    cap, fnum, conf, cur_pts[trx_ndx, fnum, :, sel_pts], cur_trx=cur_trx, crop_loc=None)

T, first_frames, end_frames, n_trx = apt.get_trx_info(trx_files[ndx], conf, 100000)
trx_fnum_start = fnum - first_frames[trx_ndx]
t_loc = apt.convert_to_orig(cur_loc[np.newaxis,...], conf, cur_trx, trx_fnum_start, None, None,1, None)
dd = np.sqrt(np.sum((t_loc[0,:,:]-cur_pts[trx_ndx,fnum,:,sel_pts])**2,axis=-1))
print dd
##

from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import RNN_postprocess
self = RNN_postprocess.RNN_pp(conf,'joint_deconv',
                              name = 'rnn_pp_conv_no_skip',
                              data_name='pp_data_joint_deconv')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# self.net_type = 'conv'
self.train()
V = self.classify_val()
##


cmd = '-cache /nrs/branson/mayank/apt_cache/alice_model_20181002 /groups/branson/home/kabram/temp/multitarget_bubble_expandedbehavior_20180425_modified1.lbl track -mov /groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343/movie.ufmf -trx /groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343/registered_trx.mat -out /nrs/branson/mayank/apt_cache/alice_model_20181002/temp.trk  -start_frame 13230 -end_frame 13300'
args = cmd.split()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import APT_interface as apt
apt.main(args)

##
import tensorflow as tf
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
conf.use_unet_loss = True
conf.batch_size = 32
conf.labelfile = '/groups/branson/bransonlab/mayank/PoseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'
import RNN_postprocess
self = RNN_postprocess.RNN_pp(conf,'unet_resnet_official',
                              name = 'rnn_pp_transformer',
                              data_name='test')
import os
self.train_rep = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
self.locs_coords = 'example'
self.debug = True
self.create_db('/groups/branson/bransonlab/mayank/PoseTF/cache/alice_moreeval/splitdata.json')

##

from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import RNN_postprocess
self = RNN_postprocess.RNN_pp(conf,'unet_resnet_official',
                              name = 'rnn_pp_conv_skip',
                              data_name='rnn_pp_bidir')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# self.net_type = 'conv'
self.train()
V = self.classify_val()

pp = V[1].reshape([-1,17,2])
ll = V[2].reshape([-1,17,2])
dd = np.sqrt(np.sum((pp-ll)**2,axis=-1))
ppo = V[3][:,:34].reshape([-1,17,2])
ddo = np.sqrt(np.sum((ppo-ll)**2,axis=-1))
xxo = np.percentile(ddo,[90,95,98,99],axis=0)*max(conf.imsz)
xx = np.percentile(dd,[90,95,98,99],axis=0)*max(conf.imsz)

#xx
res = np.array([[
         1.80364551,  1.69575246,  1.67426285,  1.76577435,  1.579601  ,
         1.84871412,  3.13844453,  2.08043529,  2.3782408 ,  1.92955603,
         2.43570741,  3.79659706,  4.06354492,  5.73352012,  6.00309658,
         3.2418871 ,  3.26953066],
       [ 2.11390513,  1.99838733,  1.976268  ,  2.05494483,  1.90984056,
         2.22680451,  3.53576434,  2.40976311,  2.9644746 ,  2.35198678,
         2.95095041,  5.07868002,  5.25308616,  7.67363181,  8.4102668 ,
         4.23697015,  4.30647103],
       [ 2.61365095,  2.45946475,  2.35853781,  2.4234413 ,  2.30256417,
         2.66892549,  4.16666702,  2.89393716,  3.92726722,  2.81578129,
         3.74144504,  7.24095993,  7.34431106, 10.08861209, 11.45328673,
         6.60340657,  6.22612457],
       [ 3.09053002,  2.81819323,  2.76718237,  2.70491357,  2.63553676,
         3.10034395,  4.42760013,  3.23359988,  4.78180579,  3.18823969,
         4.44198214,  8.97574577, 10.00289767, 12.68000074, 13.18696339,
         8.79415605,  7.90214257]])

#xxo
res_old = np.array([[
         1.49136081,  1.56254737,  1.34695339,  1.39857119,  1.2528565 ,
         1.65538806,  3.43192526,  1.62278219,  2.28333682,  1.60846648,
         2.27924607,  3.76117975,  3.85477328,  5.93197075,  5.8990881 ,
         2.850875  ,  3.31280479],
       [ 1.78629012,  1.7667829 ,  1.54653622,  1.56400294,  1.42882386,
         1.88952688,  3.83356381,  1.97244375,  2.98738912,  1.91013101,
         2.73122715,  5.04301531,  5.31638032,  8.1395703 ,  8.59475827,
         4.020976  ,  4.46488173],
       [ 2.17408585,  1.98854999,  1.85151215,  1.79422603,  1.6481121 ,
         2.25822602,  4.29838114,  2.42178381,  4.03215446,  2.39262505,
         3.6287133 ,  7.67961704,  7.54905661, 10.77928395, 11.78846035,
         6.42459216,  6.62846603],
       [ 2.6234016 ,  2.18333878,  2.13563475,  1.98056923,  1.83855431,
         2.56913315,  4.67488301,  2.71745852,  4.8243573 ,  2.7104543 ,
         4.34407322,  9.76701581, 10.34117991, 14.0200528 , 13.67878039,
         8.9695074 ,  7.77906415]])

##


from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
conf.use_unet_loss = True
conf.batch_size = 32
conf.labelfile = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'
import RNN_postprocess
self = RNN_postprocess.RNN_pp(conf,'unet_resnet_official')
self.create_db('/home/mayank/work/poseTF/cache/alice_moreeval/splitdata.json')

##
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import PoseUNet_resnet
conf.pretrained_weights = '/home/mayank/Downloads/20180601_resnet_v2_imagenet_checkpoint/model.ckpt-258931'
conf.use_unet_loss = True
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'unet_resnet_official')
self.train_umdn()
V = self.classify_val()
np.percentile(V[0],[90,95,98,99],axis=0)


##
from poseConfig import aliceConfig as conf
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUNet_resnet(conf)
self.train_unet()
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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import apt_expts
db_file = '/home/mayank/work/poseTF/cache/alice_moreeval/val_TF.tfrecords'
model_file = '/home/mayank/work/poseTF/cache/alice_moreeval/deepcut/snapshot-100000'
# db_file = '/home/mayank/work/poseTF/cache/alice_dataset/val_TF.tfrecords'
# model_file = '/home/mayank/work/poseTF/cache/alice/deeplabcut/nopretrained/snapshot-1030000'
import multiResData
tf_iterator = multiResData.tf_reader(conf, db_file, False)
tf_iterator.batch_size = 1
read_fn = tf_iterator.next
curm = 'deeplabcut'
m =model_file
import APT_interface as apt
conf.dlc_rescale = 1
pred_fn, close_fn, _ = apt.get_pred_fn(curm, conf, m)
pred, label, gt_list, ims = apt.classify_db(conf, read_fn, pred_fn, tf_iterator.N, return_ims=True)
dd = np.sqrt(np.sum((pred-label)**2,axis=-1))
np.percentile(dd,[90,95,98,99],axis=0)

## DLC results
res = np.array([[
         1.45593502,  1.56175613,  1.86534404,  1.68371394,  2.04030646,
         2.26430405,  2.29727553,  1.5884356 ,  2.20272166,  1.69477133,
         2.54932645,  3.16365042,  2.52108448,  4.7560357 ,  5.13696831,
         2.87668273,  3.80720136],
       [ 1.727567  ,  1.79561019,  2.07279565,  1.89176997,  2.27816542,
         2.50483035,  2.62167926,  1.87928389,  2.64971945,  1.97338777,
         2.97771693,  4.18735141,  3.59557106,  6.83043737,  7.97818092,
         3.52630564,  5.16467199],
       [ 2.11941871,  2.10220702,  2.37149483,  2.14059765,  2.55230252,
         2.81829645,  3.03948429,  2.27664115,  3.49234151,  2.41315372,
         3.55594812,  5.67703449,  5.87166131, 10.17130605, 10.63304046,
         5.74182495,  7.08593603],
       [ 2.40299973,  2.32213969,  2.5625888 ,  2.32836243,  2.72042676,
         3.0132749 ,  3.34428889,  2.66180828,  4.15285742,  2.69790925,
         4.04134259,  7.06591366,  8.08465109, 12.56451454, 13.47454254,
         8.02540829,  8.37623894]])


## resnet with mdn results

from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'unet_resnet')
V = self.classify_val()
np.percentile(V[0],[90,95,98,99],axis=0)
res_mdn = np.array([[
         1.33690231,  1.35031777,  1.38527519,  1.34496556,  1.31486287,
         1.58903501,  1.87642883,  1.61015561,  1.94074945,  1.48852744,
         1.78586116,  3.33855858,  2.4820255 ,  4.80972121,  5.4787146 ,
         2.25345069,  2.90739776],
       [ 1.53327055,  1.56563911,  1.56710412,  1.52024155,  1.46967479,
         1.81838348,  2.13381842,  1.86714673,  2.33396092,  1.69528391,
         2.11866614,  4.52732786,  3.53221155,  7.00090467,  7.37604247,
         3.18900169,  3.97689362],
       [ 1.77418746,  1.76923494,  1.81077659,  1.71669611,  1.68055663,
         2.15293702,  2.42186984,  2.17551583,  3.00306713,  2.03097663,
         2.67549011,  6.27879295,  6.01026813,  9.79595686, 10.46869541,
         5.64375703,  5.71019346],
       [ 2.02395307,  1.91149037,  1.9630773 ,  1.8888618 ,  1.793556  ,
         2.35801906,  2.6085125 ,  2.42272951,  3.7535454 ,  2.41056106,
         3.01898574,  7.4362199 ,  8.52999432, 12.43343784, 13.06282083,
         7.79023149,  7.50091934]])

## unet with resnet

from poseConfig import aliceConfig as conf
conf.cachedir += '_moreeval'
import PoseUNet_resnet
self = PoseUNet_resnet.PoseUNet_resnet(conf,'unet_resnet_upscale')
V = self.classify_val()
np.percentile(V[0],[90,95,98,99],axis=0)
res_unet = np.array([[
         1.05970343,  1.03275964,  1.14809753,  1.16515143,  1.10287924,
         1.34909245,  1.18188198,  1.38814807,  1.46167556,  1.29960524,
         1.45588311,  1.92257185,  1.69550887,  3.4096381 ,  3.35702312,
         1.36195582,  1.74526068],
       [ 1.23078245,  1.1635586 ,  1.3024699 ,  1.31770588,  1.25754343,
         1.56928645,  1.35805825,  1.61358304,  1.77501288,  1.54141422,
         1.78610566,  2.75925073,  2.26894522,  5.9453192 ,  6.59516979,
         1.81940353,  2.52784586],
       [ 1.4384352 ,  1.33311943,  1.53371675,  1.50998481,  1.42867661,
         1.83537083,  1.58475882,  1.97676753,  2.3773975 ,  1.81195674,
         2.29952184,  4.99346799,  4.80974426, 11.71061234, 10.77242019,
         4.49559673,  4.45160127],
       [ 1.57097699,  1.4417383 ,  1.62986411,  1.6535304 ,  1.59082202,
         2.05810817,  1.75044511,  2.26493677,  2.88732075,  2.07825604,
         2.76847443,  7.3101286 , 10.13806711, 15.02330656, 13.36510544,
         8.45899003,  6.9778458 ]])

## both unet and mdn combined predictions
res_unet_mdn = np.array([[
         1.0282816 ,  1.0758637 ,  1.1631611 ,  1.23855974,  1.09031905,
         1.76056234,  3.39241648,  1.44082375,  1.93278509,  1.90886223,
         1.84620176,  3.44122251,  2.3868682 ,  5.37394855,  5.90634796,
         3.31421729,  3.24380826],
       [ 1.21911857,  1.25758945,  1.31210904,  1.41143836,  1.25347737,
         2.00568359,  3.79769606,  1.74427636,  3.24033915,  2.19404611,
         2.4300495 ,  4.93299484,  4.05366198,  7.73866029,  8.2953728 ,
         4.39114613,  4.598647  ],
       [ 1.56120887,  1.67804403,  1.52964195,  1.63107313,  1.48549727,
         2.34546864,  4.22253604,  2.2173561 ,  4.2973756 ,  2.62086827,
         3.2937051 ,  7.36836896,  6.67862229, 10.4736629 , 11.42092788,
         5.96565291,  6.73478856],
       [ 2.63523727,  2.56366787,  1.67852331,  1.75758611,  1.7787863 ,
         2.57323191,  4.4598267 ,  2.70379853,  5.05111978,  2.86584669,
         3.85309627,  9.41663734,  8.31698276, 13.29979637, 13.92469187,
         8.00356347,  8.62493168]])

##
res_mdn_official_resnet = np.array([[
         1.70731128,  1.60504444,  1.62296522,  1.38443496,  1.45595981,
         1.81261938,  3.26370495,  1.71120892,  2.49309662,  1.74549226,
         2.20047282,  2.9684426 ,  2.87415652,  5.34084172,  5.77539606,
         2.61489395,  3.01462304],
       [ 1.94046629,  1.8179954 ,  1.87964399,  1.58671815,  1.65008794,
         2.07528246,  3.64984546,  2.02027175,  3.07770157,  2.01153982,
         2.72681853,  4.14939361,  3.93323197,  7.31763092,  8.25619089,
         3.49933188,  3.98774765],
       [ 2.28026757,  2.08211258,  2.15323578,  1.81240801,  1.8368613 ,
         2.36052895,  4.04785237,  2.44004457,  3.99621015,  2.34917346,
         3.64441105,  6.01844327,  6.62986869, 10.2657138 , 11.33856891,
         5.75752333,  5.9703133 ],
       [ 2.5982614 ,  2.28115035,  2.49422134,  2.01070053,  1.98563069,
         2.57358631,  4.39920058,  2.8247397 ,  4.8433574 ,  2.56830525,
         4.18511833,  7.31832343,  8.7957528 , 12.67337665, 13.12833612,
         8.46354569,  7.52260076]])


##
res_deconv_4 = np.array([[
         1.05096569,  1.182693  ,  1.06163617,  1.17879091,  1.1914948 ,
         1.50252784,  1.27962137,  1.43324778,  1.47212356,  1.46497075,
         1.49831017,  2.17644714,  1.730035  ,  3.17607536,  3.19198076,
         1.46510505,  1.91387281],
       [ 1.2466504 ,  1.35571998,  1.21686061,  1.35434832,  1.36471203,
         1.73258353,  1.48002082,  1.66714112,  1.81571037,  1.71589769,
         1.81139375,  3.38089246,  2.33006157,  5.48676927,  6.22809825,
         1.87382732,  2.66477678],
       [ 1.40976648,  1.60844753,  1.40921836,  1.5496749 ,  1.57801472,
         2.04661989,  1.7493455 ,  2.05406113,  2.26812612,  2.04487736,
         2.3211801 ,  5.84229978,  4.37766968, 10.68928335, 10.27935305,
         4.65703738,  4.41690465],
       [ 1.57647416,  1.80267812,  1.51985177,  1.72093114,  1.6897964 ,
         2.30142203,  1.94634484,  2.29671894,  2.93157627,  2.34572634,
         2.76828742,  7.34513814,  7.54825804, 15.87991019, 13.647675  ,
         8.79198944,  6.23033242]])

##



import tensorflow as tf
tf.reset_default_graph()

kk = tf.placeholder(tf.float32,[8,32,32,8])
w_mat = np.zeros([2,2,8,8])
for ndx in range(8):
    w_mat[:,:,ndx,ndx] = 1.
w = tf.get_variable('w', [2, 2, 8,8] ,initializer=tf.constant_initializer(w_mat))
out_shape = [8,64,64,8]
ll = tf.nn.conv2d_transpose(kk, w, output_shape=out_shape, strides=[1, 2, 2, 1], padding="SAME")
# ii = np.zeros([8,32,32,8])
# ii[0,10:12,3:5,0] = 1.
ii = np.random.randn(8,32,32,8)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
rr = sess.run(ll,feed_dict={kk:ii})
plt.figure()
plt.imshow(ii[0,:,:,0])
plt.figure()
plt.imshow(rr[0,:,:,0])

##
sel = range(13,14)
aa = ddmdn[:,sel].flatten()
bb = dd[:3288,sel].flatten()
plt.figure()
plt.hist(aa-bb,range(-15,15))
ss = aa-bb

##
selndx = 13
ss = ddmdn[:,selndx] - dd[:3288,selndx]
zz = np.where( (np.abs(ss)>3) & (np.abs(ss)<np.inf))[0]
nr = 3; nc = 5
f,ax = plt.subplots(nr,nc,sharex=True,sharey=True)
ax = ax.flatten()
for ndx in range(nc*nr):
    r = np.random.choice(zz)
    ax[ndx].imshow(ims[r,:,:,0],'gray')
    ax[ndx].scatter(label[r,selndx,0],label[r,selndx,1],c='r')
    ax[ndx].scatter(pred[r, selndx, 0], pred[r, selndx, 1], c='b')
    ax[ndx].scatter(mdn[0][r, selndx, 0], mdn[0][r, selndx, 1], c='g')
ax[0].set_title('Red: Label, Blue: DLC, Green: MDN')
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
    if conf.img_dim == 1:
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
    image = tf.reshape(image, conf.imsz + (conf.img_dim,))

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
    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, True, conf.rescale)
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


