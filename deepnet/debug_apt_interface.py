url = 'https://www.dropbox.com/sh/sfku14q2aja3xo3/AADzyjRJDec7G1ncQIaL8LIka?dl=1'
import urllib

urllib.request.urlretrieve(url,'/home/kabram/temp/tt')

##
import run_apt_expts as rae
rae.setup('stephen')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.get_dlc_results()


##
cmd = '-name 20190911T071419 -view 1 -cache \/groups/branson/home/kabram/.apt/tpebb6275b_8f29_4607_ae7d_0aa29a7efc2c -err_file /groups/branson/home/kabram/.apt/tpebb6275b_8f29_4607_ae7d_0aa29a7efc2c/cluster-deeplab/mdn/view_0/20190911T071419/trk/run032_pez3001_20190128_expt0129000017060801_vid0002_supplement_trn20190911T071419_iter1000_20190911T072419.err -model_files /groups/branson/home/kabram/.apt/tpebb6275b_8f29_4607_ae7d_0aa29a7efc2c/cluster-deeplab/mdn/view_0/20190911T071419/deepnet-1000 -type mdn /groups/branson/home/kabram/.apt/tpebb6275b_8f29_4607_ae7d_0aa29a7efc2c/cluster-deeplab/20190911T071419_20190911T071450.lbl track -mov /groups/card/home/wellsc/Internship/RawVideo/run032_pez3001_20190128/highSpeedSupplement/run032_pez3001_20190128_expt0129000017060801_vid0002_supplement.mp4 -out /groups/branson/home/kabram/.apt/tpebb6275b_8f29_4607_ae7d_0aa29a7efc2c/cluster-deeplab/mdn/view_0/20190911T071419/trk/run032_pez3001_20190128_expt0129000017060801_vid0002_supplement_trn20190911T071419_iter1000_20190911T072419.trk -start_frame 1 -end_frame 521 -crop_loc 1 384 416 832'

import APT_interface as apt
apt.main(cmd.split())

##
import run_apt_expts as rae
reload(rae)
rae.setup('leap_fly')
rae.create_gt_db()

##
import APT_interface as apt
cmd = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl -name alice_randsplit_round_4 -cache /nrs/branson/mayank/apt_cache -conf_params use_pretrained_weights False  batch_size 8  trange 5  decay_steps 20000  save_step 4000  rrange 10  adjust_contrast False  mdn_use_unet_loss True  dl_steps 40000  normalize_img_mean False  maxckpt 20  -type mdn  -train_name no_pretrained  -view 1 train -skip_db -use_cache'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
apt.main(cmd.split())


<<<<<<< HEAD
import APT_interface as apt

cmd = '-cache_dir /home/kabram/temp/apt_cache -name test_sz /home/kabram/temp/20190129T180959_20190129T181147.lbl train -use_cache'

apt.main(cmd.split())
##
A = movies.Movie('/home/kabram/Dropbox (HHMI)/Results/FlyHogHof/CVResults/frame_of_take_offHOG_run010_pez3002_20140528_expt0024000009730145_vid0012_20150510.avi')
=======
>>>>>>> c6e2fc4759574f4d2ee6c7ee54ce1e75f1061bba
##
import run_apt_expts as rae
reload(rae)
rae.setup('romain',0)
rae.get_cv_results(num_splits=6)



##
import PoseTools
import os
import glob
import APT_interface as apt
import apt_expts
import re
import run_apt_expts as rae
import multiResData
import numpy as np

reload(apt_expts)
import PoseUNet_resnet

reload(PoseUNet_resnet)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# mdn_names = ['separate_groups', 'joint_groups' ]
mdn_names = ['explicit_offsets', 'implicit_offsets' ]
mdn_names = ['normal', 'True' ]
out_dir = '/groups/branson/home/kabram/temp'

out = {}
db_file = '/nrs/branson/mayank/apt_cache/Test/mdn/view_0/larva_compare/val_TF.tfrecords'
for n in mdn_names:
    cdir = os.path.dirname(db_file)
    if n == 'deepnet':
        tfile = os.path.join(cdir, 'traindata')
    else:
        tfile = os.path.join(cdir, 'Test_{}_traindata'.format(n))

    if not os.path.exists(tfile):
        continue
    A = PoseTools.pickle_load(tfile)
    conf = A[1]

    files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*.index").format(n))
    files.sort(key=os.path.getmtime)
    aa = [int(re.search('-(\d*).index', f).groups(0)[0]) for f in files]
    aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
    if any([a < 0 for a in aa]):
        bb = int(np.where(np.array(aa) < 0)[0]) + 1
        files = files[bb:]
    files = [f.replace('.index', '') for f in files]
    files = files[-1:]
    # if len(files) > 8:
    #     gg = len(files)
    #     sel = np.linspace(0, len(files) - 1, 8).astype('int')
    #     files = [files[s] for s in sel]
    #
    mdn_out = apt_expts.classify_db_all(conf, db_file, files, 'mdn', name=n)
    out[n] = mdn_out

H = multiResData.read_and_decode_without_session(db_file, conf)
ex_ims = np.array(H[0][0])
ex_locs = np.array(H[1][0])
f = rae.plot_hist([out,ex_ims,ex_locs],[50,75,90,95,97])


##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseTools as pt
import PoseUNet_resnet

a = pt.pickle_load('/nrs/branson/mayank/apt_cache/Test/mdn/view_0/larva_compare/Test_normal_traindata')
a = pt.pickle_load('/nrs/branson/mayank/apt_cache/Test/mdn/view_0/cv_split_0/traindata')

conf = a[1]
self = PoseUNet_resnet.PoseUMDN_resnet(conf,name='deepnet')
self.train_data_name = 'traindata'
self.train_umdn(True)




##
data_type = 'roian'

import APT_interface as apt
import h5py
import PoseTools
import os
import time
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import apt_expts
import os
import ast
import apt_expts
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = ''

gt_lbl = None
lbl_file = '/groups/branson/bransonlab/apt/experiments/data/roian_apt_dlstripped.lbl'
op_af_graph = '\(0,1\),\(0,2\),\(0,3\),\(1,2\),\(1,3\),\(2,3\)'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
lbl.close()
cache_dir = '/nrs/branson/mayank/apt_cache'
all_models = ['openpose']

gpu_model = 'GeForceRTX2080Ti'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
n_splits = 3


common_conf = {}
common_conf['rrange'] = 10
common_conf['trange'] = 5
common_conf['mdn_use_unet_loss'] = True
common_conf['dl_steps'] = 100000
common_conf['decay_steps'] = 20000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8
common_conf['maxckpt'] = 20
cache_dir = '/nrs/branson/mayank/apt_cache'
train_name = 'deepnet'


assert gt_lbl is None
all_view = []
for view in range(nviews):
    out_exp = {}
    for tndx in range(len(all_models)):
        train_type = all_models[tndx]

        out_split = None
        for split in range(n_splits):
            exp_name = 'cv_split_{}'.format(split)
            mdn_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)

            if op_af_graph is not None:
                conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
            files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
            files.sort(key=os.path.getmtime)
            files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
            aa = [int(re.search('-(\d*)',f).groups(0)[0]) for f in files]
            aa = [b-a for a,b in zip(aa[:-1],aa[1:])]
            if any([a<0 for a in aa]):
                bb = int(np.where(np.array(aa)<0)[0])+1
                files = files[bb:]
            n_max = 10
            if len(files)> n_max:
                gg = len(files)
                sel = np.linspace(0,len(files)-1,n_max).astype('int')
                files = [files[s] for s in sel]

            afiles = [f.replace('.index', '') for f in files]
            afiles = afiles[-1:]
            db_file = os.path.join(mdn_conf.cachedir,'val_TF.tfrecords')
            mdn_out = apt_expts.classify_db_all(conf,db_file,afiles,train_type,name=train_name)

        out_exp[train_type] = out_split
    all_view.append(out_exp)



##

in_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4/leap_train.h5'
out_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/leap_train.h5'

import h5py
H = h5py.File(in_file,'r')
locs = H['joints'][:]
hf = h5py.File(out_file,'w')

hmaps = PoseTools.create_label_images(locs, conf.imsz[:2], 1, 5)
hmaps += 1
hmaps /= 2  # brings it back to [0,1]

hf.create_dataset('box', data=H['box'][:])
hf.create_dataset('confmaps', data=hmaps)
hf.create_dataset('joints', data=locs)

hf.close()
H.close()

##
import multiResData
A = multiResData.read_and_decode_without_session('/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/gtdata/gtdata_view0.tfrecords',5,())
ims = np.array(A[0])
locs = np.array(A[1])

out_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/leap_gt.h5'

import h5py
hf = h5py.File(out_file,'w')

hf.create_dataset('box', data=ims)
hf.create_dataset('joints', data=locs)

hf.close()

##

gt_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/leap_gt.h5'
out_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/gt_preds_002.mat/leap_gt.h5'

a = h5py.File(gt_file,'r')
b = h5py.File(out_file,'r')
locs = a['joints'][:]
preds = np.transpose(b['positions_pred'][:],[0,2,1])
hmaps = b['conf_pred'][:]

dd = np.sqrt(np.sum((locs-preds)**2,axis=-1))
np.percentile(dd,[50,75,90],axis=0)

##

import os
import APT_interface as apt
import glob
import re
import numpy as np
import multiResData
import math
import h5py
import PoseTools
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
cache_dir = '/nrs/branson/mayank/apt_cache'
exp_name = 'apt_expt_leap_original'
train_name = 'deepnet'
view = 0
train_type = 'leap'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
lbl.close()

gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))

conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)

split = False
use_cache = True
train_data = []
val_data = []

# collect the images and labels in arrays
out_fns = [lambda data: train_data.append(data), lambda data: val_data.append(data)]
splits, __ = apt.db_from_cached_lbl(conf, out_fns, split, None)

# save the split data
try:
    with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
        json.dump(splits, f)
except IOError:
    logging.warning('SPLIT_WRITE: Could not output the split data information')

for ndx in range(2):
    if not split and ndx == 1:  # nothing to do if we dont split
        continue

    if ndx == 0:
        cur_data = train_data
        out_file = os.path.join(conf.cachedir, 'leap_train.h5')
    else:
        cur_data = val_data
        out_file = os.path.join(conf.cachedir, 'leap_val.h5')

    ims = np.array([i[0] for i in cur_data])
    locs = np.array([i[1] for i in cur_data])
    info = np.array([i[2] for i in cur_data])
    hmaps = PoseTools.create_label_images(locs, conf.imsz[:2], 1, 3)
    hmaps += 1
    hmaps /= 2  # brings it back to [0,1]

    if info.size > 0:
        hf = h5py.File(out_file, 'w')
        hf.create_dataset('box', data=ims)
        hf.create_dataset('confmaps', data=hmaps)
        hf.create_dataset('joints', data=locs)
        hf.create_dataset('exptID', data=info[:, 0])
        hf.create_dataset('framesIdx', data=info[:, 1])
        hf.create_dataset('trxID', data=info[:, 2])
        hf.close()

##

import os
import APT_interface as apt
import glob
import re
import numpy as np
import multiResData
import math
import h5py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
cache_dir = '/nrs/branson/mayank/apt_cache'
exp_name = 'apt_expt'
train_name = 'deepnet'
view = 0
train_type = 'mdn'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
lbl.close()

gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))

conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
conf.normalize_img_mean = False
files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
files.sort(key=os.path.getmtime)
files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
if any([a < 0 for a in aa]):
    bb = int(np.where(np.array(aa) < 0)[0]) + 1
    files = files[bb:]
n_max = 6
if len(files) > n_max:
    gg = len(files)
    sel = np.linspace(0, len(files) - 1, n_max).astype('int')
    files = [files[s] for s in sel]

out_file = os.path.join(conf.cachedir, train_name + '_results.p')
afiles = [f.replace('.index', '') for f in files]

for m in afiles[-1:]:
    tf_iterator = multiResData.tf_reader(conf, gt_file, False)
    tf_iterator.batch_size = 1
    read_fn = tf_iterator.next
    pred_fn, close_fn, _ = apt.get_pred_fn(train_type, conf, m, name=train_name)
    bsize = conf.batch_size
    all_f = np.zeros((bsize,) + conf.imsz + (conf.img_dim,))
    n = tf_iterator.N
    pred_locs = np.zeros([n, conf.n_classes, 2])
    unet_locs = np.zeros([n, conf.n_classes, 2])
    mdn_locs = np.zeros([n, conf.n_classes, 2])
    n_batches = int(math.ceil(float(n) / bsize))
    labeled_locs = np.zeros([n, conf.n_classes, 2])
    all_ims = np.zeros([n, conf.imsz[0], conf.imsz[1], conf.img_dim])

    info = []
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            all_f[ndx, ...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])
        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']
        ulocs = ret_dict['locs_unet']
        hmaps = ret_dict['hmaps']

        for ndx in range(ppe):
            pred_locs[cur_start + ndx, ...] = base_locs[ndx, ...]
            unet_locs[cur_start + ndx, ...] = ulocs[ndx, ...]
            mdn_locs[cur_start + ndx, ...] = ret_dict['locs_mdn'][ndx, ...]
            all_ims[cur_start + ndx, ...] = all_f[ndx, ...]

    close_fn()


##

import APT_interface as apt
import h5py
import PoseTools
import os
import time
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import apt_expts
import os
import ast
import apt_expts
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped.lbl'
gt_lbl = lbl_file
op_af_graph = '\(0,1\),\(0,2\),\(2,3\),\(1,3\),\(0,4\),\(1,4\)'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
lbl.close()

cache_dir = '/nrs/branson/mayank/apt_cache'

train_type = 'mdn'
exp_name = 'apt_exp'
for view in range(nviews):
    conf = apt.create_conf(gt_lbl, view, exp_name, cache_dir, train_type)
    gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}.tfrecords'.format(view))
    apt.create_tfrecord(conf,False,None,False,True,[gt_file])



##
import APT_interface as apt
import os
import glob
import apt_expts
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cache_dir = '/nrs/branson/mayank/apt_cache'
exp_name = 'apt_expt'
train_name = 'deepnet'
view =0

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
lbl.close()

train_type = 'leap'
conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))
files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
mdn_out = apt_expts.classify_db_all(conf ,gt_file ,files ,train_type ,name=train_name)

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
import easydict
reload(apt)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
conf = apt.create_conf('/home/mayank/temp/apt_cache/multitarget_bubble/20190129T180959_20190129T181147.lbl',0,'20190129T180959','/home/mayank/temp/apt_cache','leap')
# conf.label_blur_rad = 5
# apt.create_leap_db(conf,False,use_cache=True)
args = easydict.EasyDict()
args.use_cache = True
args.skip_db = True
args.train_name = 'deepnet'
conf.op_affinity_graph = [[0,1],[1,2],[2,0]]
apt.train_openpose(conf,args,False)

##
cmd_str = '-name 20190129T180959 -view 1 -cache /home/mayank/temp/apt_cache  -conf_params  label_blur_rad 7 dl_steps 5000 leap_use_default_lr False -train_name decay_lr -type leap /home/mayank/temp/apt_cache/multitarget_bubble/20190129T180959_20190129T181147.lbl train -use_cache -skip_db'

import APT_interface as apt
apt.main(cmd_str.split())

##
# debug postprocessing
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
# print a_sz
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

