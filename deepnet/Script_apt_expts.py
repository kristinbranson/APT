import run_apt_expts_2 as rae

import sys
if sys.version_info.major > 2:
    from importlib import reload

## Accuracy over time.

import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('alice')
#rae.create_normal_dbs()
dstr = '20200604'
rae.run_normal_training(dstr=dstr) #run_type = 'submit' to actually submit jobs.
# rae.run_normal_training(dstr=dstr,queue='gpu_tesla') #run_type = 'submit' to actually submit jobs.

#dstr = 20200410 for old

##
import run_apt_expts_2 as rae
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('alice')
dstr = '20200604' # '20200410'
rae.get_normal_results(dstr=dstr) # queue = 'gpu_tesla'
# rae.get_normal_results(dstr='20200410',queue='gpu_tesla')
rae.setup('alice_difficult')
rae.get_normal_results(dstr=dstr) # queue = 'gpu_tesla'
# rae.get_normal_results(dstr='20200410',queue='gpu_tesla')

##
import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('stephen')
dstr = '20200605'
#rae.create_normal_dbs()
rae.run_normal_training(dstr=dstr) #run_type = 'submit'
# rae.run_normal_training(queue='gpu_tesla',dstr='20200411')

##
import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('stephen')
rae.get_normal_results(dstr='20200411')
rae.get_normal_results(dstr='20200411',queue='gpu_tesla')

## Accuracy over training set size

import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('alice')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
# rae.create_incremental_dbs()
alice_incr_dstr = '20200608'
rae.run_incremental_training(dstr=alice_incr_dstr) #run_type = 'submit'

##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice')
alice_incr_dstr = '20200608'
rae.get_incremental_results(dstr=alice_incr_dstr)

##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen')
# rae.create_incremental_dbs()
stephen_incr_dstr = '20200608' # '20200414'
rae.run_incremental_training(dstr=stephen_incr_dstr) #run_type = 'submit'

##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen')
stephen_incr_dstr = '20200608' # '20200414'
rae.get_incremental_results(dstr=stephen_incr_dstr)

## Whole dataset training

import run_apt_expts_2 as rae
for data_type in ['roian','brit0','brit1','brit2','romain','larva']:
    reload(rae)
    rae.setup(data_type)
    # rae.create_normal_dbs()
    rae.run_normal_training() #run_type = 'submit') # to actually submit jobs.


## Brits experiments

## training
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
for britnum in range(3):
    rae.setup('brit{}'.format(britnum))
    # rae.cv_train_britton() # use skip_db=False, run_type='submit' to actually rerun it
    rae.cv_train_from_mat(queue='gpu_tesla',dstr='20200417')
    rae.cv_train_from_mat(dstr='20200417')

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
for britnum in range(3):
    rae.setup('brit{}'.format(britnum))
    rae.get_cv_results(queue='gpu_tesla',dstr='20200417',db_from_mdn_dir=True)
    rae.get_cv_results(queue='gpu_rtx',dstr='20200417',db_from_mdn_dir=True)


## Romains experiments

## CV Training - all views

import run_apt_expts_2 as rae
reload(rae)
rae.setup('romain','')
rae.cv_train_from_mat() # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
reload(rae)
rae.setup('romain')
rae.get_cv_results(num_splits=6)



## Roain's expts
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('roian','')
rae.cv_train_from_mat(dstr='20200430') # skip_db=False,run_type='submit'
rae.cv_train_from_mat(dstr='20200430',queue='gpu_tesla') # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('roian','')
rae.get_cv_results(queue='gpu_rtx',dstr='20200430',db_from_mdn_dir=True)
rae.get_cv_results(queue='gpu_tesla',dstr='20200430',db_from_mdn_dir=True)

## Larva
from importlib import reload
import run_apt_expts_2 as rae
reload(rae)
rae.setup('larva','')
rae.cv_train_from_mat(dstr='20200428') # skip_db=False,run_type='submit'
rae.cv_train_from_mat(dstr='20200428',queue='gpu_tesla_large') # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
reload(rae)
rae.setup('larva')
rae.get_cv_results(dstr='20200428',db_from_mdn_dir=True) # skip_db=False,run_type='submit'
rae.get_cv_results(dstr='20200428',queue='gpu_tesla_large',db_from_mdn_dir=True) #



## Single animal vs multiple animal for Stephen

import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen','')
rae.create_run_individual_animal_dbs_stephen(run_type='status') # use run_type='submit' to redo.

## Results

import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen')
rae.get_individual_animal_results_stephen()

##
# Run /groups/branson/bransonlab/mayank/APT_develop/ScriptStephenSingleAnimalResults.m
out_file = '/groups/branson/home/kabram/temp/stephen_single_fly_results.mat'
from scipy import io as sio
import multiResData
import run_apt_expts_2 as rae
ss = sio.loadmat(out_file)['out']
db =['/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/mdn/view_0/apt_expt/train_TF.tfrecords',
'/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/mdn/view_1/apt_expt/train_TF.tfrecords']

ps = [50,75,90,95,97]
cmap = np.array([[0.5200, 0, 0],
                 [1.0000, 0.5200, 0],
                 [0.4800, 1.0000, 0.5200],
                 [0, 0.5200, 1.0000],
                 [0, 0, 0.5200]])

tt = ['same', 'different','random']
for view in range(2):
    im, locs, info = multiResData.read_and_decode_without_session(db[view], 5)
    ex_im = np.array(im)[0,...,0]
    ex_loc = np.array(locs)[0,...]
    npts = 5
    n_types = 3
    nc = n_types  # int(np.ceil(np.sqrt(n_types)))
    nr = 1  # int(np.ceil(n_types/float(nc)))
    f, axx = plt.subplots(nr, nc, figsize=(12, 8))
    axx = axx.flat
    for idx in range(n_types):
        dd = ss[0,view][0,idx]
        mm = np.percentile(dd, ps, axis=0)

        ax = axx[idx]
        if ex_im.ndim == 2:
            ax.imshow(ex_im, 'gray')
        elif ex_im.shape[2] == 1:
            ax.imshow(ex_im[:, :, 0], 'gray')
        else:
            ax.imshow(ex_im)

        for pt in range(ex_loc.shape[0]):
            for pp in range(mm.shape[0]):
                c = plt.Circle(ex_loc[pt, :], mm[pp, pt], color=cmap[pp, :], fill=False)
                ax.add_patch(c)
        ax.set_title(tt[idx])
        ax.axis('off')

    f.tight_layout()

## deeplabcut augment vs no-augment
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.run_dlc_augment_training() # run_type='submit'

##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.get_dlc_results()

##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.run_dlc_augment_training() #run_type='submit'

##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.get_dlc_results()


## Carsen

import run_apt_expts_2 as rae
reload(rae)
rae.setup('carsen')
rae.cv_train_from_mat(skip_db=True,run_type='status')

## results
import run_apt_expts_2 as rae
reload(rae)
rae.setup('carsen')
rae.get_cv_results(num_splits=6)



## Active Learning Experiment:

import run_apt_expts_2 as rae
reload(rae)
import os
import time
dtype = 'stephen'
rae.setup(dtype)
view = 0
for r_round in range(15):
    rae.run_active_learning(r_round,'active',view)
    rae.run_active_learning(r_round,'random',view)
    active_model = '/nrs/branson/mayank/apt_cache/{}/mdn/view_{}/active_round{}/deepnet-20000.index'.format(rae.proj_name,view,r_round)
    random_model = '/nrs/branson/mayank/apt_cache/{}/mdn/view_{}/random_round{}/deepnet-20000.index'.format(rae.proj_name,view,r_round)
    while not (os.path.exists(active_model) and os.path.exists(random_model)):
        time.sleep(200)

## results
import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen',0)
rae.get_active_results(num_rounds=15,view=1)


## Original DLC and Leap Training
import run_apt_expts_2 as rae
dtypes = ['alice']#,'stephen']
for dd in dtypes:
    reload(rae)
    rae.setup(dd)
    # rae.train_deepcut_orig()
    rae.train_leap_orig()


## Results Orig leap vs our leap
import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
dtypes = ['alice']#,'stephen']
for dd in dtypes:
    reload(rae)
    rae.setup(dd)
    rae.get_leap_results()


## Pretrained vs not training
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice')
rae.train_no_pretrained()

## Pretrained vs not results
import run_apt_expts_2 as rae
# reload(rae)
rae.setup('alice')
rae.get_no_pretrained_results()


## Difficult examples
## Create the db
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice_difficult')
rae.create_gt_db()

##

import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('alice_difficult')
rae.get_normal_results()


## LEAP
import run_apt_expts_2 as rae
reload(rae)
rae.setup('leap_fly')
rae.create_normal_dbs()
rae.create_gt_db()

##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('leap_fly')
rae.run_normal_training()

##
import run_apt_expts_2 as rae
# reload(rae)
rae.setup('leap_fly')
rae.get_normal_results()

## Our leap vs leap original
from scipy import io as sio
import os
import PoseTools as pt
import multiResData
import apt_expts
import APT_interface as apt
import run_apt_expts_2 as rae
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
rae.setup('leap_fly')
view = 0

L = sio.loadmat('/groups/branson/home/kabram/Downloads/figure_data(1).mat')
labels = np.transpose(L['positions_test'],[2,0,1])
preds = np.transpose(L['preds_test'][0,0][2],[2,0,1])
out_leap = [[preds,labels,[],[],[],[]]]
dd_leap = np.sqrt(np.sum((labels-preds)**2,1))
dd_leap = dd_leap.T

cache_dir = '/nrs/branson/mayank/apt_cache'
exp_name = 'apt_expt'
train_name = 'deepnet'
gt_file = os.path.join(cache_dir, rae.proj_name, 'gtdata', 'gtdata_view{}{}.tfrecords'.format(view, rae.gt_name))
H = multiResData.read_and_decode_without_session(gt_file, 32)
ex_im = np.array(H[0][0])[:, :, 0]
ex_loc = np.array(H[1][0])
our_res = pt.pickle_load('/nrs/branson/mayank/apt_cache/leap_dset/leap/view_0/apt_expt/deepnet_results.p')
our_preds = our_res[0][-1][0]
our_labels = our_res[0][-1][1]

conf = apt.create_conf(rae.lbl_file,0,'apt_expt',cache_dir,'leap')
orig_leap_models = ['/nrs/branson/mayank/apt_cache/leap_dset/leap/view_0/apt_expt/weights-045.h5',]
orig_leap = apt_expts.classify_db_all(conf,gt_file,orig_leap_models,'leap',name=train_name)

out_dict = {'leap':out_leap,'our leap':our_res[0],'leap_orig':orig_leap}
rae.plot_hist([out_dict,ex_im,ex_loc])

## mdn with and without unet
import run_apt_expts_2 as rae
rae.setup('alice')
rae.run_mdn_no_unet()


##
import run_apt_expts_2 as rae
rae.setup('alice')
rae.get_mdn_no_unet_results()

## Alice active learning different conditions

import h5py
from scipy import io as sio
A = sio.loadmat('/nrs/branson/mayank/apt_cache/multitarget_bubble/multitarget_bubble_expandedbehavior_20180425_condinfo.mat')

for cond in range(1,4):
    selndx = A['label_cond'][:,0] == cond

    import PoseTools as pt
    B = {}
    B['active'] = pt.pickle_load('/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/active_round0/deepnet_results.p')
    B['random'] = pt.pickle_load('/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/random_round0/deepnet_results.p')



    aa = {}
    for tt in ['active','random']:
        cur = B[tt][0]
        for ndx in range(36):
            cur[ndx][0] = cur[ndx][0][selndx,...]
            cur[ndx][1] = cur[ndx][1][selndx,...]
            cur[ndx][-1] = ndx
        cur.insert(0,cur[0])
        aa[tt] = cur

    rae.plot_results(aa)
    f = plt.gcf()
    f.savefig('/groups/branson/home/kabram/temp/alice_active_cond{}.png'.format(cond))



## Extra code to view the results
for split_num in range(3):
    res_file= '/nrs/branson/mayank/apt_cache/wheel_rig_tracker_feb_2017_cam0/mdn/view_0/cv_split_{}/deepnet_results.p'.format(split_num)
    db = '/nrs/branson/mayank/apt_cache/wheel_rig_tracker_feb_2017_cam0/mdn/view_0/cv_split_{}/val_TF.tfrecords'.format(split_num)

    import multiResData
    import PoseTools as pt
    import easydict
    res = pt.pickle_load(res_file)[0]
    ims,locs,info = multiResData.read_and_decode_without_session(db,5,())
    ims = np.array(ims)
    locs = np.array(locs)
    pred = res[-1][0]
    dd = np.sqrt(np.sum((res[-1][0]-res[-1][1])**2,axis=-1))

    gg = np.percentile(dd,[50,75,90,96],axis=0)
    pt.show_result_hist(ims[0,...],locs[0,...],gg)

    ndx = np.random.choice(ims.shape[0],6)
    pt.show_result(ims,ndx,locs,pred)
    conf = easydict.EasyDict()
    conf.adjust_contrast = True
    conf.clahe_grid_size = 20
    cims = pt.adjust_contrast(ims,conf)
    pt.show_result(cims,ndx,locs,pred)


## Original DLC and Leap Training
import run_apt_expts_2 as rae
dtypes = ['alice','stephen']
for dd in dtypes:
    reload(rae)
    rae.setup(dd)
    rae.train_deepcut_orig()
    rae.train_leap_orig()


## Results Orig leap vs our leap
import run_apt_expts_2 as rae
dtypes = ['alice','stephen']
for dd in dtypes:
    reload(rae)
    rae.setup(dd)
    rae.get_leap_results()


## Orig DPK
import run_apt_expts_2 as rae
dtypes = ['alice', 'stephen']
for dd in dtypes:
    reload(rae)
    rae.setup(dd)
    rae.train_dpk_orig()


## Videos for results


## Alice

starts = [4525,2550,3941]
starts = [s-1 for s in starts]
ends = [4850,3075,4025]
ends = [s-1 for s in ends]

movid = ['/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00020_CsChr_RigB_20150908T133237/movie.ufmf']
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice')
for ndx in range(len(starts)):
    for train_type in rae.all_models:
        rae.track(movid=movid,
              start_ndx=starts[ndx],
              end_ndx=ends[ndx],
                  train_type=train_type
                  )

## Stephen

starts = [500]
ends = [1000]

movid = 660
import run_apt_expts_2 as rae
reload(rae)
rae.setup('stephen')
for ndx in range(len(starts)):
    for train_type in rae.all_models:
        rae.track(movid=movid, train_type=train_type,
                  start_ndx=starts[ndx],end_ndx=ends[ndx] )




## Roian

movid = 2
starts = [68745,104640,148094]
n_frames = 300
import run_apt_expts_2 as rae
reload(rae)
rae.setup('roian')
for ndx in range(len(starts)):
    for train_type in rae.all_models:
        rae.track(movid=movid,
              start_ndx=starts[ndx],
              end_ndx=starts[ndx]+n_frames,
                  train_type=train_type
                  )



## Larva

starts = [900]
ends = [1100]
movid = 3
exp_name = 'cv_split_7'

import run_apt_expts_2 as rae
reload(rae)
rae.setup('larva')
for ndx in range(len(starts)):
    for train_type in rae.all_models:
        rae.track(movid=movid, train_type=train_type,
                  start_ndx=starts[ndx],end_ndx=ends[ndx],
                  exp_name=exp_name)


## Romain

starts = [26300]
ends = [26700]
movid = 1

import run_apt_expts_2 as rae
reload(rae)
rae.setup('romain')
for ndx in range(len(starts)):
    for train_type in rae.all_models:
        rae.track(movid=movid, train_type=train_type,
                  start_ndx=starts[ndx],end_ndx=ends[ndx])


##


starts = [38000]
ends = [38500]

for britnum in range(3):

    if britnum == 0:
        movid = 6
    else:
        movid = 4

    import run_apt_expts_2 as rae
    reload(rae)
    rae.setup('brit{}'.format(britnum))
    for ndx in range(len(starts)):
        for train_type in rae.all_models:
            rae.track(movid=movid, train_type=train_type,
                      start_ndx=starts[ndx],end_ndx=ends[ndx])

