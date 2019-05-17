import run_apt_expts as rae


## Accuracy over time.

import run_apt_expts as rae
reload(rae)
rae.setup('alice')
rae.run_normal_training() #run_type = 'submit' to actually submit jobs.

##
import run_apt_expts as rae
reload(rae)
rae.setup('alice')
rae.get_normal_results()

##
import run_apt_expts as rae
reload(rae)
rae.setup('stephen')
rae.run_normal_training() #run_type = 'submit' to actually submit jobs.

##
import run_apt_expts as rae
reload(rae)
rae.setup('stephen')
rae.get_normal_results()


## Accuracy over training set size

import run_apt_expts as rae
reload(rae)
rae.setup('alice')
# rae.create_incremental_dbs()
rae.run_incremental_training() #run_type = 'submit' to actually submit jobs.

##
import run_apt_expts as rae
reload(rae)
rae.setup('alice')
rae.get_incremental_results()

##
import run_apt_expts as rae
reload(rae)
rae.setup('stephen')
# rae.create_incremental_dbs()
rae.run_incremental_training() #run_type = 'submit' to actually submit jobs.

##
import run_apt_expts as rae
reload(rae)
rae.setup('stephen')
rae.get_incremental_results() #run_type = 'submit' to actually submit jobs.


## Single animal vs multiple animal for Stephen

import run_apt_expts as rae
reload(rae)
rae.setup('stephen','')
rae.create_run_individual_animal_dbs_stephen(run_type='status') # use run_type='submit' to redo.

## Results

import run_apt_expts as rae
reload(rae)
rae.setup('stephen')
rae.get_individual_animal_results_stephen()

##
# Run /groups/branson/bransonlab/mayank/APT_develop/ScriptStephenSingleAnimalResults.m
out_file = '/groups/branson/home/kabram/temp/stephen_single_fly_results.mat'
from scipy import io as sio
import multiResData
import run_apt_expts as rae
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
import run_apt_expts as rae
reload(rae)
rae.setup('alice')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.run_dlc_augment_training() # run_type='submit'

##
import run_apt_expts as rae
reload(rae)
rae.setup('alice')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.get_dlc_results()

##
import run_apt_expts as rae
reload(rae)
rae.setup('stephen')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.run_dlc_augment_training() #run_type='submit'

##
import run_apt_expts as rae
reload(rae)
rae.setup('stephen')
for round in range(5):
    rae.dlc_aug_use_round = round
    rae.get_dlc_results()

## Whole dataset training

import run_apt_expts as rae
for data_type in ['roian','brit0','brit1','brit2','romain','larva']:
    reload(rae)
    rae.setup(data_type)
    # rae.create_normal_dbs()
    rae.run_normal_training() #run_type = 'submit') # to actually submit jobs.


## Brits experiments

## training
import run_apt_expts as rae
reload(rae)
for britnum in range(3):
    rae.setup('brit{}'.format(britnum))
    rae.cv_train_britton() # use skip_db=False, run_type='submit' to actually rerun it

## results
import run_apt_expts as rae
reload(rae)
for britnum in range(3):
    rae.setup('brit{}'.format(britnum))
    rae.get_cv_results(num_splits=3)


## Romains experiments

## CV Training - all views

import run_apt_expts as rae
reload(rae)
rae.setup('romain','')
rae.cv_train_from_mat() # skip_db=False,run_type='submit'

## results
import run_apt_expts as rae
reload(rae)
rae.setup('romain','')
rae.get_cv_results(num_splits=6)



## Roain's expts
import run_apt_expts as rae
reload(rae)
rae.setup('roian','')
rae.cv_train_from_mat() # skip_db=False,run_type='submit'

## results
import run_apt_expts as rae
reload(rae)
rae.setup('roian')
rae.get_cv_results(num_splits=4)


## Larva

import run_apt_expts as rae
reload(rae)
rae.setup('larva','')
rae.cv_train_from_mat() # skip_db=False,run_type='submit'

## results
import run_apt_expts as rae
reload(rae)
rae.setup('larva')
rae.get_cv_results(num_splits=8)


## Carsen

import run_apt_expts as rae
reload(rae)
rae.setup('carsen')
rae.cv_train_from_mat(skip_db=True,run_type='status')

## results
import run_apt_expts as rae
reload(rae)
rae.setup('carsen')
rae.get_cv_results(num_splits=6)



## Active Learning Experiment:

import run_apt_expts as rae
reload(rae)
import os
import time
rae.setup('alice','')
for round in range(8):
    rae.run_active_learning(round,'active')
    rae.run_active_learning(round,'random')
    active_model = '/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/active_round{}/deepnet-00020000.index'.format(round)
    random_model = '/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/random_round{}/deepnet-00020000.index'.format(round)
    while not (os.path.exists(active_model) and os.path.exists(random_model)):
        time.sleep(200)

## results
import run_apt_expts as rae
reload(rae)
rae.setup('alice')
rae.get_active_results()


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


## Videos for results


## Alice

starts = [4525,2550,3941]
starts = [s-1 for s in starts]
ends = [4850,3075,4025]
ends = [s-1 for s in ends]

movid = ['/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00020_CsChr_RigB_20150908T133237/movie.ufmf']
import run_apt_expts as rae
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
import run_apt_expts as rae
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
import run_apt_expts as rae
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

import run_apt_expts as rae
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

import run_apt_expts as rae
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

    import run_apt_expts as rae
    reload(rae)
    rae.setup('brit{}'.format(britnum))
    for ndx in range(len(starts)):
        for train_type in rae.all_models:
            rae.track(movid=movid, train_type=train_type,
                      start_ndx=starts[ndx],end_ndx=ends[ndx])

