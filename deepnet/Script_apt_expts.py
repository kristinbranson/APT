
## Stephen's Experiments

## Single animal vs multiple animal

import run_apt_expts as rae
rae.setup('stephen','')
rae.create_run_individual_animal_dbs_stephen(run_type='status') # use run_type='submit' to redo.


## deeplabcut augment vs no-augment
import run_apt_expts as rae
rae.setup('stephen','')
rae.run_dlc_augment_training(run_type='submit')


## Brits experiments

## training
import run_apt_expts as rae
for britnum in range(3):
    rae.setup('brit{}'.format(britnum),'')
    rae.cv_train_britton() # use skip_db=False, run_type='submit' to actually rerun it

## results
import run_apt_expts as rae
for britnum in range(3):
    rae.setup('brit{}'.format(britnum),'')
    rae.get_cv_results(num_splits=3)


## Romains experiments

## CV Training - all views

import run_apt_expts as rae
rae.setup('romain','')
rae.cv_train_from_mat() # skip_db=False,run_type='submit'

## results
import run_apt_expts as rae
rae.setup('romain','')
rae.get_cv_results(num_splits=6)



## Roain's expts
import run_apt_expts as rae
rae.setup('roian','')
rae.cv_train_from_mat() # skip_db=False,run_type='submit'

## results
import run_apt_expts as rae
rae.setup('roian')
rae.get_cv_results(num_splits=4)


## Larva

import run_apt_expts as rae
rae.setup('larva','')
rae.cv_train_from_mat() # skip_db=False,run_type='submit'

## results
import run_apt_expts as rae
rae.setup('larva')
rae.get_cv_results(num_splits=8)



## Active Learning Experiment:

import run_apt_expts as rae
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


##

