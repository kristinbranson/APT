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
# dstr = '20200604'
# dstr = '20200706'
# rae.run_normal_training(dstr=dstr) #run_type = 'submit'
dstr = '20210708' #'20210629'
rae.create_normal_dbs(expname='touching')
rae.run_normal_training(dstr=dstr,expname='touching') #run_type = 'submit'
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
# dstr = '20200706' #'20200604' # '20200410'
# rae.get_normal_results(dstr=dstr)
dstr = '20210708' #'20210629'
rae.get_normal_results(dstr=dstr,exp_name='touching')
# rae.setup('alice_difficult')
# rae.get_normal_results(dstr=dstr)

##
import run_apt_expts_2 as rae
import sys
from importlib import reload
reload(rae)
rae.setup('stephen')
# dstr = '20200605'
dstr = '20200706'
#rae.create_normal_dbs()
rae.run_normal_training(dstr=dstr) #run_type = 'submit'

##
import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('stephen')
dstr = '20200706' #'20200411'
rae.get_normal_results(dstr=dstr)

## Accuracy over training set size

import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('alice')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
# # rae.create_incremental_dbs()
# alice_incr_dstr = '20200716' #'20200608'
# rae.run_incremental_training(dstr=alice_incr_dstr) #run_type = 'submit'
dstr = '20210708' #'20210629'
import run_apt_ma_expts as rae_ma
import os
robj = rae_ma.ma_expt('alice')
ma_inc_file = os.path.join(robj.trnp_dir,'inc_data.pkl')
ma_loc = os.path.join(robj.trnp_dir, 'grone', rae_ma.loc_file_str)

rae.create_incremental_dbs_ma(ma_inc_file, ma_loc)
rae.run_incremental_training(dstr=dstr) #run_type = 'submit'


##
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
alice_incr_dstr = '20210708' #'20200716' #'20200608'
rae.get_incremental_results(dstr=alice_incr_dstr)

##
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('stephen')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
# rae.create_incremental_dbs()
stephen_incr_dstr = '20200717' #'20200608' # '20200414'
rae.run_incremental_training(dstr=stephen_incr_dstr) #run_type = 'submit'

##
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('stephen')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
stephen_incr_dstr = '20200717' #'20200608' # '20200414'
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
dstr = '20200710'
for britnum in range(3):
    rae.setup('brit{}'.format(britnum))
    rae.all_models = [m for m in rae.all_models if 'orig' not in m]

    # rae.cv_train_britton() # use skip_db=False, run_type='submit' to actually rerun it
    # rae.cv_train_from_mat(queue='gpu_tesla',dstr=dstr)
    rae.cv_train_from_mat(dstr=dstr)

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
dstr = '20200710'
for britnum in range(3):
    rae.setup('brit{}'.format(britnum))
    rae.all_models = [m for m in rae.all_models if 'orig' not in m]

    # rae.get_cv_results(queue='gpu_tesla',dstr='20200417',db_from_mdn_dir=True)
    rae.get_cv_results(queue='gpu_rtx',dstr=dstr,db_from_mdn_dir=True)


## Romains experiments

## CV Training - all views

import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('romain','')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
dstr = '20200912'
rae.cv_train_from_mat(dstr=dstr,queue='gpu_tesla') # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('romain')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
dstr = '20200912'
rae.get_cv_results(dstr=dstr,queue='gpu_tesla')



## Roain's expts
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
dstr = '20200804' # '20200712'
rae.setup('roian','')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
# rae.cv_train_from_mat(dstr=dstr) # skip_db=False,run_type='submit'
rae.cv_train_from_mat(dstr=dstr,queue='gpu_tesla') # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('roian')
dstr = '20200804' #'20200712'
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
# rae.get_cv_results(queue='gpu_rtx',dstr=dstr,db_from_mdn_dir=True)
rae.get_cv_results(queue='gpu_tesla',dstr=dstr,db_from_mdn_dir=True)

## Larva
from importlib import reload
import run_apt_expts_2 as rae
reload(rae)
rae.setup('larva','')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]

dstr =  '20200804' #'20200714' # '20200428'
rae.cv_train_from_mat(dstr=dstr,queue='gpu_tesla_large') # skip_db=False,run_type='submit'
# rae.cv_train_from_mat(dstr=dstr,queue='gpu_tesla_large') # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('larva')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
# rae.get_cv_results(dstr='20200428',db_from_mdn_dir=True) # skip_db=False,run_type='submit'
dstr = '20200804' #'20200428'
rae.get_cv_results(dstr=dstr,queue='gpu_tesla_large',db_from_mdn_dir=True) #

## LEAP
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('leap_fly')
rae.create_normal_dbs()
rae.create_gt_db()

##
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('leap_fly')
dstr  = '20200824'
rae.run_normal_training(dstr=dstr)

##
import run_apt_expts_2 as rae
# reload(rae)
rae.setup('leap_fly')
dstr  = '20200824'
rae.get_normal_results(dstr=dstr,db_from_mdn_dir=True)


## Single animal vs multiple animal for Stephen

import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('stephen')
dstr = '20200914'
rae.create_run_individual_animal_dbs_stephen(dstr=dstr)

## Results

import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('stephen')
dstr = '20200914'
rae.get_individual_animal_results_stephen(dstr=dstr)

##
# Run /groups/branson/bransonlab/mayank/APT_develop/ScriptStephenSingleAnimalResults.m
out_file = '/groups/branson/home/kabram/temp/stephen_single_fly_results.mat'
from scipy import io as sio
import multiResData
import matplotlib
matplotlib.use('TkAgg')
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
    im, locs, info, occ = multiResData.read_and_decode_without_session(db[view], 5)
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

eres03gt = ade.assess('e00_r03', dset='alice', usegt_tfr=True)



# SH
ade.assess_set2(exps, range(2), 'stephen', usegt_tfr=True,  )


## DPK/bub ia vs pt
import run_apt_expts_2 as rae
dtypes = ['alice']
for dd in dtypes:
    reload(rae)
    rae.setup(dd)
    rae.train_dpk_orig()

## APT- vs DPK-style train
rae.train_dpk_orig(expname='e00_r06',
                   run_type='submit',
                   exp_note='r00 split; dpk_train_style=apt',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt'
                   )
rae.train_dpk_orig(expname='e01_r00',
                   run_type='submit',
                   exp_note='full trainset; apt-style train; no val',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0
                   )

## APT(new/final) vs DPK, Aug16
rae.train_dpk_orig(expname='e04_r00',
                   run_type='submit',
                   exp_note='r00 split; dpk_train_style=apt (new; step_loss, bsize=8 etc)',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   )
rae.train_dpk_orig(expname='e04_r01',
                   run_type='submit',
                   exp_note='r00 split; dpk_train_style=apt (new; step_loss, bsize=8 etc). no val just as sanity check',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0,
                   )

## APT(new/final) vs DPK, Aug16, Romn
rae.train_dpk_orig(expname='cmp_r00',
                   run_type='submit',
                   exp_note='dpk_train_style=apt (new; step_loss etc). no val',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0,
                   )
rae.train_dpk_orig(expname='cmp_r01',
                   run_type='submit',
                   exp_note='dpk_train_style=dpk',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpk',
                   dpk_val_batch_size=11,
                   )
rae.train_dpk_orig(expname='cmp_r02',
                   run_type='submit',
                   exp_note='dpk_train_style=dpktrnonly. probing this as seeing diffs between dpk/cheat and aptnew',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpktrnonly',
                   dpk_val_batch_size=0,
                   )




## DPK, no val
rae.train_dpk_orig(expname='e02_r00',
                   run_type='submit',
                   exp_note='r02 split; dpktrnonly-style train (no val)',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpktrnonly',
                   )

rae.train_dpk_orig(expname='e04_r00',
                   run_type='submit',
                   exp_note='round 2 of: try to repro previous sh results (feb2020) which were ~1px better than this round. dpktrainonly-style train, no val; smaller bsize, no pretrained?! round 2, also use previous stripped lbl which had different dataaug. round1 diverged during training, both views...',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpktrnonly',
                   dpk_val_batch_size=0,
                   batch_size=4,
                   dpk_use_pretrained=0,
                   brange="\\(-0.1,0.1\\)",
                   trange=20,
                   normalize_img_mean=0,
                   rrange=10,
                   crange="\\(0.9,1.1\\)"
                   )

rae.train_dpk_orig(expname='e05_r00',
                   run_type='submit',
                   exp_note='figured out (prob) why we are not reproing. rerun e00_r00 split, forcing display_steps (stepsperepoch) to 50. dpk (cheat) trnstyle; bsize=4; ptw',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpk',
                   dpk_val_batch_size=10,
                   batch_size=4,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0,
                   dpk_early_stop_style='\\"ipynb\\"'
                   )

rae.train_dpk_orig(expname='e08_r00',
                   run_type='submit',
                   exp_note='like e05 but wout ptw',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpk',
                   dpk_val_batch_size=10,
                   batch_size=4,
                   dpk_use_pretrained=0,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0,
                   dpk_early_stop_style='\\"ipynb\\"'
                   )

rae.train_dpk_orig(expname='e09_r00',
                   run_type='submit',
                   exp_note='like e05 but with 1/2 the base lr; apt-style we reduce the base lr when reducing bsize',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpk',
                   dpk_val_batch_size=10,
                   batch_size=4,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0,
                   dpk_early_stop_style='\\"ipynb\\"',
                   dpk_base_lr_used=.0005
                   )

# roian
rae.train_dpk_orig(expname='cv_s00_r00',
                   run_type='submit',
                   exp_note='roian first try',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0, #dpk_early_stop_style='\\"ipynb2\\"'
                   )

rae.train_dpk_orig(expname='cv_s00_r01',
                   run_type='submit',
                   exp_note='roian first try',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpktrnonly',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1, #                   display_step=50,
                   dpk_auto_steps_per_epoch=1, #dpk_early_stop_style='\\"ipynb2\\"'
                   )

rae.train_dpk_orig(expname='cv_s00_r02',
                   run_type='submit',
                   exp_note='roian first try. dpk-cheat, hand-selected valbsize gah',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpk',
                   dpk_val_batch_size=12,
                   dpk_use_pretrained=1, #                   display_step=50,
                   dpk_auto_steps_per_epoch=1, #dpk_early_stop_style='\\"ipynb2\\"'
                   )

rae.train_dpk_orig(expname='cv_s00_r03',
                   run_type='submit',
                   exp_note='trying half base lr',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpktrnonly',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1, #                   display_step=50,
                   dpk_auto_steps_per_epoch=1, #dpk_early_stop_style='\\"ipynb2\\"'
                   dpk_base_lr_used=.001/2,
                   )

# larva

rae.train_dpk_orig(expname='cv_s00_r00',
                   run_type='submit',
                   exp_note='larv first try. apt-style',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0, #                   dpk_early_stop_style='\\"ipynb2\\"'
                   )


rae.train_dpk_orig(expname='cv_s00_r01',
                   run_type='submit',
                   exp_note='larv first try',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpktrnonly',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0, #dpk_early_stop_style='\\"ipynb2\\"'
                   )

rae.train_dpk_orig(expname='cv_s00_r02',
                   run_type='submit',
                   exp_note='larv first try. dpk-cheat, hand-selected valbsize gah',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpk',
                   dpk_val_batch_size=9,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0, #dpk_early_stop_style='\\"ipynb2\\"'
                   )

rae.train_dpk_orig(expname='cv_s00_r03',
                   run_type='submit',
                   exp_note='larv first try',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0, #                   dpk_early_stop_style='\\"ipynb2\\"'
                   dl_steps=200000,
                   batch_size=4,
                   )

# bs1
rae.train_dpk_orig(expname='cv_s00_r00',
                   run_type='submit',
                   exp_note='bs first try. apt-style',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0, #                   dpk_early_stop_style='\\"ipynb2\\"'
                   batch_size=8,
                   dl_steps=50000,
                   )

rae.train_dpk_orig(expname='cv_s00_r01',
                   run_type='submit',
                   exp_note='bs first try. apt-style',
                   dpk_use_augmenter=0,
                   dpk_train_style='apt',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1,
                   display_step=50,
                   dpk_auto_steps_per_epoch=0, #                   dpk_early_stop_style='\\"ipynb2\\"'
                   batch_size=4,
                   dl_steps=200000,
                   )

rae.train_dpk_orig(expname='cv_s00_r02',
                   run_type='submit',
                   exp_note='bsfirst try',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpktrnonly',
                   dpk_val_batch_size=0,
                   dpk_use_pretrained=1,
                   display_step=100,
                   dpk_auto_steps_per_epoch=0, #dpk_early_stop_style='\\"ipynb2\\"'
                   batch_size=8,
                   dl_steps=100000,
                   )

rae.train_dpk_orig(expname='cv_s00_r03',
                   run_type='submit',
                   exp_note='bs first try, dpk cheat',
                   dpk_use_augmenter=0,
                   dpk_train_style='dpk',
                   dpk_val_batch_size=10,
                   dpk_use_pretrained=1,
                   display_step=100,
                   dpk_auto_steps_per_epoch=0, #dpk_early_stop_style='\\"ipynb2\\"'
                   batch_size=8,
                   dl_steps=100000,
                   )


## FINAL TRAIN AUG 2020
rae.setup('alice')
rae.all_models = ['dpk']
#rae.create_normal_dbs()
dstr = '20200818'
rae.run_normal_training(dstr=dstr,run_type='submit') #run_type = 'submit' to actually submit jobs.
# rae.run_normal_training(dstr=dstr,queue='gpu_tesla') #run_type = 'submit' to actually submit jobs.

import run_apt_expts_2 as rae
rae.setup('alice')
rae.all_models = ['dpk']
#rae.create_normal_dbs()
dstr = '20200818'
rae.run_incremental_training(dstr=dstr,run_type='dry') #run_type = 'submit' to actually submit jobs.
# rae.run_normal_training(dstr=dstr,queue='gpu_tesla') #run_type = 'submit' to actually submit jobs.

# ar by time
clist, tslist = ade.get_all_ckpt_h5(opj(exps.bub.root,'apt_expt_r01'))
for cpt in clist:
    ade.assess('apt_expt_r01',
               dset='alice',
               view=0,
               usegt_tfr=True,
               doplot=False,
               ckpt=os.path.basename(cpt),
               )

edir = opj(exps.bub.root,'apt_expt_r01')
res = ade.get_assess_results(edir)
resf = opj(exps.bub.root, 'ar_vw0_bytime_20200901.mat')
apt.savemat_with_catch_and_pickle(resf, res)



rae.setup('stephen')
rae.all_models = ['dpk']
# dstr = '20200605'
dstr = '20200818'
#rae.create_normal_dbs()
rae.run_normal_training(dstr=dstr) #run_type = 'submit'
# rae.run_normal_training(queue='gpu_tesla',dstr='20200411')

rae.setup('stephen')
rae.all_models = ['dpk']
dstr = '20200828'
rae.run_normal_training(expname='apt_expt_skel2', dstr=dstr, run_type='submit',exp_note='Trying "B" skels jff')
# rae.run_normal_training(queue='gpu_tesla',dstr='20200411')


rae.setup('stephen')
rae.all_models = ['dpk']
dstr = '20200901'
rae.run_normal_training(expname='apt_expt_r02',
                        dstr=dstr,
                        run_type='submit',
                        exp_note='Relative to apt_expt: i) "B" skels (more consistent with our other skels and with dpk ppr); ii) optimized keras/tfdata input pipeline')
# rae.run_normal_training(queue='gpu_tesla',dstr='20200411')

# vw0 by time
clist, tslist = ade.get_all_ckpt_h5(opj(exps.sh0.root,'apt_expt_r02'))
for cpt in clist:
    ade.assess('apt_expt_r02',
               dset='stephen',
               view=0,
               usegt_tfr=True,
               doplot=False,
               ckpt=os.path.basename(cpt),
               )

edir = opj(exps.sh0.root, 'apt_expt_r02')
res = ade.get_assess_results(edir)
resf = opj(exps.sh0.root, 'sh_vw0_time_20200901.mat')
apt.savemat_with_catch_and_pickle(resf, res)

# vw1 by time
clist, tslist = ade.get_all_ckpt_h5(opj(exps.sh1.root,'apt_expt_r02'))
for cpt in clist:
    ade.assess('apt_expt_r02',
               dset='stephen',
               view=1,
               usegt_tfr=True,
               doplot=False,
               ckpt=os.path.basename(cpt),
               )

edir = opj(exps.sh1.root, 'apt_expt_r02')
res = ade.get_assess_results(edir)
resf = opj(exps.sh1.root, 'sh_vw1_time_20200901.mat')
apt.savemat_with_catch_and_pickle(resf, res)





rae.setup('stephen')
rae.all_models = ['dpk']
#rae.create_normal_dbs()
dstr = '20200818'
rae.run_incremental_training(dstr=dstr,run_type='submit') #run_type = 'submit' to actually submit jobs.
# rae.run_normal_training(dstr=dstr,queue='gpu_tesla') #run_type = 'submit' to actually submit jobs.

rae.setup('roian')
rae.all_models = ['dpk']
dstr = '20200828'
rae.cv_train_from_mat(run_type='submit', dstr=dstr, queue='gpu_tesla') # skip_db=False,run_type='submit'

rae.setup('roian')
rae.all_models = ['dpk']
dstr = '20200831'
rae.cv_train_from_mat(
    run_type='submit',
    exp_name_pfix='r01_',
    split_idxs=[1,2,3,4],
    dstr=dstr,
    queue='gpu_tesla') # skip_db=False,run_type='submit'

clist, tslist = ade.get_all_ckpt_h5(opj(exps.ron.root,'r01_cv_split_0'))
ade.assess('r01_cv_split_0',
           dset='roian',
           view=0,
           usegt_tfr=False,
           doplot=False,
           tstbsize=12,
           ckpt=os.path.basename(clist[-1]),
           )

rae.setup('larva')
rae.all_models = ['dpk']
dstr = '20200903'
rae.cv_train_from_mat(
    run_type='submit',
    dstr=dstr,
    queue='gpu_tesla',
    split_idxs=[0,1,2],
    exp_note='real larva model after picking "best" parallelization',
    nslots=8,
    dpk_tfdata_num_para_calls_parse=5,
    dpk_tfdata_num_para_calls_dataaug=16)

# larva toy model, optimizing parallelization
rae.setup('larva')
rae.all_models = ['dpk']
dstr = '20200902'
rae.cv_train_from_mat(
    run_type='submit',
    exp_name_pfix='para05_',
    dstr=dstr,
    queue='gpu_tesla',
    split_idxs=[0,], # skip_db=False,run_type='submit'
    exp_note='real larva model titrate parn/slots',
    nslots=9,
    dpk_tfdata_num_para_calls_parse=5,
    dpk_tfdata_num_para_calls_dataaug=18,
)

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



## ======================================================
## =================== MUlti Animal =====================
## ======================================================


## Roian

## Add neg ROIs for mask experiments
from importlib import reload
import run_apt_ma_expts as rae_ma
reload(rae_ma)

robj = rae_ma.ma_expt('roian')
robj.add_neg_roi_roian()

## Run training
from importlib import reload
import run_apt_ma_expts as rae_ma
reload(rae_ma)

robj = rae_ma.ma_expt('roian')
robj.run_train(run_type='dry')

## view training images

import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('roian')
robj.show_samples()

##
import run_apt_ma_expts as rae_ma

robj = rae_ma.ma_expt('roian')
robj.get_status()


##
import run_apt_ma_expts as rae_ma

robj = rae_ma.ma_expt('roian')
robj.get_results()

## incremental training
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('roian')
robj.setup_incremental()

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('roian')
robj.run_incremental_train(t_types=[('grone','crop','mask')],queue='gpu_tesla')

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('roian')
robj.show_incremental_results(t_type=['grone','crop','mask','first'])

## ID Track movies
import run_apt_ma_expts as rae_ma
import PoseTools as pt
import os
robj = rae_ma.ma_expt('roian')

loc_file = os.path.join(robj.gt_dir, rae_ma.loc_file_str)
A = pt.json_load(loc_file)
gt_movies = A['movies']

t_types = [('2stageHT','crop','nomask'),
           ('2stageHT','nocrop','nomask'),
           ('2stageBBox','nomask'),
           ('grone','crop','mask'),
           ('grone','nocrop','mask')
           ]
run_type = 'dry'
for cur_mov in gt_movies[-1:]:
    exp_name = os.path.splitext(os.path.split(cur_mov)[1])[0]
    for tt in t_types:
        cur_str = '_'.join(tt)
        out_trk = os.path.join(robj.trk_dir,exp_name + f'_{cur_str}.trk')
        robj.track(cur_mov,out_trk,t_types=[tt,],run_type=run_type)


######################
## Alice

## Add neg ROIs for experiments
from importlib import reload
import run_apt_ma_expts as rae_ma
reload(rae_ma)

robj = rae_ma.ma_expt('alice')
# robj.get_neg_roi_alice(debug=True) # view the neg rois
robj.add_neg_roi_alice()

## Run training
from importlib import reload
import run_apt_ma_expts as rae_ma
reload(rae_ma)

robj = rae_ma.ma_expt('alice')
robj.run_train(run_type='dry')

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('alice')
robj.show_samples()

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('alice')
robj.get_status()


## incremental training
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('alice')
robj.setup_incremental()

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('alice')
robj.run_incremental_train(t_types=[('grone','crop')])

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('alice')
robj.get_incremental_results(t_types=[('grone','crop')])

## Track movies
import os
import run_apt_ma_expts as rae_ma
import PoseTools as pt
robj = rae_ma.ma_expt('alice')

loc_file = os.path.join(robj.gt_dir, rae_ma.loc_file_str)
A = pt.json_load(loc_file)
gt_movies = A['movies']

run_type = 'dry'
for cur_mov in gt_movies:
    exp_name = os.path.split(os.path.split(cur_mov)[0])[1]
    out_trk = os.path.join(robj.trk_dir,exp_name + '_grone.trk')
    robj.track(cur_mov,out_trk,t_types=[('grone','crop')],run_type=run_type)


