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
dstr = '20231207' #'20210708' #'20210629'
# rae.create_normal_dbs(expname='touching')
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
dstr = '20231207' #'20210708' #'20210629'
# rae.create_gt_db()
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
#dstr = '20200706'
dstr = '20231207'
# rae.create_normal_dbs()
rae.run_normal_training(dstr=dstr) #run_type = 'submit'

##
import run_apt_expts_2 as rae
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('stephen')
# dstr = '20200706' #'20200411'
dstr = '20231207'
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
dstr = '20231207'#'20210708' #'20210629'
import run_apt_ma_expts as rae_ma
import os
robj = rae_ma.ma_expt('alice')
ma_inc_file = os.path.join(robj.trnp_dir,'inc_data.pkl')
ma_loc = os.path.join(robj.trnp_dir, 'grone', rae_ma.loc_file_str)

rae.create_incremental_dbs_ma(ma_inc_file, ma_loc)
rae.run_incremental_training(dstr=dstr) #run_type = 'submit'


##
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('alice')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
alice_incr_dstr = '20231207' #'20210708' #'20200716' #'20200608'
rae.get_incremental_results(dstr=alice_incr_dstr)

##
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('stephen')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
# rae.create_incremental_dbs()
stephen_incr_dstr = '20231207' #'20200717' #'20200608' # '20200414'
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
for data_type in ['brit0','brit1','brit2','romain','larva']:
    reload(rae)
    rae.setup(data_type)
    # rae.create_normal_dbs()
    rae.run_normal_training() #run_type = 'submit') # to actually submit jobs.


## Brits experiments

## training
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
dstr = '20231207' #'20200710'
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
dstr = '20231207' # '20200710'
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
dstr = '20231207' # '20200912'
rae.cv_train_from_mat(dstr=dstr) # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('romain')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
rae.all_models = [m for m in rae.all_models if 'hrformer' not in m]

dstr = '20231207' #'20200912'
rae.get_cv_results(dstr=dstr,db_from_mdn_dir=True)#,queue='gpu_tesla')



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

dstr = '20231207' # '20200804' #'20200714' # '20200428'
rae.cv_train_from_mat(dstr=dstr) # skip_db=False,run_type='submit'
# rae.cv_train_from_mat(dstr=dstr,queue='gpu_tesla_large') # skip_db=False,run_type='submit'

## results
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('larva')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
rae.all_models = [m for m in rae.all_models if 'hrformer' not in m]
# rae.get_cv_results(dstr='20200428',db_from_mdn_dir=True) # skip_db=False,run_type='submit'
dstr = '20231207' #'20200804' #'20200428'
rae.get_cv_results(dstr=dstr,queue='gpu_rtx8000',db_from_mdn_dir=True) #

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
dstr = '20231207' #'20210708' #'20210629'
rae.get_normal_results(dstr=dstr,exp_name='touching',db_from_mdn_dir=True,last_model_only=True)



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
rae_ma.alg_names = rae_ma.alg_names[:-3]
robj = rae_ma.ma_expt('roian')
robj.add_neg_roi_roian()

## create gt db
from importlib import reload
import run_apt_ma_expts as rae_ma
reload(rae_ma)
robj = rae_ma.ma_expt('roian')
robj.create_gt_db()



## Run training
from importlib import reload
import run_apt_ma_expts as rae_ma
reload(rae_ma)

robj = rae_ma.ma_expt('roian')
robj.run_train(run_type='dry')

## Add more margin to bbox for CID
# maynot be needed anymore because using grone loc.json 16 APril 2024
import os,json,shutil
import PoseTools as pt
in_file = '/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/multi_cid/view_0/cid_crop_nomask_07122023/train_TF.json'
if not os.path.exists(in_file+'.bak'):
    shutil.copy(in_file,in_file+'.bak')
    J = pt.json_load(in_file)
    for ann in J['annotations']:
        ann['bbox'] = [ann['bbox'][0]-10,ann['bbox'][1]-10,ann['bbox'][2]+20,ann['bbox'][3]+20]
    with open(in_file,'w') as f:
        json.dump(J,f)
robj = rae_ma.ma_expt('roian')
robj.run_train(run_type='dry',t_types=[('cid','crop','nomask')])

##
import os,json,shutil
import PoseTools as pt
from importlib import reload
import run_apt_ma_expts as rae_ma
reload(rae_ma)

in_file = '/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/multi_cid/view_0/cid_nocrop_nomask_07122023/train_TF.json'
if not os.path.exists(in_file+'.bak'):
    shutil.copy(in_file,in_file+'.bak')
    J = pt.json_load(in_file)
    for ann in J['annotations']:
        ann['bbox'] = [ann['bbox'][0]-10,ann['bbox'][1]-10,ann['bbox'][2]+20,ann['bbox'][3]+20]
    with open(in_file,'w') as f:
        json.dump(J,f)
robj = rae_ma.ma_expt('roian')
robj.run_train(run_type='dry',t_types=[('cid','nocrop','nomask')])

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
robj.get_results(res_dstr='20240418')

##
import run_apt_ma_expts as rae_ma

robj = rae_ma.ma_expt('roian')
robj.show_results()


## incremental training
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('roian')
robj.setup_incremental()

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('roian')
robj.run_incremental_train(t_types=[('grone','crop','mask')],queue='gpu_rtx8000')

##
import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('roian')
robj.get_incremental_results(t_types=[['grone','crop','mask','first'],])


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
unmarked_movies = ['/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_0.mjpg','/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_1.mjpg','/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg',]

marked_movies_close = ['/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210902_m186741silpb_no_odor_m186370_ft181918.mjpg',
'/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210903_m186370silpb_no_odor_m186741_ft181918.mjpg',
'/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210906_m186741silpb_no_odor_m186370_f0181320.mjpg',
'/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210907_m186370silpb_no_odor_m186741_f0186560.mjpg']

# gt_movies = marked_movies_close

t_types = [('2stageHT','crop','nomask'),
           ('2stageHT','nocrop','nomask'),
           ('2stageBBox','nomask'),
           ('grone','crop','nomask'),
           ('grone','nocrop','nomask'),
           ('2stageBBox_hrformer','nomask'),
           ('cid','crop'),('cid','nocrop'),
           ('dekr','crop','nomask'),
           ('dekr','nocrop','nomask'),
           ]
if gt_movies[0] == unmarked_movies[0]:
    robj.params[()]=[{'max_n_animals':4},{}]
run_type = 'dry'
sing_img = '/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif'
# sing_img = '/groups/branson/home/kabram/bransonlab/singularity/mmpose_1x.sif'

for cur_mov in gt_movies[-4:]:
    exp_name = os.path.splitext(os.path.split(cur_mov)[1])[0]
    for tt in t_types:
        cur_str = '_'.join(tt)
        out_trk = os.path.join(robj.trk_dir,exp_name + f'_{cur_str}.trk')
        robj.track(cur_mov,out_trk,t_types=[tt,],run_type=run_type,sing_img=sing_img)

## for close movies find frames with large differences between different algos

import run_apt_ma_expts as rae_ma
import PoseTools as pt
import os
import TrkFile as Trk
from reuse import find_dist_match
robj = rae_ma.ma_expt('roian')

marked_movies_close = ['/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210902_m186741silpb_no_odor_m186370_ft181918.mjpg',
'/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210903_m186370silpb_no_odor_m186741_ft181918.mjpg',
'/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210906_m186741silpb_no_odor_m186370_f0181320.mjpg',
'/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_data/set_10/mjpg/20210907_m186370silpb_no_odor_m186741_f0186560.mjpg']

t_types = [('grone','crop','nomask'),
           ('2stageBBox_hrformer','nomask'),
           ('dekr','crop','nomask'),
           ]
all_d = []
for cur_mov in marked_movies_close:
    exp_name = os.path.splitext(os.path.split(cur_mov)[1])[0]
    all_t = []
    for tt in t_types:
        cur_str = '_'.join(tt)
        out_trk = os.path.join(robj.trk_dir,exp_name + f'_{cur_str}_tracklet.trk')
        curt = Trk.Trk(out_trk)
        all_t.append(curt)
    nfr = max(all_t[0].endframes)+1
    d_all = np.zeros((nfr,len(t_types),len(t_types),2,4))
    for ndx in range(nfr):
        curp = [tt.getframe(ndx)[:,:,0] for tt in all_t]
        curv = [~np.all(np.isnan(curp[xx][:,0,]),axis=0) for xx in range(len(curp))]
        curp = [curp[xx][:,:,curv[xx]] for xx in range(len(curp))]
        for xx in range(len(curp)):
            for yy in range(xx+1,len(curp)):
                if (curp[xx].shape[2] != 2) or (curp[yy].shape[2] != 2):
                    d_all[ndx,xx,yy] = np.nan
                    continue
                dd = np.linalg.norm(curp[xx][..., None] - curp[yy][:, :, None], axis=1)

                dd_m = find_dist_match(np.transpose(dd[None], [0, 3, 2, 1]))[0]
                d_all[ndx,xx,yy] = dd_m

    all_d.append(d_all)

## show few examples

import movies
movid = 2

cap = movies.Movie(marked_movies_close[movid])
sel = np.unique(np.where(~(np.max(all_d[movid],axis=(-1,-2))<50))[0])
cur_mov = marked_movies_close[movid]
exp_name = os.path.splitext(os.path.split(cur_mov)[1])[0]

all_t = []
for tt in t_types:
    cur_str = '_'.join(tt)
    out_trk = os.path.join(robj.trk_dir, exp_name + f'_{cur_str}_tracklet.trk')
    curt = Trk.Trk(out_trk)
    all_t.append(curt)

##
ndx = np.random.choice(sel)
plt.figure(73)
plt.clf()
fr = cap.get_frame(ndx)[0]
plt.imshow(fr,'gray')
skel = np.array([[0,1],[0,2],[2,3]])
ccs = ['r','g','b']

for xx in range(len(all_t)):
    curp = all_t[xx].getframe(ndx)
    curp = curp[:,:,0]
    curp = curp[:,:,~np.all(np.isnan(curp[:,0]),axis=0)]
    curp = np.transpose(curp,(2,0,1))
    mdskl(curp,skel,cc=ccs[xx])

all_d[movid][ndx].max(axis=(-1,-2))

## check consistency between different trackers

all_aa = []; all_bb = []; all_cc = []
all_sel = []
for movid in range(len(marked_movies_close)):
    jj = all_d[movid].copy()
    jj[np.isnan(jj)] = 3000
    sel = np.unique(np.where(~(np.max(all_d[movid],axis=(-1,-2))<50))[0])
    aa = np.minimum(jj[sel][:,0,1].max(axis=(-1,-2)),jj[sel][:,0,2].max(axis=(-1,-2)))
    bb = np.minimum(jj[sel][:,0,1].max(axis=(-1,-2)),jj[sel][:,1,2].max(axis=(-1,-2)))
    cc = np.minimum(jj[sel][:,0,2].max(axis=(-1,-2)),jj[sel][:,1,2].max(axis=(-1,-2)))
    all_aa.append(aa)
    all_bb.append(bb)
    all_cc.append(cc)
    all_sel.extend(list(zip([movid]*len(sel),sel)))
aa = np.concatenate(all_aa,0)
bb = np.concatenate(all_bb,0)
cc = np.concatenate(all_cc,0)
print(len(aa)/4/108000)
print(np.count_nonzero(aa>2500),np.count_nonzero(bb>2500),np.count_nonzero(cc>2500))
ff(); plt.hist([aa,bb,cc],list(range(0,200,30)))
ff(); plt.hist([aa,bb,cc])

## create list of frames to label

all_sel = []
thresh = 50
for movid in range(len(marked_movies_close)):
    jj = all_d[movid].copy()
    jj[np.isnan(jj)] = 3000
    sel = np.unique(np.where(~(np.max(all_d[movid],axis=(-1,-2))<thresh))[0])
    all_sel.extend(list(zip([movid+1]*len(sel),sel+1)))

n_sel = 300
sel = np.random.choice(len(all_sel),n_sel,replace=False)
sel = [all_sel[xx] for xx in sel]

out_file = '/groups/branson/bransonlab/mayank/apt_results/roian_ma_close_labels_suggestions_20240506.mat'
from scipy import io as sio
sio.savemat(out_file,{'sel':sel,'movies':marked_movies_close})


## show heatmaps for an example frame where mice are close
movid = 0
fr = 106839
net = 'dekr'
cstr = ''
if net == 'grone':
    net_name = 'multi_mdn_joint_torch'
elif net == 'dekr':
    net_name = 'multi_dekr'
    cstr = 'mmpose_net "dekr"'
elif net == 'cid':
    net_name = 'multi_cid'
    cstr = 'mmpose_net "cid"'
cmd = f'/groups/branson/home/kabram/temp/ma_expts/roian/trn_packdir_07122023/{net}/conf_crop.json -name {net}_crop_nomask_07122023 -json_trn_file /groups/branson/home/kabram/temp/ma_expts/roian/trn_packdir_07122023/grone/loc_neg.json -conf_params multi_loss_mask False link_id True link_id_training_iters 100000 {cstr} -cache /groups/branson/bransonlab/mayank/apt_cache_2  -type {net_name} track -mov {marked_movies_close[movid]} -out /groups/branson/home/kabram/temp/out.trk -start_frame {fr+1} -end_frame {fr+2}'

import APT_interface as apt
apt.main(cmd.split())

# for dekr, breakpoint at line 457 in /environments/apt-ultramodern-frozen-2023-11-10/lib/python3.10/site-packages/mmpose/models/heads/hybrid_heads/dekr_head.py

# for dekr, breakpoint at line 457 in /environments/apt-ultramodern-frozen-2023-11-10/lib/python3.10/site-packages/mmpose/models/heads/hybrid_heads/dekr_head.py
# for cid, breakpont at line 250 in /schtuff/pinned-ampere-env/lib/python3.10/site-packages/mmpose/models/detectors/cid.py
# dekr output in /groups/branson/home/kabram/temp/roian_dekr_hmap_close_0_47628.pkl and /groups/branson/home/kabram/temp/roian_dekr_hmap_close_0_47628.mat

## show results of different trackers on an overlapping mice frame
import PoseTools as pt
movid = 0
fr = 106839

import movies

cap = movies.Movie(marked_movies_close[movid])
cur_mov = marked_movies_close[movid]
exp_name = os.path.splitext(os.path.split(cur_mov)[1])[0]

t_types = [('grone','crop','nomask'),
           ('2stageBBox_hrformer','nomask'),
           ('dekr','crop','nomask'),
           ('cid','crop'),
           ('2stageBBox','nomask')
           ]

all_t = []
for tt in t_types:
    cur_str = '_'.join(tt)
    out_trk = os.path.join(robj.trk_dir, exp_name + f'_{cur_str}_tracklet.trk')
    curt = Trk.Trk(out_trk)
    all_t.append(curt)

ndx = fr
fr1 = cap.get_frame(ndx)[0]
f,ax = plt.subplots(2,3,figsize=(15,10))
ax = ax.flatten()
skel = np.array([[0,1],[0,2],[2,3]])
nets = ['GRONe','DeTR+HRFormer','DeKR','CiD','DeTR+GRONe']
xlim = [1400,1800]
ylim = [500,100]

for xx in range(len(all_t)):
    curp = all_t[xx].getframe(ndx)
    curp = curp[:,:,0]
    curp = curp[:,:,~np.all(np.isnan(curp[:,0]),axis=0)]
    curp = np.transpose(curp,(2,0,1))
    ax[xx].imshow(fr1,'gray')
    mdskl(curp,skel,cc='r',ax=ax[xx])
    ax[xx].set_title(nets[xx])
    ax[xx].axis('off')
    ax[xx].set_xlim(xlim)
    ax[xx].set_ylim(ylim)


f.tight_layout()
f.savefig(f'/groups/branson/home/kabram/temp/roian_close_mice_{movid}_{fr}.png')

## show dekr heatmaps

zz = pt.pickle_load(f'/groups/branson/home/kabram/temp/roian_dekr_hmap_close_{movid}_{fr}.pkl')
hh = np.array([cv2.resize(hhx,zz['img'].shape) for hhx in zz['hmap']])
ff(); imshow(zz['img'][ylim[1]:ylim[0],xlim[0]:xlim[1]],'gray')
imshow(hh.sum(axis=0)[ylim[1]:ylim[0],xlim[0]:xlim[1]],alpha=0.2)
plt.axis('off')
plt.tight_layout()
plt.savefig(f'/groups/branson/home/kabram/temp/roian_dekr_hmap_close_{movid}_{fr}.png')

pt.show_stack(hh[:,ylim[1]:ylim[0],xlim[0]:xlim[1]],2,2)
plt.savefig(f'/groups/branson/home/kabram/temp/roian_dekr_hmap_close_{movid}_{fr}_stack.png')


## show cid heatmaps
zz = pt.pickle_load(f'/groups/branson/home/kabram/temp/roian_cid_hmap_close_{movid}_{fr}.pkl')
hh = np.array([cv2.resize(hhx,zz['img'].shape) for hhx in zz['hmap']])
ff(); imshow(zz['img'][ylim[1]:ylim[0],xlim[0]:xlim[1]],'gray')
imshow(hh.sum(axis=0)[ylim[1]:ylim[0],xlim[0]:xlim[1]],alpha=0.2)
plt.axis('off')
plt.tight_layout()
plt.savefig(f'/groups/branson/home/kabram/temp/roian_cid_hmap_close_{movid}_{fr}.png')

pt.show_stack(hh[:,ylim[1]:ylim[0],xlim[0]:xlim[1]],2,2)
plt.savefig(f'/groups/branson/home/kabram/temp/roian_cid_hmap_close_{movid}_{fr}_stack.png')


## unmarked mice tracking stuff in script_inc_exp.py

##
import TrkFile as trk
# tfile = '/groups/branson/home/kabram/temp/ma_expts/roian/trks/20210924_four_female_mice_again_grone_crop_mask_tracklet.trk'
tfile = '/groups/branson/home/kabram/temp/ma_expts/roian/trks/20210924_four_female_mice_0_grone_crop_mask_tracklet.trk'
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch

tt = trk.Trk(tfile)
nfr = max(tt.nframes)
ov = np.zeros(nfr)
ndets = np.zeros(nfr)
for ndx in range(nfr):
    curp = tt.getframe(ndx)
    vv = ~np.all(np.isnan(curp[:,0,0,:]),axis=0)
    ndets[ndx] = np.sum(vv)
    if vv.sum()<2:
        continue
    curp = curp[:,:,0,vv]
    bb =[]
    for xx in range(curp.shape[2]):
        bb.append(np.min(curp[...,xx],axis=0).tolist() + (np.max(curp[...,xx],axis=0)).tolist())
    overlaps = bbox_overlaps(torch.Tensor(bb),torch.Tensor(bb))
    overlaps = overlaps.numpy().flatten()
    overlaps[::vv.sum()+1] = 0
    ov[ndx] =  overlaps.max()

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


##
import run_apt_ma_expts as rae_ma

robj = rae_ma.ma_expt('alice')
robj.get_results(res_dstr='20231226')

##
import run_apt_ma_expts as rae_ma

robj = rae_ma.ma_expt('alice')
robj.show_results()


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

## create list of unique movies
import os
import run_apt_ma_expts as rae_ma
import PoseTools as pt
robj = rae_ma.ma_expt('alice')

loc_file = os.path.join(robj.gt_dir, rae_ma.loc_file_str)
A = pt.json_load(loc_file)
movies = A['movies']

social_movies = [
    '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531//movie.ufmf',
    '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629//movie.ufmf',
    '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigC_20201216T164812/movie.ufmf',
    '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigC_20201216T155818/movie.ufmf',
    '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigD_20201216T155952/movie.ufmf',
    '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigD_20201216T175902/movie.ufmf',
    '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/socialCsChr_JHS_BDPAD_BDPDBD_CsChrimson_RigC_20190910T152823/movie.ufmf',
    '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA71G01_Unknown_RigC_20201216T153727/movie.ufmf',
    '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/socialCsChr_JHS_BDPAD_BDPDBD_CsChrimson_RigA_20190910T152328/movie.ufmf',
]

movies = A['movies']+social_movies
nm = []
all_e = []
for cur_mov in movies:
    exp_name = os.path.split(os.path.split(cur_mov)[0])[1]
    if exp_name in all_e:
        continue
    all_e.append(exp_name)
    nm.append(cur_mov)

movies = nm

## Run tracking
# to do id tracking at scale 2, need to make changes in run_apt_ma_expts.py

run_type = 'dry'
tts = [('grone','crop'),('2stageBBox','first'),('2stageBBox_hrformer','first')]
for cur_mov in movies:
    exp_name = os.path.split(os.path.split(cur_mov)[0])[1]
    for curt in tts:
        out_trk = os.path.join(robj.trk_dir,exp_name + f'_{curt[0]}_scale2.trk')
        robj.track(cur_mov,out_trk,t_types=[curt,],run_type=run_type)

## Tracking without hard mining
hm_movies = ['/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629//movie.ufmf','/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531/movie.ufmf','/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf'] # movies that have GT
run_type = 'dry'
tts = [('grone','crop'),('2stageBBox_hrformer','first')]
robj.params[()][0]['link_id_mining_steps']='1'
for cur_mov in hm_movies: #movies[:3]:
    exp_name = os.path.split(os.path.split(cur_mov)[0])[1]
    for curt in tts:
        out_trk = os.path.join(robj.trk_dir,exp_name + f'_{curt[0]}_nohardmine.trk')
        robj.track(cur_mov,out_trk,t_types=[curt,],run_type=run_type)

## ID tracking accuracy

import TrkFile as trkf
tts = [('grone','crop'),('2stageBBox_hrformer','first')]
acc = []
missing = []
for cur_mov in movies:
    exp_name = os.path.split(os.path.split(cur_mov)[0])[1]
    ac = []
    for curt in tts:
        for sc in range(2):
            if sc == 0:
                out_trk = os.path.join(robj.trk_dir,exp_name + f'_{curt[0]}.trk')
            else:
                out_trk = os.path.join(robj.trk_dir,exp_name + f'_{curt[0]}_scale2.trk')
            if not os.path.exists(out_trk):
                missing.append([cur_mov,curt,sc])
                continue
            trk = trkf.Trk(out_trk)
            counts = []
            nfr = max(trk.nframes)
            tlen = np.zeros((nfr))
            clen = []
            for xx in range(trk.ntargets):
                jj = trk.gettarget(xx)[0, 0, :, 0]
                tlen[~np.isnan(jj)] += 1
                cx = ~np.isnan(jj)
                clen.append(cx)
                counts.append(np.count_nonzero(cx))

            ixx = np.argsort(counts)[::-1]
            ilen = np.zeros((nfr,))

            n_animals = round(np.median(tlen))
            for nx in range(n_animals):
                xx = ixx[nx]
                ilen += clen[xx]

            coverage = np.count_nonzero(ilen==tlen)/nfr

            counts= np.array(counts)
            avg_len = np.sum(counts)/n_animals
            ac.append([curt[0],coverage,n_animals,np.sum(counts)/n_animals/nfr,sc])
    acc.append([exp_name,ac])

print(acc)

## ID accuracy using GT data

# Using https://github.com/cheind/py-motmetrics#References
# prev matlab code in /groups/branson/bransonlab/mayank/APT/matlab/script_compare_fixerror_id.m

info = []
info.append({})

# Alice data, my GT
info[0]['mov_file'] = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf'
# info[0]['ctrax'] = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/registered_trx.mat';
info[0]['ctrax'] = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie/movie_JAABA/trx.mat'
#id_trk = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_bbox.trk'
# id_trk = '/groups/branson/home/kabram/temp/ar_id.trk'
info[0]['id_trk'] = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_grone.trk'

# info[0]['fix_error_trx'] = '/groups/branson/home/kabram/temp/fixed_id_trx_break.mat'
info[0]['fix_error_trx'] = '/groups/branson/bransonlab/mayank/apt_results/cx_GMR_SS00030_CsChr_RigC_20150826T144616_fix_error_GT.mat'
info[0]['ht_pts'] = [0,6]

# Alice data -- social flies kb's gt

info.append({})
# id_trk = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigA_20201212T163531_grone.trk';
# id_trk = '/groups/branson/home/kabram/temp/ar_social_id_grone_70imsz.trk';
info[-1]['id_trk'] = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigA_20201212T163531_grone.trk';
info[-1]['mov_file'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531//movie.ufmf';
info[-1]['fix_error_trx'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531/fixedtrx_20230117T174500.mat';
info[-1]['ht_pts'] = [0,6]
info[-1]['ctrax'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531/registered_trx.mat'

# Alice data -- social flies kb's gt --1

info.append({})
#id_trk = '/groups/branson/home/kabram/temp/ar_flytracker2_idlinked.trk';
info[-1]['id_trk'] = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigB_20201212T163629_grone.trk';
info[-1]['mov_file'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629//movie.ufmf'
info[-1]['fix_error_trx'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629/fixed_trx.mat'
info[-1]['ht_pts'] = [0,6]
info[-1]['ctrax'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629/registered_trx.mat'

# roian unmarked mice
info.append({})
info[-1]['mov_file'] = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_0.mjpg';
info[-1]['id_trk'] = '/groups/branson/home/kabram/temp/20210924_four_female_mice_0_unlabeled_mice_grone_occluded_MA_Bottom_Up_motion_link.trk'

# info[-1]['fix_error_trx'] = '/groups/branson/home/kabram/temp/fixed_roian_id_trx_break.mat'
info[-1]['fix_error_trx'] = '/groups/branson/bransonlab/mayank/apt_results/20210924_four_female_mice_0_fix_error_GT.mat'
info[-1]['ht_pts'] = [0,1]

##
import motmetrics as mm
import numpy as np
import APT_interface as apt
import TrkFile as trkf
from poseConfig import conf
conf.has_trx_file = True
import os


def get_trx_frame(trx,fr):
    # pts is n_trx x 2 x 2, where second dim has the head and tail points
    valid_trx = np.where( (trx['first_frames']<=fr) & (trx['end_frames']>fr) )[0]
    pts = np.ones([trx['n_trx'],2,2])*np.nan
    for vv in range(trx['n_trx']):
        if vv in valid_trx:
            fro = fr-int(trx['trx'][vv]['firstframe'][0,0])+1
            xx = trx['trx'][vv]['x'].flatten()[fro]-1
            yy = trx['trx'][vv]['y'].flatten()[fro]-1
            theta = trx['trx'][vv]['theta'].flatten()[fro]
            a = trx['trx'][vv]['a'].flatten()[fro]

            hh = np.array([xx+a*np.cos(theta)*2,yy+a*np.sin(theta)*2])
            tt = np.array([xx-a*np.cos(theta)*2,yy-a*np.sin(theta)*2])

            pts[vv,0,:] = hh
            pts[vv,1,:] = tt
    return pts

res = []
other_res = ['grone_scale2','2stageBBox','2stageBBox_scale2']
for cur_info in info:
    cur_res = {}
    acc = mm.MOTAccumulator(auto_id=True)
    t2 = apt.get_trx_info(cur_info['fix_error_trx'],conf,None)
    t1 = trkf.Trk(cur_info['id_trk'])
    nfr = max(t1.nframes)
    ht = cur_info['ht_pts']
    for fr in range(nfr):
        f1 = t1.getframe(fr)
        vf1 = ~np.isnan(f1[0,0,0,:])
        vix = np.where(vf1)[0]
        f1 = f1[...,vf1]
        f1 = np.transpose(f1[ht,:,0],[2,0,1])
        f2 = get_trx_frame(t2,fr)

        d_mat = np.linalg.norm(f1[None,:]-f2[:,None,:,:],axis=-1).mean(axis=-1)

        acc.update(range(f2.shape[0]),vix,d_mat)

    mh = mm.metrics.create()
    summ = mh.compute(acc,metrics=mm.metrics.motchallenge_metrics,return_dataframe=False)
    # print('-------------------')
    # print(cur_info['mov_file'])
    # print(summ)
    # print()
    cur_res['grone'] = summ

    for tname in other_res:
        nname = cur_info['id_trk'].replace('grone.trk',tname+'.trk')
        if not os.path.exists(nname):
            continue
        acc = mm.MOTAccumulator(auto_id=True)
        t1 = trkf.Trk(nname)
        for fr in range(nfr):
            f1 = t1.getframe(fr)
            vf1 = ~np.isnan(f1[0,0,0,:])
            vix = np.where(vf1)[0]
            f1 = f1[...,vf1]
            f1 = np.transpose(f1[ht,:,0],[2,0,1])
            f2 = get_trx_frame(t2,fr)

            d_mat = np.linalg.norm(f1[None,:]-f2[:,None,:,:],axis=-1).mean(axis=-1)

            acc.update(range(f2.shape[0]),vix,d_mat)

        mh = mm.metrics.create()
        summ = mh.compute(acc,metrics=mm.metrics.motchallenge_metrics,return_dataframe=False)
        cur_res[tname] = summ


    if 'ctrax' not in cur_info:
        res.append([ename, cur_res])
        continue

    t1 = apt.get_trx_info(cur_info['ctrax'],conf,None)
    acc = mm.MOTAccumulator(auto_id=True)
    for fr in range(nfr):
        f1 = get_trx_frame(t1, fr)
        f2 = get_trx_frame(t2, fr)

        d_mat = np.linalg.norm(f1[None,] - f2[:,None, :], axis=-1).mean(axis=-1)

        acc.update(range(f2.shape[0]),range(f1.shape[0]),  d_mat)

    mh = mm.metrics.create()
    summ = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, return_dataframe=False)
    cur_res['ctrax'] = summ
    # print('CTRAX-------------------')
    # print(summ)
    # print()
    ename = os.path.split(os.path.split(cur_info['mov_file'])[0])[1]
    res.append([ename,cur_res])

## print the results
for rr in res:
    print(rr[0])
    for kk,vv in rr[1].items():
        print(kk,vv['idf1'])



## split movie into four quadrants

mov_file = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629//movie.ufmf'
import os

mov_str = os.path.split(os.path.split(mov_file)[0])[1]
import movies
import cv2
import tqdm
out_dir = '/groups/branson/bransonlab/mayank/apt_expts/alice_split_movie'


##
cap = movies.Movie(mov_file)
nfr = cap.get_n_frames()

out_files = {}
for fr in tqdm.tqdm(range(nfr)):
    im = cap.get_frame(fr)[0]
    h,w = im.shape

    for xx in range(2):
        for yy in range(2):
            if fr == 0:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_file = os.path.join(out_dir, f'{mov_str}_{xx}_{yy}.mp4')
                out_files[(xx,yy)] = cv2.VideoWriter(out_file, fourcc, 20.0, (w // 2, h // 2))
            im1 = im[xx*h//2:(xx+1)*h//2,yy*w//2:(yy+1)*w//2]
            im1 = np.repeat(im1[...,None],3,-1)
            out_files[(xx,yy)].write(im1)

for xx in range(2):
    for yy in range(2):
        out_files[(xx,yy)].release()

## track the split movies

import run_apt_ma_expts as rae_ma
robj = rae_ma.ma_expt('alice')

movies = []
for xx in range(2):
    for yy in range(2):
        cur_mov = os.path.join(out_dir, f'{mov_str}_{xx}_{yy}.mp4')
        movies.append(cur_mov)

run_type = 'dry'
tts = [('grone','crop'),('2stageBBox','first'),('2stageBBox_hrformer','first')]
for cur_mov in movies:
    exp_name = os.path.split(cur_mov)[1].split('.')[0]
    for curt in tts:
        out_trk = os.path.join(robj.trk_dir,exp_name + f'_{curt[0]}_scale2.trk')
        robj.track(cur_mov,out_trk,t_types=[curt,],run_type=run_type,conf_params={'imsz':"\(512,512\)",'min_n_animals':0})

## the command to track the split movies joints was created manually: /groups/branson/home/kabram/temp/ma_expts/alice/log/alice_grone_crop_mask_12072023_nochr_TrpA65F12_Unknown_RigB_20201212T163629_joint_grone_scale2.bsub.sh

## join the split ouptut into a single trk file

import TrkFile
trks = []
x_off = []
y_off = []
for xx in range(2):
    for yy in range(2):
        trks.append(TrkFile.Trk(f'/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigB_20201212T163629_{xx}_{yy}_joint_grone_scale2.trk'))
        x_off.append(xx*512)
        y_off.append(yy*512)

trko = trks[0].copy()
d_shape = trko.pTrk.data[0].shape[:-1]
for tr in range(trko.ntargets):
    if trko.startframes[tr] == -1:
        trko.pTrk.data[tr] = np.ones((d_shape[0],d_shape[1],0))*np.nan
        trko.pTrkTag.data[tr] = np.ones((d_shape[0],0))*0
        trko.pTrkConf.data[tr] = np.zeros((d_shape[0],0))
        trko.pTrkTS.data[tr] = np.zeros((d_shape[0],0))
    for xx in range(1,4):
        curt = trks[xx]
        if (len(curt.startframes)<=tr) or (curt.startframes[tr] == -1):
            continue
        if curt.startframes[tr] < trko.startframes[tr]:
            ns = trko.startframes[tr] - curt.startframes[tr]
            trko.startframes[tr] = curt.startframes[tr]
        else:
            ns = 0
        if curt.endframes[tr] > trko.endframes[tr]:
            ne = curt.endframes[tr] - trko.endframes[tr]
            trko.endframes[tr] = curt.endframes[tr]
        else:
            ne = 0
        arr_s = np.ones((d_shape[0],d_shape[1],ns))*np.nan
        arr_e = np.ones((d_shape[0],d_shape[1],ne))*np.nan
        trko.pTrk.data[tr] = np.concatenate([arr_s,trko.pTrk.data[tr],arr_e],axis=-1)
        trko.pTrkTag.data[tr] = np.concatenate([np.ones((d_shape[0],ns))*0,trko.pTrkTag.data[tr],np.ones((d_shape[0],ne))*0],axis=-1)
        trko.pTrkConf.data[tr] = np.concatenate([np.zeros((d_shape[0],ns)),trko.pTrkConf.data[tr],np.zeros((d_shape[0],ne))],axis=-1)
        trko.pTrkTS.data[tr] = np.concatenate([np.zeros((d_shape[0],ns)),trko.pTrkTS.data[tr],np.zeros((d_shape[0],ne))],axis=-1)

        for fr in range(trko.endframes[tr]-trko.startframes[tr]+1):
            if np.all(np.isnan(trko.pTrk.data[tr][:,:,fr])):
                curd = curt.gettargetframe(tr,fr+trko.startframes[tr],True)
                trko.pTrk.data[tr][:,:,fr] = curd[0][...,0,0] + np.array([y_off[xx],x_off[xx]])
                for kk in curd[1].keys():
                    if kk == 'pTrkAnimalConf': continue
                    trko.__dict__[kk].data[tr][:,fr] = curd[1][kk][...,0,0]


trko.save('/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigB_20201212T163629_joint_grone_scale2.trk',saveformat='tracklet')

## measure the ID accuracy

import motmetrics as mm
import numpy as np
import APT_interface as apt
import TrkFile as trkf
from poseConfig import conf
conf.has_trx_file = True
import os

info = [{}]
#id_trk = '/groups/branson/home/kabram/temp/ar_flytracker2_idlinked.trk';
info[-1]['id_trk'] = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigB_20201212T163629_joint_grone_scale2.trk';
info[-1]['mov_file'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629//movie.ufmf'
info[-1]['fix_error_trx'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629/fixed_trx.mat'
info[-1]['ht_pts'] = [0,6]
info[-1]['ctrax'] = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629/registered_trx.mat'



def get_trx_frame(trx,fr):
    # pts is n_trx x 2 x 2, where second dim has the head and tail points
    valid_trx = np.where( (trx['first_frames']<=fr) & (trx['end_frames']>fr) )[0]
    pts = np.ones([trx['n_trx'],2,2])*np.nan
    for vv in range(trx['n_trx']):
        if vv in valid_trx:
            fro = fr-int(trx['trx'][vv]['firstframe'][0,0])+1
            xx = trx['trx'][vv]['x'].flatten()[fro]-1
            yy = trx['trx'][vv]['y'].flatten()[fro]-1
            theta = trx['trx'][vv]['theta'].flatten()[fro]
            a = trx['trx'][vv]['a'].flatten()[fro]

            hh = np.array([xx+a*np.cos(theta)*2,yy+a*np.sin(theta)*2])
            tt = np.array([xx-a*np.cos(theta)*2,yy-a*np.sin(theta)*2])

            pts[vv,0,:] = hh
            pts[vv,1,:] = tt
    return pts

cur_info = info[-1]
cur_res = {}
acc = mm.MOTAccumulator(auto_id=True)
t2 = apt.get_trx_info(cur_info['fix_error_trx'],conf,None)
t1 = trkf.Trk(cur_info['id_trk'])
nfr = max(t1.nframes)
ht = cur_info['ht_pts']
for fr in range(nfr):
    f1 = t1.getframe(fr)
    vf1 = ~np.isnan(f1[0,0,0,:])
    vix = np.where(vf1)[0]
    f1 = f1[...,vf1]
    f1 = np.transpose(f1[ht,:,0],[2,0,1])
    f2 = get_trx_frame(t2,fr)

    d_mat = np.linalg.norm(f1[None,:]-f2[:,None,:,:],axis=-1).mean(axis=-1)

    acc.update(range(f2.shape[0]),vix,d_mat)

mh = mm.metrics.create()
summ = mh.compute(acc,metrics=mm.metrics.motchallenge_metrics,return_dataframe=False)
cur_res['grone_scale2'] = summ

for kk,vv in cur_res.items():
    print(kk,vv['idf1'])





## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Rat7M
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import PoseTools as pt
import APT_interface as apt
nets = [
        ['mdn_joint_fpn','mdn',{}],
        ['mdn_joint_fpn','mdn_hrnet',{'mdn_use_hrnet':True}],
        ['mmpose','hrnet',{'mmpose_net':'\\"hrnet\\"'}],
         ['mmpose','mspn',{'mmpose_net':'\\"mspn\\"'}],
        ['mmpose','hrformer',{'mmpose_net':'\\"hrformer\\"'}],
#        ['deeplabcut','deeplabcut',{}],
        ]

## convert the data to coco format
for vw in range(1,7):
    cmd = f'/groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/conf.json -conf_params cachedir \"/groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}\" dl_steps 5 -json_trn_file /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/loc.json -ignore_local 1 -type mdn_joint_fpn -train_name dummy -cache /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw} train -use_cache'
    print(cmd)
    apt.main(cmd.split())

##

for vw in range(1,7):
    for net in nets:
        c_str = ''
        for kk,vv in net[2].items():
            c_str += f' {kk} {vv} '
        cmd = f'APT_interface.py /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/conf.json -conf_params cachedir \\"/groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}\\" {c_str} -json_trn_file /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/loc.json -ignore_local 1 -type {net[0]} -train_name {net[1]} -cache /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw} train -use_cache -skip_db'
        print(cmd)
        name = f'rat7m_{vw}_{net[1]}'
        pt.submit_job(name,cmd,dir='/groups/branson/home/kabram/del',queue='gpu_a100')


## track movies
import os
import glob
mov_dir = '/groups/branson/bransonlab/datasets/Rat7M/data'
movs = []
for vw in range(1,7):
    movs.append(glob.glob(os.path.join(mov_dir,f'*camera{vw}*.mp4')))

out_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/trk'
os.makedirs(out_dir,exist_ok=True)
import PoseTools as pt

net = nets[0]
c_str = ''
for kk, vv in net[2].items():
    c_str += f' {kk} {vv} '

force = False
for vw in range(1,7):
    for mm in movs[vw-1]:
        mname = os.path.splitext(os.path.split(mm)[1])[0]
        out_trk = f'{out_dir}/{mname}_{net[1]}.trk'
        if os.path.exists(out_trk) and not force:
            continue
        cmd = f'APT_interface.py /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/conf.json -conf_params cachedir \\"/groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}\\" {c_str} -json_trn_file /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/loc.json -ignore_local 1 -type {net[0]} -train_name {net[1]} -cache /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw} track -mov {mm} -out {out_trk}'
        name = f'rat7m_{mname}_{net[1]}'
        pt.submit_job(name,cmd,dir='/groups/branson/home/kabram/del/rat7m',queue='gpu_a100')


# track using only view 1

all_movs= glob.glob(os.path.join(mov_dir,f'*.mp4'))

vw = 1
for mm in all_movs:
    mname = os.path.splitext(os.path.split(mm)[1])[0]
    out_trk = f'{out_dir}/{mname}_{net[1]}_only_vw{vw}.trk'
    if os.path.exists(out_trk) and not force:
        continue
    cmd = f'APT_interface.py /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/conf.json -conf_params cachedir \\"/groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}\\" {c_str} -json_trn_file /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw}/loc.json -ignore_local 1 -type {net[0]} -train_name {net[1]} -cache /groups/branson/bransonlab/mayank/apt_cache_2/Rat7M/view_{vw} track -mov {mm} -out {out_trk}'
    name = f'rat7m_{mname}_{net[1]}_only_vw{vw}'
    pt.submit_job(name,cmd,dir='/groups/branson/home/kabram/del/rat7m',queue='gpu_a100')
