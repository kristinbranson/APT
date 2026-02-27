from __future__ import print_function
from __future__ import division

##  #######################        SETUP


import matplotlib
matplotlib.use('TkAgg')
import APT_interface as apt
import h5py
import PoseTools
import shutil
import subprocess
import time
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ast
import apt_expts
import os
import pickle
import multiResData
import random
import json
import tensorflow as tf
import hdf5storage
import easydict
import sys
import getpass

if getpass.getuser() == 'leea30':
    import apt_dpk_exps as ade
import util as util
import time


ISPY3 = sys.version_info >= (3, 0)

data_type = None
lbl_file = None
op_af_graph = None
gt_lbl = None
nviews = None
proj_name = None
trn_flies = None
cv_info_file = None
gt_name = ''
dpk_skel_csv = None
dpk_py_path = '/groups/branson/home/leea30/git/dpk:/groups/branson/home/leea30/git/imgaug'
expname_dict_normaltrain = None
# sing_img = '/groups/branson/bransonlab/mayank/singularity/tf23_mmdetection.sif'
default_sing_img = '/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif'

if getpass.getuser() == 'kabram':
    # cache_dir = '/nrs/branson/mayank/apt_cache_2'
    cache_dir = '/groups/branson/bransonlab/mayank/apt_cache_2'
    sdir = os.path.join(cache_dir,'out')
    results_dir = '/groups/branson/bransonlab/mayank/apt_results'
    job_run_dir = os.path.dirname(os.path.realpath(__file__))
elif getpass.getuser() == 'leea30':
    dlroot = '/groups/branson/bransonlab/apt/dl.al.2020'
    cache_dir = os.path.join(dlroot, 'cache')
    sdir = os.path.join(dlroot, 'out')
    results_dir = os.path.join(dlroot, 'res')
    job_run_dir = '/groups/branson/home/leea30/git/apt.aldl/deepnet'
elif getpass.getuser() == 'bransonk':
    dlroot = '/groups/branson/home/bransonk/apt_results'
    cache_dir = os.path.join(dlroot, 'cache')
    sdir = os.path.join(dlroot, 'out')
    results_dir = os.path.join(dlroot, 'res')
    job_run_dir = job_run_dir = os.path.dirname(os.path.realpath(__file__))
else:
    assert False, "Add your cache and out directory"

# all_models = ['mdn','mdn_unet','deeplabcut','mdn_joint', 'openpose','resnet_unet','unet','mdn_joint_fpn','mmpose','leap','leap_orig','deeplabcut_orig']
all_models = ['deeplabcut', 'openpose','mdn_joint_fpn','mmpose_mspn','mmpose_hrnet','mmpose_hrformer','deeplabcut_orig']

print("Your cache is: {}".format(cache_dir))
print("Your models are: {}".format(all_models))

# gpu_model = 'GeForceRTX2080Ti'
n_splits = 3

dlc_aug_use_round = 0

# common_conf procedure. Always call reload() first to initialize the global common_conf state.
# then setup, then individual actions furthwe tweak conf. Always call reload first!

common_conf = {}
# common_conf['rrange'] = 10
# common_conf['trange'] = 5
# common_conf['brange'] = '\(-0.1,0.1\)'
# common_conf['crange'] = '\(0.9,1.1\)'
common_conf['mdn_use_unet_loss'] = False # MDN by default is now without UNet
common_conf['dl_steps'] = 100000
common_conf['decay_steps'] = 25000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8
# common_conf['normalize_img_mean'] = False
# common_conf['adjust_contrast'] = False
common_conf['maxckpt'] = 200 # Save all models
common_conf['ignore_occluded'] = False
common_conf['pretrain_freeze_bnorm'] = True
common_conf['step_lr'] = True
common_conf['lr_drop_step'] = 0.15
common_conf['normalize_loss_batch'] = False
common_conf['use_scale_factor_range'] = True
common_conf['predict_occluded'] = False
common_conf['mdn_pred_dist'] = True

common_conf['mmpose_net'] = '\\"mspn\\"'

# These parameters got added when we moved to min changes to DLC and leap code. These don't exist the stripped label file and hence adding them manually.

# for leap
# common_conf['use_leap_preprocessing'] = False
# common_conf['leap_val_size'] = 0.15
# common_conf['leap_preshuffle'] = True
# common_conf['leap_filters'] = 64
# common_conf['leap_val_batches_per_epoch'] = 10
# common_conf['leap_reduce_lr_factor'] = 0.1
# common_conf['leap_reduce_lr_patience'] = 3
# common_conf['leap_reduce_lr_min_delta'] = 1e-5
# common_conf['leap_reduce_lr_cooldown'] = 0
# common_conf['leap_reduce_lr_min_lr'] = 1e-10
# common_conf['leap_amsgrad'] = False
# common_conf['leap_upsampling'] = False

# for deeplabcut.
common_conf['dlc_intermediate_supervision'] = False
common_conf['dlc_intermediate_supervision_layer'] = 12
common_conf['dlc_location_refinement'] = True
common_conf['dlc_locref_huber_loss'] = True
common_conf['dlc_locref_loss_weight'] = 0.05
common_conf['dlc_locref_stdev'] = 7.2801
common_conf['dlc_use_apt_preprocess'] = True

def setup(data_type_in,gpu_device='0'):
    global lbl_file, op_af_graph, gt_lbl, data_type, nviews, proj_name, trn_flies, cv_info_file, gt_name, \
        dpk_skel_csv, expname_dict_normaltrain, cache_dir, sdir

    data_type = data_type_in
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

    if data_type == 'alice' or data_type=='alice_difficult':
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
        # Old as on 20210629
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20200317_stripped20200403_new_skl_20200817.lbl'
        # gt_lbl = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_allGT_stripped.lbl'
        # Current as on 20210629 -- the op graph is proper and is the one that should be used
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_training_20210523_allGT_AR_mdnGTres_stripped20210629_fiximsz.lbl'
        gt_lbl = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_training_20210523_allGT_AR_mdnGTres_stripped20210629_fiximsz_gt.lbl'

        # op_af_graph_mk = '\(0,1\),\(0,2\),\(0,3\),\(0,4\),\(0,5\),\(5,6\),\(5,7\),\(5,9\),\(9,16\),\(9,10\),\(10,15\),\(9,14\),\(7,11\),\(7,8\),\(8,12\),\(7,13\)'
        # op_af_graph_al = '\(0,1\),\(0,2\),\(0,3\),\(0,4\),\(0,5\),\(5,6\),\(5,7\),\(5,9\),\(9,16\),\(9,10\),\(10,15\),\(5,14\),\(7,11\),\(7,8\),\(8,12\),\(5,13\)'
        # op_af_graph_kb_orig = '\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(3,16\),\(4,11\),\(5,6\),\(5,7\),\(5,9\),\(7,8\),\(5,13\),\(8,12\),\(9,10\),\(5,14\),\(10,15\)'
        # op_af_graph = op_af_graph_kb_orig

        if getpass.getuser() == 'leea30':
            dpk_skel_csv = ade.dbs['alice']['skel']

        expname_dict_normaltrain = {'deeplabcut': 'apt_expt',
                                    'dpk': 'ntrans5_postrescalefixes',
                                    'mdn': 'apt_expt',
                                    'leap': 'apt_expt_mayank',
                                    'openpose': 'apt_expt2',
                                    'sb': 'postrescalefixes',
                                    }

        if data_type == 'alice_difficult':
            gt_lbl = '/nrs/branson/mayank/apt_cache/multitarget_bubble/multitarget_bubble_expandedbehavior_20180425_allGT_MDNvsDLC_labeled_alMassaged20190809_stripped.lbl'
            gt_name = '_diff'
    elif data_type == 'stephen':
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped.lbl'
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updated20200317_compress20200325_stripped20200403.lbl'
        # lbl_file = 'sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped_newparams_20200409.lbl' # this has data from 1, but params from 2
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updated20200317_stripped_mdn.lbl'


        gt_lbl = lbl_file
        #op_af_graph = '\(0,2\),\(1,3\),\(1,4\),\(2,4\)'
        # for vw2; who knows vw1
        # op_af_graph = '\(0,2\),\(1,3\),\(2,4\),\(3,4\),\(2,3\)'
        if getpass.getuser() == 'leea30':
            dpk_skel_csv = ade.dbs[data_type]['skel']

        trn_flies = [212, 216, 219, 229, 230, 234, 235, 241, 244, 245, 251, 254, 341, 359, 382, 417, 714, 719]
        trn_flies = trn_flies[::2]
        # common_conf['trange'] = 20

        expname_dict_normaltrain = {'deeplabcut': 'apt_expt',
                                    'dpk': 'expt_n_transition_min5',
                                    'mdn': 'apt_expt',
                                    'openpose': 'postrescale',
                                    'sb': 'postrescale',
                                    }

    elif data_type == 'roian':
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/roian_apt_dlstripped_newmovielocs.lbl'
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/four_points_all_mouse_linux_tracker_updated20200423_new_skl_20200817.lbl_mdn.lbl'
        # op_af_graph = '\(0,1\),\(0,2\),\(0,3\)'
        # cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/RoianTrainCVInfo20190420.mat'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/roian_xval_20200428.mat'
        # common_conf['rrange'] = 180
        # common_conf['trange'] = 5

        if getpass.getuser() == 'leea30':
            dpk_skel_csv = ade.dbs[data_type]['skel']
    elif data_type == 'brit0':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam0_20200318_compress20200327_new_skl_20200817.lbl_mdn.lbl'
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_0.lbl'
        # op_af_graph = '\(0,4\),\(1,4\),\(2,4\),\(3,4\)'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/brit_1_cv_info_20200408.mat'
        # common_conf['trange'] = 20
    elif data_type == 'brit1':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam1_20209327_compress20200330.lbl_mdn.lbl'
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_1.lbl'
        # op_af_graph = '\(\(0,1\),\)'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/brit_2_cv_info_20200408.mat'
        # common_conf['trange'] = 20
        # dpk_skel_csv = ade.skeleton_csvs[data_type]
    elif data_type == 'brit2':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam2_20200330_compress20200330_new_skl_20200817.lbl_mdn.lbl'
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_2.lbl'
        # op_af_graph = '\(2,0\),\(2,1\)'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/brit_3_cv_info_20200408.mat'
        # common_conf['trange'] = 20
    elif data_type == 'romain':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/romainTrackNov18_updateDec06_al_portable_mdn60k_openposewking_MKfixedmovies_20200911.lbl_mdn.lbl'
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/romain_dlstripped_trn1027.mat'
        # op_af_graph = '(0,6),(6,12),(3,9),(9,15),(1,7),(7,13),(4,10),(10,16),(5,11),(11,17),(2,8),(8,14),(12,13),(13,14),(14,18),(18,17),(17,16),(16,15)'
        # op_af_graph = op_af_graph.replace('(','\(')
        # op_af_graph = op_af_graph.replace(')','\)')
        # dpk_skel_csv = ade.skeleton_csvs[data_type]
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/RomainTrainCVInfo20190419.mat'
        # cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/RomainTrainCVInfo20200107.mat'
        # common_conf['trange'] = 20

        # this is not normaltrain this is cv
        expname_dict_normaltrain = {'deeplabcut': '',
                                    'dpk': 'bs4rs2',
                                    'mdn': '',
                                    'openpose': 'bs4rs2',
                                    'sb': 'bs4rs2',
                                    'unet': '',
                                    'resnet_unet': '',
                                    'leap': '',
                                    }

    elif data_type == 'larva':
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/larva_dlstripped_20190420.lbl'
        # lbl_file = '/groups/branson/bransonlab/apt/experiments/data/gtParamMassage20200409/Larva94A04_CM_fixedmovies_20200331_mdn.lbl'
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/Larva94A04_CM_fixedmovies_20200331.lbl_mdn.lbl'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/LarvaTrainCVInfo20190419.mat'
        # j = tuple(zip(range(27), range(1, 28)))
        # op_af_graph = '{}'.format(j)
        # op_af_graph = op_af_graph.replace('(','\(')
        # op_af_graph = op_af_graph.replace(')','\)')
        # op_af_graph = op_af_graph.replace(' ','')
        if getpass.getuser() == 'leea30':
            dpk_skel_csv = ade.dbs[data_type]['skel']
        # common_conf['trange'] = 20
        # common_conf['rrange'] = 180



    elif data_type == 'carsen':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/carsen_dlstripped_20190501T150134.lbl'
        common_conf['trange'] = 20
        # op_af_graph = '\(\(0,1\),\)'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/CarsenTrainCVInfo20190514.mat'

    elif data_type == 'leap_fly':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/leap_dataset_gt_stripped_new_skl_20200820.lbl'
        gt_lbl = lbl_file
        # gg = np.array(((1,2),(1,3),(1,4),(4,5),(5,6),(7,8),(8,9),(9,10),(11,12),(12,13),(13,14),(15,16),(16,17),(17,18),(19,20),(20,21),(21,22),(23,24),(24,25),(25,26),(27,28),(28,29),(29,30),(4,31),(4,32),(4,7),(4,19),(5,11),(5,15),(5,23),(5,27)))
        # gg = gg-1
        # op_af_graph = '{}'.format(gg.tolist())
        # op_af_graph = op_af_graph.replace('[','\(')
        # op_af_graph = op_af_graph.replace(']','\)')
        # op_af_graph = op_af_graph.replace(' ','')
    else:
        lbl_file = ''

    lbl = h5py.File(lbl_file,'r')
    proj_name = apt.read_string(lbl['projname'])
    nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
    lbl.close()

def run_jobs(cmd_name,
             cur_cmd,
             redo=False,
             run_dir=job_run_dir,
             queue='gpu_any',
             precmd='',
             logdir=sdir,nslots=11,sing_img=default_sing_img, timeout=80*60,n_omp_threads=5):
    logfile = os.path.join(logdir,'opt_' + cmd_name + '.log')
    errfile = os.path.join(logdir,'opt_' + cmd_name + '.err')

    run = False
    if redo:
        run = True
    elif not os.path.exists(errfile):
        run = True
    else:
        ff = open(errfile,'r').read().lower()
        if ff.find('error'):
            run = True
        else:
            run = False

    if run:
        ss_script = os.path.join(os.path.dirname(run_dir), 'matlab', 'repo_snapshot.sh')
        ss_dst = os.path.join(logdir, '{}.snapshot'.format(cmd_name))
        ss_cmd = "{} > {}".format(ss_script, ss_dst)
        subprocess.call(ss_cmd, shell=True)
        print(ss_cmd)
        PoseTools.submit_job(cmd_name, cur_cmd, logdir,
                             # gpu_model=gpu_model,
                             run_dir=run_dir,
                             queue=queue,
                             precmd=precmd,numcores=nslots,
                             timeout=timeout,sing_image=sing_img,n_omp_threads=n_omp_threads)
    else:
        print('NOT submitting job {}'.format(cmd_name))


def get_tstr(tin):
    if np.isnan(tin):
        return ' --------- '
    else:
        return time.strftime('%m/%d %H:%M',time.localtime(tin))

def get_traindata_file_flexible(cache_dir, run_name,expname):
    tfile0 = os.path.join(cache_dir, 'traindata')
    tfile1 = os.path.join(cache_dir, expname + '_' + run_name + '_traindata')
    tfile2 = os.path.join(cache_dir, run_name + '_traindata')
    all_f = [tfile0,tfile1,tfile2]
    a = [ff for ff in all_f if os.path.exists(ff)]
    assert len(a)==1, 'Exactly one traindata file must exist: {} or {} or {}'.format(tfile0, tfile1, tfile2)
    return a[0]


def get_log_files(logdir, cmd_name_base, ext):
    g = os.path.join(logdir, '{}_*.{}'.format(cmd_name_base, ext))
    files = glob.glob(g)
    files.sort(key=os.path.getmtime)
    return files

def check_train_status(cmd_name_base, cache_dir, exp_name,run_name='deepnet'):
    logdir = os.path.join(cache_dir, 'log')
    scriptfiles = get_log_files(logdir, cmd_name_base, 'bsub.sh')
    # errfile = os.path.join(logdir, 'opt_' + cmd_name_base + '.err')

    nsubmits = len(scriptfiles)
    if nsubmits >= 1:
        submit_time = os.path.getmtime(scriptfiles[-1])
    else:
        submit_time = np.nan
        print("Warning: cannot find submit script")

    '''
    if os.path.exists(errfile):
        start_time = os.path.getmtime(errfile)
    else:
        start_time = np.nan
    '''
    train_dist = -1
    val_dist = -1

    files1 = glob.glob(os.path.join(cache_dir, "{}-[0-9]*").format(run_name))
    files2 = glob.glob(os.path.join(cache_dir, "{}_202[0-9][0-9][0-9][0-9][0-9]-[0-9]*").format(run_name))
    files = files1+files2

    files.sort(key=os.path.getmtime)
    files = [f for f in files if os.path.splitext(f)[1] in ['.index','']]

    # latest model, time, train_dist etc
    if len(files)>0:
        latest = files[-1]
        latest_model_iter = int(re.search('-(\d*)', latest).groups(0)[0])
        latest_time = os.path.getmtime(latest)
        if latest_time < submit_time:
            latest_time = np.nan
            latest_model_iter = -1
        else:
            # if submit_time is nan, assume the latest model is up to date
            tfile = get_traindata_file_flexible(cache_dir, run_name,exp_name)
            A = PoseTools.pickle_load(tfile)
            if type(A) is list:
                train_dist = A[0]['train_dist'][-1]
                val_dist = A[0]['val_dist'][-1]
    else:
        latest_model_iter = -1
        latest_time = np.nan

    # trn time
    sec_per_iter = np.nan
    if len(files) > 0:
        first = files[0]
        first_model_iter = int(re.search('-(\d*)', first).groups(0)[0])
        first_time = os.path.getmtime(first)
        if latest_time > submit_time and first_time > submit_time:
            diter = latest_model_iter - first_model_iter
            dtime = latest_time - first_time
            sec_per_iter = np.array(dtime)/np.array(diter)
            #min_per_5kiter = sec_per_iter * 5000 / 60
    if np.isnan(sec_per_iter) or np.isinf(sec_per_iter):
        trntime5kiter = '---'
    else:
        trntime5kiter = str(datetime.timedelta(seconds=np.round(sec_per_iter*5000)))

    print('latest iter: {:06d} at {}, {:45s}. nsub: {} submit: {}. train:{:.2f} val:{:.2f}. trntime/5kiter:{}'.format(
        latest_model_iter, get_tstr(latest_time), cmd_name_base, nsubmits, get_tstr(submit_time),
        train_dist, val_dist, trntime5kiter))
    return latest_model_iter

def plot_results(data_in,ylim=None,xlim=None):
    if sys.version_info[0] == 3:
        # for Python3
        import tkinter as Tkinter
    else:
        # for Python2
        import Tkinter
    try:
        return plot_results1(data_in,ylim,xlim)
    except Tkinter.TclError:
        pass

def plot_results1(data_in,ylim=None,xlim=None):
    ps = [50, 75, 90, 95]
    k = list(data_in.keys())[0]
    npts = data_in[k][0][0].shape[1]
    nc = int(np.ceil(np.sqrt(npts+1)))
    nr = int(np.ceil((npts+1)/float(nc)))
    f, ax = plt.subplots(nr, nc, figsize=(10, 10))
    ax = ax.flat
    leg = []
    cc = PoseTools.get_colors(len(data_in))
    all_k = data_in.keys()
    for idx,k in enumerate(all_k):
        mm = []
        mt = []
        for o in data_in[k]:
            dd = np.sqrt(np.sum((o[0] - o[1]) ** 2, axis=-1))
            mm.append(np.nanpercentile(dd, ps, axis=0))
            mt.append(o[-1])
        if len(mt) == 0:
            continue
        t0 = mt[0]
        mt = np.array([t - t0 for t in mt])
        mm = np.array(mm)

        for ndx in range(npts):
            ax[ndx].plot(mt[1:], mm[1:, :, ndx], color=cc[idx, :])
            if xlim is not None:
                ax[ndx].set_xlim([0, xlim])
            if ylim is not None:
                ax[ndx].set_ylim([0, ylim])
        leg.append('{}'.format(k))
        ax[-1].plot([0, 1], [0, 1], color=cc[idx, :])
    ax[-1].legend(leg)
    return f


def plot_hist(in_exp, ps = [50,75,90,95],cmap=None):
    if sys.version_info[0] == 3:
        # for Python3
        import tkinter as Tkinter
    else:
        # for Python2
        import Tkinter
    try:
        return plot_hist1(in_exp,ps,cmap)
    except Tkinter.TclError:
        pass


def plot_hist1(in_exp,ps = [50, 75, 90, 95],cmap=None):
    data_in, ex_im, ex_loc = in_exp
    k = list(data_in.keys())[0]
    npts = data_in[k][0][0].shape[1]

    n_types = len(data_in)
    nc = n_types # int(np.ceil(np.sqrt(n_types)))
    nr = 1#int(np.ceil(n_types/float(nc)))
    nc = int(np.ceil(np.sqrt(n_types)))
    nr = int(np.ceil(n_types/float(nc)))
    if cmap is None:
        cmap = PoseTools.get_cmap(n_types,'hsv')
    f, axx = plt.subplots(1, 1, figsize=(12, 12), squeeze=False)
    axx = axx.flat
    ax = axx[0]
    if ex_im.ndim == 2:
        ax.imshow(ex_im, 'gray')
    elif ex_im.shape[2] == 1:
        ax.imshow(ex_im[:, :, 0], 'gray')
    else:
        ax.imshow(ex_im)

    all_jj = []
    all_str = []
    for idx,k in enumerate(data_in.keys()):
        if len(data_in[k]) == 0:
            continue
        o = data_in[k][-1]
        dd = np.sqrt(np.sum((o[0] - o[1]) ** 2, axis=-1))

        jj = np.sort(dd,axis=0)
        all_jj.append(jj)
        mm = np.nanpercentile(dd, ps, axis=0)
        ttl = '{} (n={})'.format(k,dd.shape[0])
        all_str.append(ttl)

    for nndx1, n in enumerate(all_str):
        plt.plot([0, 0], [1, 1], c=cmap[nndx1])
    plt.legend(all_str)
    plt.scatter(ex_loc[:, 0], ex_loc[:, 1], c='r', s=10, marker='+')

    for idx,k in enumerate(data_in.keys()):
        for pt in range(ex_loc.shape[0]):
            jj = all_jj[idx]
            for ix in range(ex_loc.shape[0]):
                vv = ~np.all(np.isnan(o[1][..., 0]), axis=-1)
                n_ex = np.count_nonzero(~np.isnan(o[1][vv][..., ix, 0]))
                st = n_ex * 8 // 10
                n_t = n_ex - st
                th = np.arange(n_t) / n_t * np.pi * 2
                xj = jj[st:n_ex, ix] * np.sin(th)
                yj = jj[st:n_ex, ix] * np.cos(th)
                plt.scatter(xj + ex_loc[ix, 0], yj + ex_loc[ix, 1], c=cmap[idx], marker='.', alpha=0.5, s=2)
                min_jj = min([all_jj[nndx1][st, ix] for nndx1 in range(len(all_jj))])
                aa = np.maximum(0.5 * min_jj / jj[st:n_ex, ix], 0.05)
                aa[np.isnan(aa)] = 0
                for qx, ixx in enumerate(range(st, n_ex - 1)):
                    plt.plot(xj[qx:qx + 2] + ex_loc[ix, 0], yj[qx:qx + 2] + ex_loc[ix, 1], color=cmap[idx], lw=2,alpha=aa[qx + 1])


    for ax in axx:
        ax.set_xlim([0,ex_im.shape[1]])
        ax.set_ylim([ex_im.shape[0],0])
        ax.axis('off')

    f.tight_layout()
    return f



def plot_hist1_circular(in_exp,ps = [50, 75, 90, 95],cmap=None):
    data_in, ex_im, ex_loc = in_exp
    k = list(data_in.keys())[0]
    npts = data_in[k][0][0].shape[1]

    n_types = len(data_in)
    nc = n_types # int(np.ceil(np.sqrt(n_types)))
    nr = 1#int(np.ceil(n_types/float(nc)))
    nc = int(np.ceil(np.sqrt(n_types)))
    nr = int(np.ceil(n_types/float(nc)))
    if cmap is None:
        cmap = PoseTools.get_cmap(len(ps),'cool')
    f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
    axx = axx.flat
    for idx,k in enumerate(data_in.keys()):
        if len(data_in[k]) == 0:
            continue
        o = data_in[k][-1]
        dd = np.sqrt(np.sum((o[0] - o[1]) ** 2, axis=-1))
        mm = np.nanpercentile(dd, ps, axis=0)

        n = dd.shape[0]

        ax = axx[idx]
        if ex_im.ndim == 2:
            ax.imshow(ex_im,'gray')
        elif ex_im.shape[2] == 1:
            ax.imshow(ex_im[:,:,0],'gray')
        else:
            ax.imshow(ex_im)

        for pt in range(ex_loc.shape[0]):
            for pp in range(mm.shape[0]):
                c = plt.Circle(ex_loc[pt,:],mm[pp,pt],color=cmap[pp,:],fill=False)
                ax.add_patch(c)
        ttl = '{} (n={})'.format(k,n)
        ax.set_title(ttl)
        ax.axis('off')

    f.tight_layout()
    return f


def save_mat(out_exp,out_file):
    out_file += PoseTools.datestr()
    import hdf5storage
    out_arr = {}
    for k in out_exp.keys():
        cur = out_exp[k]
        all_dd = []
        for c in cur:
            dd = {}
            dd[u'pred'] = c[0]
            dd[u'labels'] = c[1]
            dd[u'info'] = np.array(c[2])
            dd[u'model_file'] = c[3]
            dd[u'model_timestamp'] = c[5]
            iter = int(re.search('-(\d*)', c[3]).groups(0)[0])
            dd[u'model_iter'] = iter
            if type(c[4]) in [list,tuple] and len(c[4])>0:
                dd[u'mdn_pred'] = c[4][0][0]
                dd[u'unet_pred'] = c[4][0][1]
                dd[u'mdn_conf'] = c[4][0][2]
                dd[u'unet_conf'] = c[4][0][3]

            all_dd.append(dd)
        if sys.version_info[0] >= 3:
            unicode = str
        out_arr[unicode(k)] = all_dd
    hdf5storage.savemat(out_file,out_arr,truncate_existing=True)


# conf_opts format notes
# APT_interf accepts conf_params as a string of PVs. This is prob good, we need a hard boundary where
# APT_interf can always be run with arbitrary specification from the cmdline.
# This does mean that conf_params needs to be serializable to string and in some cases this means escaping parens etc.
# Formats for values:
# 1. double-escaped strings, because conf_opts first gets printed (removing one escape), then gets parsed (removing another)
# 2. single-escaped strings, etc
# 3. actual values. These are converted to strings via print() and vice versa via ast.literal_eval
# Formats for structure:
# A. dict
# B. pv list

'''
def conf_opts_dict_to_pvlist(conf_opts):
    if ISPY3:
        pvs = list(conf_opts.items())
        pvlist = [i for el in pvs for i in el]
    else:
        assert False, "todo"
'''

def run_training_conf_helper_dpk(conf_opts, keywordargs):
    '''
    dpk, dpk-style train (or more precisely, *not* apt-style train).

    :param conf_opts: modified in place
    :param keywordargs: modified in place
    :return: none
    '''

    dpk_train_style = keywordargs['dpk_train_style']

    # nepoch = 300  # hardcoded based on observing dpk-style
    conf_opts['batch_size'] = 16
    if data_type == 'alice':
        # ntrn=4232, 300 epochs, 264.5 steps/ep
        conf_opts['dl_steps'] = 79500
    elif data_type == 'stephen':
        # sh: ntrn=4493, 300 epochs, 281 steps/ep
        conf_opts['dl_steps'] = 84000
    elif data_type == 'roian':
        conf_opts['dl_steps'] = 50000
        conf_opts['batch_size'] = 16
        conf_opts['rescale'] = 2
        # rn: ntrn~2500, bsize=16 => 164 spe => 300 epochs =>
        # patience20=3k iters, patience50=7.5k iters
        #                bsize=8 => 312 spe
        #                bsize=4 => 625spe => patience=20 is 12.5k iters
    elif data_type == 'larva':
        conf_opts['dl_steps'] = 50000
        conf_opts['batch_size'] = 16
        conf_opts['rescale'] = 4
        # ntrn~433
        #   bsize=16 => 27 spe => pat20=540 iters, pat50=1350 iters
        #               50 spe    pat20=1k iters   pat50=2.5k iters
        #   bsize=4 => 108 spe
    elif data_type == 'brit1':
        pass
        # ntrn~1500, roughly like leapfly
        # bsize=16 => 94 spe => pat20=1880 iters, pat50=4700 iters
        # bsize=8 => 190 spe => pat20=3700 iters, pat50=10k iters
        # bsize=4 => 375 spe => pat20=7500 iters, pat50=18750 iters
        #conf_opts['dl_steps'] = 50000
        #conf_opts['batch_size'] = 16
    elif data_type == 'romain':
        conf_opts['batch_size'] = 4
        conf_opts['rescale'] = 2
        conf_opts['dl_steps'] = 200000
        # ntrn=421,nval=187=11*17
        #   bsize=4 => 105 spe => pat20=2k steps

        # leapfly: ntrn~1350. bsize=16=>84 spe. pat20=1680 steps
    else:
        assert False

    if dpk_train_style == 'dpk':
        conf_opts['dpk_reduce_lr_style'] = '\\"ipynb\\"'
        conf_opts['dpk_early_stop_style'] = '\\"ipynb2\\"'
        keywordargs['dpk_train_style'] = '\\"dpk\\"'  # copied to conf_opts below

    elif dpk_train_style == 'dpktrnonly':
        # like 'dpk', but LR and ES use trn_loss not val_loss
        conf_opts['dpk_reduce_lr_style'] = '\\"ipynb\\"'
        conf_opts['dpk_early_stop_style'] = '\\"ipynb2\\"'
        keywordargs['dpk_train_style'] = '\\"dpktrnonly\\"'  # copied to conf_opts below

    else:
        assert False
    '''
    if conf_opts['dpk_use_augmenter']:
        # conf_opts['dpk_augmenter_type'] = xxx
        assert False, 'Todo'
    else:
        # use PT. conf/conf_opts should have PT dataaug stuff properly config'd
        pass
    '''



def run_trainining_conf_helper(train_type, view0b, gpu_queue, kwargs):
    '''
    Helper function that takes common_conf and further sets up for particular train_type

    :param train_type:
    :param kwargs:
    :return:
    '''

    conf_opts = common_conf.copy()
    # conf_opts.update(other_conf[conf_id])

    conf_opts['save_step'] = conf_opts['dl_steps'] // 20

    #these are now defaults
    #if train_type == 'dpk' and 'dpk_train_style' not in kwargs:
    #    kwargs['dpk_train_style'] = 'apt'
    #    kwargs['dpk_val_batch_size'] = 0

    if train_type.startswith('mmpose'):
        conf_opts['mmpose_net'] = '\\"' + train_type.split('_')[1] + '\\"'


    if train_type == 'dpk' and kwargs['dpk_train_style'] != 'apt':
        # 'dpk_orig'
        run_training_conf_helper_dpk(conf_opts, kwargs)

    elif gpu_queue == 'gpu_rtx':

        if train_type == 'dpk':
            conf_opts['dpk_reduce_lr_style'] = '\\"__UNUSED__\\"'
            conf_opts['dpk_early_stop_style'] = '\\"__UNUSED__\\"'
            kwargs['dpk_train_style'] = '\\"apt\\"'  # copied to conf_opts below

            # We are going to punt on lrmult for dpk, see slack discuss. Note the
            # keras handling of loss differs from PoseBase. Typically the LR linear
            # scaling rule would reduce the LR and 1e-4 is already 10x smaller than
            # the 'default' dpk factory base LR.
            conf_opts['dpk_base_lr_used'] = 0.0001

            ''' this was older apt-sty
            conf_opts['dpk_base_lr_used'] = 0.001
            conf_opts['gamma'] = 0.2  # matches decay const dpk-style
            conf_opts['decay_steps'] = 25000
            '''

        if data_type in ['brit0', 'brit1', 'brit2']:
            if train_type in ['unet','openpose']:
                conf_opts['batch_size'] = 2
            else:
                conf_opts['batch_size'] = 4

        if data_type in ['roian']:
            # fix at bs=4,rs=1 maybe have to run on tesla
            if train_type in ['dpk', 'openpose', 'sb','mdn_unet','resnet_unet','unet','mdn_joint_fpn']:
                conf_opts['batch_size'] = 4
                conf_opts['rescale'] = 2
            elif train_type in ['mdn','deeplabcut','leap']:
                conf_opts['batch_size'] = 4

        if data_type in ['romain']:
            if train_type in ['mdn_unet']:
                conf_opts['batch_size'] = 2
            elif train_type in ['resnet_unet']:
                conf_opts['batch_sze'] = 2
            elif train_type in ['unet']:
                conf_opts['batch_size'] = 2
                conf_opts['rescale'] = 2
            elif train_type in ['sb', 'dpk', 'openpose', 'leap']:
                # added leap random guess, otherwise OOM
                conf_opts['rescale'] = 2
                conf_opts['batch_size'] = 4
            else:
                conf_opts['batch_size'] = 4

        if data_type in ['larva']:
            conf_opts['batch_size'] = 4
            # conf_opts['adjust_contrast'] = True
            # conf_opts['clahe_grid_size'] = 20

            if train_type in ['openpose']:
                conf_opts['op_backbone'] = '\\"vgg\\"'
                conf_opts['batch_size'] = 2
                conf_opts['op_hires'] = True
                conf_opts['op_hires_ndeconv'] = 1
                #conf_opts['rescale'] = 1
                #conf_opts['batch_size'] = 4

            if train_type in ['sb', 'dpk']:
                conf_opts['rescale'] = 4
                conf_opts['batch_size'] = 4
            if train_type in ['unet', 'resnet_unet']:
                conf_opts['rescale'] = 4
                conf_opts['batch_size'] = 4
            if train_type in ['mdn_unet','leap','mdn_joint_fpn']:
                conf_opts['batch_size'] = 4
                conf_opts['rescale'] = 2

        # if data_type == 'stephen':
        #     conf_opts['batch_size'] = 8

        if data_type == 'carsen':
            if train_type in [ 'unet', 'resnet_unet']:
                conf_opts['rescale'] = 2.
            elif train_type in ['mdn_unet']:
                conf_opts['rescale'] = 2.
            else:
                conf_opts['rescale'] = 1.
            if train_type in ['unet']:
                conf_opts['batch_size'] = 4
            else:
                conf_opts['batch_size'] = 8

    elif gpu_queue in ['gpu_tesla','gpu_tesla_large']:

        if train_type == 'dpk':
            conf_opts['dpk_reduce_lr_style'] = '\\"__UNUSED__\\"'
            conf_opts['dpk_early_stop_style'] = '\\"__UNUSED__\\"'
            kwargs['dpk_train_style'] = '\\"apt\\"'  # copied to conf_opts below

            # We are going to punt on lrmult for dpk, see slack discuss. Note the
            # keras handling of loss differs from PoseBase. Typically the LR linear
            # scaling rule would reduce the LR and 1e-4 is already 10x smaller than
            # the 'default' dpk factory base LR.
            conf_opts['dpk_base_lr_used'] = 0.0001

        if data_type in ['romain']:
            conf_opts['batch_size'] = 4
        if data_type in ['larva']:
            conf_opts['batch_size'] = 4
            conf_opts['rescale'] = 2
        if data_type in ['roian']:
            conf_opts['batch_size'] = 4
            # if train_type in ['resnet_unet','unet']:
            #     conf_opts['batch_size'] = 4

    # if op_af_graph is not None:
    #     conf_opts['op_affinity_graph'] = op_af_graph

    if dpk_skel_csv is not None:
        conf_opts['dpk_skel_csv'] = '\\"' + dpk_skel_csv[view0b] + '\\"'

    # Use exp decay for unet and resnet_unet
    if train_type in ['unet','resnet_unet']:
        conf_opts['step_lr'] = False

    if train_type in ['unet']:
        conf_opts['learning_rate_multiplier'] =  3.

    if op_af_graph is not None:
        conf_opts['op_affinity_graph'] = op_af_graph

    if train_type.startswith('mmpose') or train_type in ['mmpose','mdn_joint_fpn']:
        conf_opts['db_format'] = '\\"coco\\"'

    for k in kwargs.keys():
        conf_opts[k] = kwargs[k]

    return set_training_params(conf_opts,train_type)


def set_training_params(conf_opts,train_type='mdn'):
    if train_type == 'dpk' and conf_opts['dpk_train_style'] != '\\"apt\\"':
        return conf_opts

    bsz = conf_opts['batch_size']
    default_bsz = 8
    conf_opts['dl_steps'] = common_conf['dl_steps']*default_bsz//bsz
    conf_opts['decay_steps'] = conf_opts['dl_steps']//2
    # conf_opts['decay_steps'] = common_conf['decay_steps']*default_bsz//bsz
    # conf_opts['learning_rate_multiplier'] = bsz/float(default_bsz)
    if train_type == 'deeplabcut':
        conf_opts['dl_steps'] = None
        conf_opts['save_step'] = 8*conf_opts['save_step']
    return conf_opts


def cp_exp_bare(src_exp_dir, dst_exp_dir, usesymlinks=True):
    '''
    Copy training dbs etc from existing expdir to new empty expdir
    :param src_exp_dir existing expdir
    :param dst_exp_dir: new expdir, created if nec
    :return:
    '''

    if usesymlinks:
        assert os.path.dirname(src_exp_dir) == os.path.dirname(dst_exp_dir), \
            "src and dst expdirs must be located in same parent dir"
        src_exp_dir_base = os.path.basename(src_exp_dir)

    if not os.path.exists(dst_exp_dir):
        os.makedirs(dst_exp_dir)
        print("Created dir {}",format(dst_exp_dir))

    GLOBSPECS = ['*.tfrecords', 'splitdata.json']
    for globspec in GLOBSPECS:
        gs = os.path.join(src_exp_dir, globspec)
        globres = glob.glob(gs)
        for src in globres:
            fileshort = os.path.basename(src)
            dst = os.path.join(dst_exp_dir, fileshort)
            if usesymlinks:
                srcrel = os.path.join('..',src_exp_dir_base,fileshort)
                os.symlink(srcrel, dst)
                print("Symlinked {}->{}".format(dst, srcrel))
            else:
                shutil.copyfile(src, dst)
                print("Copied {}->{}".format(src, dst))


def run_trainining(exp_name,train_type,view,run_type,
                   train_name='deepnet',
                   cp_from_existing_exp=None,  # short expname same dir as exp_name
                   exp_note='',
                   queue='gpu_rtx8000',
                   dstr=PoseTools.datestr(),
                   nslots=None,
                   **kwargs
                   ):

    # if run_type == 'submit':
    #     time.sleep(25) # five second time to figure out if I really wanted to submit the job
    gpu_str = '_tesla' if queue in ['gpu_tesla','gpu_tesla_large'] else ''
    train_name_dstr = train_name + gpu_str + '_' + dstr

    precmd, cur_cmd, cmd_name, cmd_name_base, conf_opts = \
        apt_train_cmd(exp_name, train_type, view, train_name_dstr, queue, **kwargs)
    if nslots is None:
        if queue in ['gpu_tesla_large','gpu_rtx8000']:
            if train_type == 'leap':
                nslots = 5
            elif data_type == 'larva' and train_type in ['mdn','mdn_joint','mdn_joint_fpn','mdn_unet']:
                nslots = 5
            else:
                nslots = 2
        elif queue in ['gpu_tesla']:
            if train_type == 'leap':
                nslots = 6
            else:
                nslots = 5
        else:
            if train_type == 'leap':
                nslots = 10
            elif data_type == 'larva' and train_type in ['mdn','mdn_joint','mdn_joint_fpn']:
                nslots = 7
            else:
                nslots = 5

    if run_type == 'dry':
        print(cmd_name)
        print("precmd: {}".format(precmd))
        print("cmd: {}".format(cur_cmd))
        return conf_opts, cur_cmd, cmd_name
    elif run_type == 'submit':
        # C+P mirror of APT_interf
        exp_dir_parent = os.path.join(cache_dir, proj_name, train_type, 'view_{}'.format(view))
        exp_dir = os.path.join(exp_dir_parent, exp_name)

        if cp_from_existing_exp is not None:
            existing_exp = os.path.join(exp_dir_parent, cp_from_existing_exp)
            assert os.path.exists(existing_exp)
            assert not os.path.exists(exp_dir), "exp_dir already exists: {}".format(exp_dir)

            #exp_dir_bak = '{}.bak{}'.format(exp_dir, now_str)
            #os.rename(exp_dir, exp_dir_bak)
            #print("Existing expdir {} renamed to {}.".format(exp_dir, exp_dir_bak))
            cp_exp_bare(existing_exp, exp_dir)


        # code snapshot is done downstream in run_jobs/PoseTools submit

        # logdir
        explog_dir = os.path.join(exp_dir, 'log')
        if not os.path.exists(explog_dir):
            os.makedirs(explog_dir, exist_ok=True)  # Py3.2+ only

        if len(exp_note)>0:
            notefile = os.path.join(exp_dir,'EXPNOTE')
            if os.path.exists(notefile):
                print("Notefile {} exists already; not overwriting".format(notefile))
            else:
                with open(notefile,'w') as fh:
                    fh.write(exp_note)
                print("Wrote note to {}".format(notefile))

        print(cur_cmd)
        print()
        run_jobs(cmd_name, cur_cmd, precmd=precmd, logdir=explog_dir, queue=queue,nslots=nslots)
    elif run_type == 'status':
        #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        #time0 = time.process_time()
        conf = create_conf_help(train_type, view, exp_name, quiet=True, **kwargs)
        #time1 = time.process_time()
        #print("create_conf time is {}".format(time1 - time0))

        #time0 = time.process_time()
        check_train_status(cmd_name_base, conf.cachedir,conf.expname,train_name_dstr)
        #time1 = time.process_time()
        #print("check train status time is {}".format(time1 - time0))

def apt_train_cmd(exp_name, train_type, view, train_name, queue, **kwargs):

    common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file, exp_name, cache_dir)
    end_cmd = 'train -skip_db -use_cache'

    conf_opts = run_trainining_conf_helper(train_type, view, queue, kwargs)  # this is copied from common_conf
    conf_str = apt.conf_opts_dict2pvargstr(conf_opts)

    cmd_opts = {}
    if train_type.startswith('mmpose'):
        cmd_opts['type'] = 'mmpose'
        cmd_opts['train_name'] = train_name + '_' + train_type.split('_')[1]
    else:
        cmd_opts['type'] = train_type
        cmd_opts['train_name'] = train_name
    cmd_opts['view'] = view + 1
    opt_str = ''
    for k in cmd_opts.keys():
        opt_str = '{} -{} {} '.format(opt_str, k, cmd_opts[k])

    now_str = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    cur_cmd = common_cmd + conf_str + opt_str + end_cmd
    cmd_name_base = '{}_view{}_{}_{}_{}'.format(data_type, view, exp_name, train_type, train_name)
    cmd_name = '{}_{}'.format(cmd_name_base, now_str)
    precmd = 'export PYTHONPATH="{}"'.format(dpk_py_path) if train_type == 'dpk' else ''

    return precmd, cur_cmd, cmd_name, cmd_name_base, conf_opts

def create_conf_help(train_type, view, exp_name, queue='gpu_rtx', quiet=False, **kwargs):
    '''
    Call apt.create_conf after customizing the conf for the given train_type/view/kwargs.
    :param train_type:
    :param view:
    :param exp_name:
    :param kwargs:
    :return:
    '''
    conf_opts = run_trainining_conf_helper(train_type, view, queue,kwargs)
    pvlist = apt.conf_opts_dict2pvargstr(conf_opts)
    pvlist = apt.conf_opts_pvargstr2list(pvlist)
    if train_type.startswith('mmpose'):
        train_type = train_type.split('_')[0]
    #time0 = time.process_time()
    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type,
                           conf_params=pvlist, quiet=quiet)
    #print("create_conf time is {}".format(time.process_time()-time0))
    return conf

def get_apt_conf(**kwargs):
    '''
    Create/generate confs as (hopefully) done by Apt_interf
    :return:
    '''

    res = [None,] * nviews
    exp_name = 'apt_expt'
    for view in range(nviews):
        for tndx in range(len(all_models)):
            train_type = all_models[tndx]
            conf = create_conf_help(train_type, view, exp_name, kwargs)
            if res[view] is None:
                res[view] = {}
            res[view][all_models[tndx]] = conf

    return res


def create_normal_dbs(expname ='apt_expt'):

    # assert gt_lbl is not None
    for view in range(nviews):
        for tndx in range(len(all_models)):
            train_type = all_models[tndx]
            conf = create_conf_help(train_type, view, expname)
            if 'deeplabcut' in train_type:
                apt.create_deepcut_db(conf,split=False,use_cache=True)
            elif 'leap' in train_type:
                apt.create_leap_db(conf,split=False,use_cache=True)
            elif train_type.startswith('mmpose') or train_type in ['mmpose', 'mdn_joint_fpn']:
                apt.create_coco_db(conf, split=False, use_cache=True)
            else:
                apt.create_tfrecord(conf,split=False,use_cache=True)


def cv_train_from_mat(skip_db=True, run_type='status', create_splits=False,
                      exp_name_pfix='',  # prefix for exp_name
                      split_idxs=None,  # optional list of split indices to run (0-based)
                      view_idxs=None,  # optional list of view indices to run (0-based)
                      queue='gpu_rtx8000',
                      override_setup={},
                      **kwargs):
    if len(override_setup)==0:
        assert data_type in ['romain','larva','roian','carsen','brit0','brit1','brit2']
    else:
        for k,v in override_setup.items():
            globals()[k] = v

    data_info = h5py.File(cv_info_file, 'r')
    cv_info = apt.to_py(np.squeeze(data_info['cvi']).astype('int'))
    n_splits = max(cv_info) + 1
    conf = apt.create_conf(lbl_file,0,'cv_dummy',cache_dir,'mdn')
    lbl_movies = multiResData.find_local_dirs(conf.labelfile)
    in_movies = [PoseTools.read_h5_str(data_info[k]) for k in data_info['movies'][0,:]]
    assert lbl_movies == in_movies or data_type in ['romain','roian']
    label_info = get_label_info(conf)
    fr_info = apt.to_py(np.squeeze(data_info['frame']).astype('int'))
    m_info = apt.to_py(np.squeeze(data_info['movieidx']).astype('int'))
    if 'target' in data_info.keys():
        t_info = apt.to_py(np.squeeze(data_info['target']).astype('int'))
        in_info = [(a,b,c) for a,b,c in zip(m_info,fr_info,t_info)]
    else:
        in_info = [(a,b,0) for a,b in zip(m_info,fr_info)]
    diff1 = list(set(label_info)-set(in_info))
    diff2 = list(set(in_info)-set(label_info))
    print('Number of labels that exists in label file but not in mat file:{}'.format(len(diff1)))
    print('Number of labels that exists in mat file but not in label file:{}'.format(len(diff2)))
    # assert all([a == b for a, b in zip(in_info, label_info)])

    if split_idxs is not None:
        assert all([x in range(n_splits) for x in split_idxs])
    else:
        split_idxs = range(n_splits)

    if view_idxs is not None:
        assert all([x in range(nviews) for x in view_idxs])
    else:
        view_idxs = range(nviews)

    for sndx in split_idxs:
        val_info = [l for ndx,l in enumerate(in_info) if cv_info[ndx]==sndx]
        trn_info = list(set(label_info)-set(val_info))
        cur_split = [trn_info,val_info]
        exp_name = '{}cv_split_{}'.format(exp_name_pfix, sndx)
        split_file = os.path.join(cache_dir,proj_name,exp_name) + '.json'
        if not skip_db and create_splits:
            if os.path.exists(split_file):
                print("Warning, overwriting existing split file {}", format(split_file))
            def convert(o):
                if isinstance(o, np.int64): return int(o)
                raise TypeError
            with open(split_file,'w') as f:
                json.dump(cur_split,f,default=convert)

        # create the dbs
        if not skip_db:
            for view in view_idxs:
                for train_type in all_models:
                    conf = create_conf_help(train_type, view, exp_name, **kwargs)
                    #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                    conf.splitType = 'predefined'
                    if 'deeplabcut' in train_type:
                        apt.create_deepcut_db(conf, split=True, split_file=split_file, use_cache=True)
                    elif 'leap' in train_type:
                        apt.create_leap_db(conf, split=True, split_file=split_file, use_cache=True)
                    elif 'mmpose' in train_type or train_type in ['mmpose', 'mdn_joint_fpn']:
                        apt.create_coco_db(conf, split=True, split_file=split_file, use_cache=True)
                    else:
                        apt.create_tfrecord(conf, split=True, split_file=split_file, use_cache=True)

    for view in view_idxs:
        for train_type in all_models:
            for sndx in split_idxs:
                exp_name = '{}cv_split_{}'.format(exp_name_pfix, sndx)
                run_trainining(exp_name,train_type,view,run_type, queue=queue, **kwargs)

def my_move(src, destpath):
    if os.path.exists(src):
        shutil.move(src,destpath) # COMMENT ME to test
        #myMove.counter += 1
    else:
        print("warning: file {} DNE.".format(src))

def cv_train_backup(run_id, dryrun=False, exp_name_pfix=''):
    data_info = h5py.File(cv_info_file, 'r')
    cv_info = apt.to_py(data_info['cvi'].value[:, 0].astype('int'))
    n_splits = max(cv_info) + 1

    split_idxs = range(n_splits)
    view_idxs = range(nviews)

    for view in view_idxs:
        for train_type in all_models:
            for sndx in split_idxs:
                exp_name = '{}cv_split_{}'.format(exp_name_pfix, sndx)
                conf = create_conf_help(train_type, view, exp_name)
                edir = conf.cachedir

                GLOBSKEEP = ['splitdata.json', 'train_TF.tfrecords', 'val_TF.tfrecords', 'run*']
                artskeep = []
                for gl in GLOBSKEEP:
                    gl = os.path.join(edir,gl)
                    #print "glob:", gl
                    artskeep += glob.glob(gl)
                artsall = glob.glob(os.path.join(edir, '*'))
                artsbak = set(artsall) - set(artskeep)
                
                if dryrun:
                    print(edir)
                    for a in artsbak:
                        print("  {} -> {}".format(a, run_id))
                    for a in artskeep:
                        print("  Kept: {}".format(a))
                else:
                    arcdir = os.path.join(edir, run_id)
                    assert not os.path.exists(arcdir)
                    os.mkdir(arcdir)
                    for a in artsbak:
                        my_move(a, arcdir)



def cv_train_britton(skip_db=True, run_type='status'):
    assert False, 'Use cv from mat'
    assert data_type[:4] == 'brit'
    britnum = int(data_type[4])
    n_splits = 3
    for sndx in range(n_splits):
        exp_name = 'cv_split_{}'.format(sndx)
        split_file = os.path.join(cache_dir,proj_name,exp_name) + '.json'
        # create the dbs
        if not skip_db:
            for view in range(nviews):
                for train_type in all_models:
                    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                    conf.splitType = 'predefined'
                    if 'deeplabcut' in train_type:
                        apt.create_deepcut_db(conf, split=True, split_file=split_file, use_cache=True)
                    elif 'leap' in train_type:
                        apt.create_leap_db(conf, split=True, split_file=split_file, use_cache=True)
                    else:
                        apt.create_tfrecord(conf, split=True, split_file=split_file, use_cache=True)


    for view in range(nviews):
        for train_type in all_models:
            for sndx in range(n_splits):
                exp_name = 'cv_split_{}'.format(sndx)
                run_trainining(exp_name,train_type,view,run_type)




def create_cv_dbs():
    assert False, 'This should not be used anymore'
    exp_name = 'apt_expt'
    assert gt_lbl is None
    common_conf = apt.create_conf(lbl_file, 0, exp_name, cache_dir, 'mdn')
    assert not os.path.exists(os.path.join(common_conf.cachedir, 'cv_split_fold_0.json'))
    alltrain, splits, split_files = apt.create_cv_split_files(common_conf, n_splits)
    for view in range(nviews):
        for tndx in range(len(all_models)):
            train_type = all_models[tndx]
            for split in range(n_splits):
                cur_split_file = os.path.join(common_conf.cachedir, 'cv_split_fold_{}.json'.format(split))
                conf = apt.create_conf(lbl_file, view, 'cv_split_{}'.format(split), cache_dir, train_type)
                conf.splitType = 'predefined'
                if 'deeplabcut' in train_type:
                    apt.create_deepcut_db(conf,split=True,use_cache=True,split_file=cur_split_file)
                elif 'leap' in train_type:
                    apt.create_leap_db(conf,split=True,use_cache=True,split_file=cur_split_file)
                else:
                    apt.create_tfrecord(conf,split=True,use_cache=True,split_file=cur_split_file)


## create incremental dbs

# assert False,'Are you sure?'

def create_incremental_dbs(do_split=False):
    import json
    import os
    exp_name = 'db_sz'
    lbl = h5py.File(lbl_file,'r')
    m_ndx = apt.to_py(lbl['preProcData_MD_mov'][()][0, :].astype('int'))
    t_ndx = apt.to_py(lbl['preProcData_MD_iTgt'][()][0, :].astype('int'))
    f_ndx = apt.to_py(lbl['preProcData_MD_frm'][()][0, :].astype('int'))

    n_mov = lbl['movieFilesAll'].shape[1]

    n_labels = m_ndx.shape[0]
    n_rounds = 8
    n_min = 5
    n_samples = np.logspace(np.log10(n_min),np.log10(n_labels),n_rounds).round().astype('int')

    rand_ndx = np.random.permutation(n_labels)
    lbl.close()
    for ndx, cur_s in enumerate(n_samples):
        cur = rand_ndx[:cur_s]
        splits = [[], []]
        for ex in range(n_labels):
            cur_m = m_ndx[ex]
            cur_t = t_ndx[ex]
            cur_f = f_ndx[ex]
            cur_info = [cur_m,cur_f, cur_t]
            if ex in cur:
                splits[0].append(cur_info)
            else:
                splits[1].append(cur_info)

        exp_name = '{}_randsplit_round_{}'.format(data_type,ndx)
        for view in range(nviews):
            for tndx in range(len(all_models)):
                train_type = all_models[tndx]
                #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                conf = create_conf_help(train_type, view, exp_name)
                mdn_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
                split_file= os.path.join(mdn_conf.cachedir,'randsplitinfo.json')
                if do_split and not os.path.exists(split_file):
                    def convert(o):
                        if isinstance(o, np.int64): return int(o)
                        raise TypeError
                    with open(split_file,'w') as f:
                        json.dump(splits,f,default=convert)
                        print("Wrote split file {}".format(split_file))

                conf.splitType = 'predefined'
                if 'deeplabcut' in train_type:
                    apt.create_deepcut_db(conf, split=True, split_file=split_file,use_cache=True)
                elif 'leap' in train_type:
                    apt.create_leap_db(conf, split=True, split_file=split_file, use_cache=True)
                elif train_type in ['mmpose','mdn_joint_fpn'] or train_type.startswith('mmpose'):
                    apt.create_coco_db(conf,split=True,split_file=split_file,use_cache=True)
                else:
                    apt.create_tfrecord(conf, split=True, split_file=split_file, use_cache=True)

def create_incremental_dbs_ma(pkl_file,ma_loc, do_split=False):
    # incremental dbs to match MA incremental
    import json
    import os
    import PoseTools as pt
    import multiResData

    exp_name = 'db_sz'
    lbl = h5py.File(lbl_file,'r')
    m_ndx = apt.to_py(lbl['preProcData_MD_mov'][()][0, :].astype('int'))
    t_ndx = apt.to_py(lbl['preProcData_MD_iTgt'][()][0, :].astype('int'))
    f_ndx = apt.to_py(lbl['preProcData_MD_frm'][()][0, :].astype('int'))

    n_mov = lbl['movieFilesAll'].shape[1]
    mov_files = multiResData.find_local_dirs(lbl_file,0)
    n_labels = m_ndx.shape[0]
    lbl.close()

    T = pt.json_load(ma_loc)

    inc_info = pt.pickle_load(pkl_file)
    sel = inc_info['sel']
    perm_lbls = inc_info['perm_lbls']
    n_samples = inc_info['n_samples']

    for ndx1, cur_s in enumerate(n_samples):
        cur = sel[ndx1]
        valt = []
        traint = []
        for ndx,curt in enumerate(T['locdata']):
            single_mov_ndx = mov_files.index(curt['mov'])
            if ndx in cur:
                traint.append([single_mov_ndx,curt['frm']-1])
            else:
                valt.append([single_mov_ndx,curt['frm']-1])

        splits = [[], []]
        for ex in range(n_labels):
            cur_m = m_ndx[ex]
            cur_t = t_ndx[ex]
            cur_f = f_ndx[ex]
            cur_info = [cur_m,cur_f, cur_t]
            if [cur_m,cur_f] in traint:
                splits[0].append(cur_info)
            else:
                splits[1].append(cur_info)

        exp_name = '{}_randsplit_round_{}'.format(data_type,ndx1)
        for view in range(nviews):
            for tndx in range(len(all_models)):
                train_type = all_models[tndx]
                #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                conf = create_conf_help(train_type, view, exp_name)
                mdn_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
                split_file= os.path.join(mdn_conf.cachedir,'randsplitinfo.json')
                if do_split or (not os.path.exists(split_file)):
                    def convert(o):
                        if isinstance(o, np.int64): return int(o)
                        raise TypeError
                    with open(split_file,'w') as f:
                        json.dump(splits,f,default=convert)
                        print("Wrote split file {}".format(split_file))

                conf.splitType = 'predefined'
                if 'deeplabcut' in train_type:
                    apt.create_deepcut_db(conf, split=True, split_file=split_file,use_cache=True)
                elif 'leap' in train_type:
                    apt.create_leap_db(conf, split=True, split_file=split_file, use_cache=True)
                elif train_type.startswith('mmpose') or train_type in ['mmpose','mdn_joint_fpn']:
                    apt.create_coco_db(conf,split=True,split_file=split_file,use_cache=True)
                else:
                    apt.create_tfrecord(conf, split=True, split_file=split_file, use_cache=True)


## create invidual animals dbs

def create_individual_animal_db_alice():
    import multiResData
    import random
    import json

    assert data_type == 'alice'
    fly1 = [0,3]
    fly2 = [2,8]
    view = 0
    train_type = 'mdn'
    exp_name = 'single_vs_many'
    n_train = 13 # fly1 and fly2 26 labeled examples
    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
    gt_db = '/nrs/branson/mayank/apt_cache/multitarget_bubble/gtdata/gtdata_view0.tfrecords'

    split_file1 = os.path.join(cache_dir,proj_name,'single_multiple_fly1.json')
    split_file2 = os.path.join(cache_dir,proj_name,'single_multiple_fly2.json')
    split_fileo = os.path.join(cache_dir,proj_name,'single_multiple_other.json')

    assert not os.path.exists(split_file1)

    ims,locs,info = multiResData.read_and_decode_without_session(gt_db,17,())
    id1 = []; id2 = []; ido = []
    for ndx in range(len(info)):
        if info[ndx][0] == fly1[0] and info[ndx][2] == fly1[1]:
            id1.append(ndx)
        elif info[ndx][0] == fly2[0] and info[ndx][2] == fly2[1]:
            id2.append(ndx)
        else:
            ido.append(ndx)

    random.shuffle(ido)
    random.shuffle(id1)
    random.shuffle(id2)
    ido_train = ido[:n_train]
    ido_test = ido[n_train:]
    id1_train = id1[:n_train]
    id1_test = id1[n_train:]
    id2_train = id2[:n_train]
    id2_test = id2[n_train:]

    split1 = [[],[]] # train on random half of fly 1
    split2 = [[],[]] # train of random half of fly 2
    split3 = [[],[]] # train on random flies other than fly1 and fly2

    for ndx in range(len(info)):
        if ndx in id1_train:
            split1[0].append(info[ndx])
        else:
            split1[1].append(info[ndx])
        if ndx in id2_train:
            split2[0].append(info[ndx])
        else:
            split2[1].append(info[ndx])
        if ndx in ido_train:
            split3[0].append(info[ndx])
        else:
            split3[1].append(info[ndx])


    with open(split_file1,'w') as f:
        json.dump(split1,f)
    with open(split_file2,'w') as f:
        json.dump(split2,f)
    with open(split_fileo,'w') as f:
        json.dump(split3,f)


    exp_name = 'single_vs_many_fly1'
    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
    conf.splitType = 'predefined'
    envs = multiResData.create_envs(conf, split=True)
    out_fns = [lambda data: envs[0].write(apt.tf_serialize(data)),
               lambda data: envs[1].write(apt.tf_serialize(data))]
    out_splits1 = apt.db_from_lbl(conf, out_fns, True, split_file1, on_gt=True)
    with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
        json.dump(out_splits1, f)
    envs[0].close()
    envs[1].close()

    exp_name = 'single_vs_many_fly2'
    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
    conf.splitType = 'predefined'

    envs = multiResData.create_envs(conf, split=True)
    out_fns = [lambda data: envs[0].write(apt.tf_serialize(data)),
               lambda data: envs[1].write(apt.tf_serialize(data))]
    out_splits2 = apt.db_from_lbl(conf, out_fns, True, split_file2, on_gt=True)
    with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
        json.dump(out_splits2, f)
    envs[0].close()
    envs[1].close()

    exp_name = 'single_vs_many_other'
    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
    conf.splitType = 'predefined'
    envs = multiResData.create_envs(conf, split=True)
    out_fns = [lambda data: envs[0].write(apt.tf_serialize(data)),
               lambda data: envs[1].write(apt.tf_serialize(data))]
    out_splits3 = apt.db_from_lbl(conf, out_fns, True, split_fileo, on_gt=True)
    with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
        json.dump(out_splits3, f)
    envs[0].close()
    envs[1].close()


def create_run_individual_animal_dbs_stephen(skip_db = True, run_type='status',dstr=PoseTools.datestr()):
    # assert False, 'This should be run from run_apt_expts'
    info_file = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190416.mat'

    data_info = h5py.File(info_file,'r')
    assert data_type == 'stephen'
    assert trn_flies is not None
    train_type = 'mdn_joint_fpn' #'mdn'
    n_sel = 50

    conf = apt.create_conf(lbl_file,0,'dummy',cache_dir,train_type)
    lbl_movies, _ = multiResData.find_local_dirs(conf)
    in_movies = [PoseTools.read_h5_str(data_info[k]) for k in data_info['trnmovies'][0,:]]
    assert lbl_movies == in_movies

    fly_ids = data_info['trnmidx2flyid'].value.astype('int')
    label_info = get_label_info(conf)

    for cur_fly in trn_flies:
        cur_fly_movies = [ix for ix,j in enumerate(fly_ids[0,:]) if j==cur_fly]
        fly_train_info = [j for j in label_info if j[0] in cur_fly_movies]
        assert len(fly_train_info) > 50
        sel_train = random.sample(fly_train_info,n_sel)
        sel_val = list(set(label_info)-set(sel_train))
        assert len(label_info) == len(sel_train) + len(sel_val)

        cur_split = [sel_train,sel_val]
        exp_name = 'train_fly_{}'.format(cur_fly)
        cur_split_file = os.path.join(cache_dir,proj_name,exp_name) + '.json'
        if not skip_db:
            assert not os.path.exists(cur_split_file)
            with open(cur_split_file,'w') as f:
                json.dump(cur_split,f)

        # create the dbs
        for view in range(nviews):
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            conf.splitType = 'predefined'
            if not skip_db:
                apt.create_tfrecord(conf, split=True, split_file=cur_split_file, use_cache=True)
            run_trainining(exp_name,train_type,view,run_type,dstr=dstr)

    # one experiment with random labels
    sel_train = random.sample(label_info,n_sel)
    sel_val = list(set(label_info)-set(sel_train))
    assert len(label_info) == len(sel_train) + len(sel_val)

    cur_split = [sel_train,sel_val]
    exp_name = 'train_fly_random'
    cur_split_file = os.path.join(cache_dir,proj_name,exp_name)
    if not skip_db:
        assert not os.path.exists(cur_split_file)
        with open(cur_split_file,'w') as f:
            json.dump(cur_split,f)

    # create the dbs
    for view in range(nviews):
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        conf.splitType = 'predefined'
        if not skip_db:
            apt.create_tfrecord(conf, split=True, split_file=cur_split_file, use_cache=True)
        run_trainining(exp_name,train_type,view,run_type,dstr=dstr)




##   ----------- TRAINING  -----------------


def run_normal_training(expname = 'apt_expt',
                        run_type = 'status',
                        **kwargs
                        ):

    kwargs['save_time'] = 20 # save every 20 min

    results = {}
    for train_type in all_models:
        for view in range(nviews):
            key = "{}_vw{}".format(train_type, view)
            results[key] = run_trainining(expname, train_type, view, run_type, **kwargs)

    return results


def run_cv_training(run_type='status'):
    assert False, 'This should not be used anymore'

    common_conf = {}
    common_conf['dl_steps'] = 40000
    common_conf['maxckpt'] = 3

    assert gt_lbl is None
    for view in range(nviews):
        for tndx in range(len(all_models)):
            train_type = all_models[tndx]
            for split in range(n_splits):
                exp_name = 'cv_split_{}'.format(split)
                common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file, exp_name, cache_dir)
                end_cmd = 'train -skip_db -use_cache'
                cmd_opts = {}
                cmd_opts['type'] = train_type
                cmd_opts['view'] = view + 1
                conf_opts = common_conf.copy()
                # conf_opts.update(other_conf[conf_id])
                conf_opts['save_step'] = conf_opts['dl_steps'] // 10
                if data_type in ['brit0' ,'brit1']:
                    if train_type == 'unet':
                        conf_opts['batch_size'] = 2
                    else:
                        conf_opts['batch_size'] = 4
                if data_type in ['romain']:
                    if train_type in ['mdn']:
                        conf_opts['batch_size'] = 2
                    elif train_type in ['unet']:
                        conf_opts['batch_size'] = 1
                    else:
                        conf_opts['batch_size'] = 4
                if op_af_graph is not None:
                    conf_opts['op_affinity_graph'] = op_af_graph

                if len(conf_opts) > 0:
                    conf_str = ' -conf_params'
                    for k in conf_opts.keys():
                        conf_str = '{} {} {} '.format(conf_str, k, conf_opts[k])
                else:
                    conf_str = ''

                opt_str = ''
                for k in cmd_opts.keys():
                    opt_str = '{} -{} {} '.format(opt_str, k, cmd_opts[k])

                cur_cmd = common_cmd + conf_str + opt_str + end_cmd
                cmd_name = '{}_view{}_{}_{}'.format(data_type, view, exp_name, train_type)
                if run_type == 'submit':
                    print(cur_cmd)
                    print()
                    run_jobs(cmd_name, cur_cmd)
                elif run_type == 'status':
                    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                    check_train_status(cmd_name, conf.cachedir)


## DLC augment vs no augment ---- TRAINING ----

# assert False,'Are you sure?'

def run_dlc_augment_training(run_type = 'status'):
    # run_type = 'submit'; redo = False
    # gpu_model = 'TeslaV100_SXM2_32GB'
    train_type = 'deeplabcut'
    common_conf['dl_steps'] = 100000

    other_conf = [{'dlc_augment':True},{'dlc_augment':False,'dl_steps':300000}]
    cmd_str = ['dlc_aug','dlc_noaug']
    cache_dir = '/nrs/branson/mayank/apt_cache'
    # exp_name = 'apt_expt'

    use_round = dlc_aug_use_round
    exp_name = '{}_randsplit_round_{}'.format(data_type,use_round)

    for view in range(nviews):

        for conf_id in range(len(other_conf)):

            common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file, exp_name,cache_dir)
            end_cmd = 'train -skip_db -use_cache'
            cmd_opts = {}
            cmd_opts['type'] = train_type
            cmd_opts['train_name'] = cmd_str[conf_id]
            cmd_opts['view'] = view + 1
            conf_opts = common_conf.copy()
            conf_opts.update(other_conf[conf_id])
            conf_opts['save_step'] = conf_opts['dl_steps']//20
            if data_type == 'stephen':
                conf_opts['batch_size'] = 4

            if len(conf_opts) > 0:
                conf_str = ' -conf_params'
                for k in conf_opts.keys():
                    conf_str = '{} {} {} '.format(conf_str,k,conf_opts[k])
            else:
                conf_str = ''

            opt_str = ''
            for k in cmd_opts.keys():
                opt_str = '{} -{} {} '.format(opt_str,k,cmd_opts[k])

            cur_cmd = common_cmd + conf_str + opt_str + end_cmd

            cmd_name = '{}_view{}_{}_{}'.format(data_type,view,cmd_str[conf_id],use_round)
            if run_type == 'submit':
                print(cur_cmd)
                print()
                run_jobs(cmd_name,cur_cmd)
            elif run_type == 'status':
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                check_train_status(cmd_name,conf.cachedir,cmd_str[conf_id])


def train_deepcut_orig(run_type='status'):
    train_type = 'deeplabcut'
    cache_dir = '/nrs/branson/mayank/apt_cache'
    dlc_cmd = '/groups/branson/bransonlab/mayank/apt_expts/deepcut/pose-tensorflow/train.py'

    for view in range(nviews):
        for round in range(8):
            exp_name = '{}_randsplit_round_{}'.format(data_type, round)
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            cmd_name = '{}_view{}_dlcorig_{}'.format(data_type, view, round)
            if run_type == 'submit':
                apt_expts.create_deepcut_cfg(conf)
                args = easydict.EasyDict()
                args.lbl_file = lbl_file
                args.split_type = None
                run_dir = conf.cachedir
                cur_cmd = dlc_cmd

                print(cur_cmd)
                print()
                run_jobs(cmd_name,cur_cmd,run_dir=run_dir)
            elif run_type == 'status':
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                check_train_status(cmd_name,conf.cachedir,'snapshot')


def train_leap_orig(run_type='status',skip_db=True):
    train_type = 'leap'
    cache_dir = '/nrs/branson/mayank/apt_cache'
    run_dir = '/groups/branson/bransonlab/mayank/apt_expts/leap'

    for view in range(nviews):
        for round in range(8):
            exp_name = '{}_randsplit_round_{}_orig'.format(data_type, round)
            exp_name1 = '{}_randsplit_round_{}'.format(data_type, round)
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            conf1 = apt.create_conf(lbl_file, view, exp_name1, cache_dir, train_type)
            cmd_name = '{}_view{}_leaporig_{}'.format(data_type, view, round)
            if run_type == 'submit':
                in_file = os.path.join(conf1.cachedir, 'leap_train.h5')
                out_file = os.path.join(conf.cachedir, 'leap_train.h5')
                if not skip_db:
                    ii = h5py.File(in_file,'r')
                    ims = ii['box'][:]
                    locs = ii['joints'][:]
                    eid = ii['exptID'][:]
                    frid = ii['framesIdx'][:]
                    tid = ii['trxID'][:]
                    ii.close()

                    hmaps = PoseTools.create_label_images(locs, conf.imsz[:2], 1, 5)
                    hmaps = (hmaps + 1) / 2  # brings it back to [0,1]

                    hf = h5py.File(out_file, 'w')
                    hf.create_dataset('box', data=ims)
                    hf.create_dataset('confmaps', data=hmaps)
                    hf.create_dataset('joints', data=locs)
                    hf.create_dataset('exptID', data=eid)
                    hf.create_dataset('framesIdx', data=frid)
                    hf.create_dataset('trxID', data=tid)
                    hf.close()

                cur_cmd = 'leap/training_MK.py {}'.format(out_file)

                print(cur_cmd)
                print()
                run_jobs(cmd_name,cur_cmd,run_dir=run_dir)
            elif run_type == 'status':
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                check_train_status(cmd_name,conf.cachedir,'asflk')


def train_dpk_orig(expname='dpkorig',
                   run_type='status',
                   exp_note='DPK_origstyle',
                   dpk_train_style='dpk',  # or 'apt'
                   **kwargs
                   ):
    global all_models
    all_models = ['dpk']
    run_normal_training(expname=expname,
                        run_type=run_type,
                        exp_note=exp_note,
                        dpk_train_style=dpk_train_style, #dpk_use_augmenter=0,
                        **kwargs
                        )



def train_no_pretrained(run_type='status'):
    train_type = 'mdn'

    common_conf['brange'] = '\(-0.2,0.2\)'
    common_conf['crange'] = '\(0.7,1.3\)'

    for view in range(nviews):
        for round in range(8):
            exp_name = '{}_randsplit_round_{}'.format(data_type, round)
            run_trainining(exp_name,train_type,view,run_type,train_name='no_pretrained',use_pretrained_weights=False,dl_steps=100000,maxckpt=30,save_step=5000)


def run_mdn_no_unet(run_type = 'status'):

    common_conf['dl_steps'] = 100000
    common_conf['maxckpt'] = 20
    common_conf['save_time'] = 20 # save every 20 min
    common_conf['mdn_use_unet_loss'] = False

    train_type = 'mdn'
    for view in range(nviews):
        run_trainining('apt_expt',train_type,view, run_type,train_name='no_unet')


def run_incremental_training(run_type='status', viewidxs=None, roundidxs=None,
                             modelsrun=None, **kwargs):
    # Expt where we find out how training error changes with amount of training data

    n_rounds = 8

    if viewidxs is None:
        viewidxs = range(nviews)

    if roundidxs is None:
        roundidxs = range(n_rounds)

    if modelsrun is None:
        modelsrun = all_models

#    info = []
    for view in viewidxs:
#        r_info = []
        for ndx in roundidxs:
            exp_name = '{}_randsplit_round_{}'.format(data_type,ndx)
            cur_info = {}
            for train_type in modelsrun:
                expnote = 'incremental training'
                run_trainining(exp_name, train_type, view, run_type, exp_note=expnote, **kwargs)
#            r_info.append(cur_info)
#        info.append(r_info)


## SINGLE animals vs multiple ---- TRAINING ----

def run_single_animal_training(run_type = 'status'):

    assert data_type == 'alice'
    view = 0
    train_type = 'mdn'

    gpu_model = 'GeForceRTX2080Ti'
    sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

    exp_names = ['single_vs_many_fly1', 'single_vs_many_fly2', 'single_vs_many_other']

    cur_info = {}
    for exp_name in exp_names:

        common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file, exp_name, cache_dir)
        end_cmd = 'train -skip_db -use_cache'
        cmd_opts = {}
        cmd_opts['type'] = train_type
        cmd_opts['view'] = view + 1
        conf_opts = common_conf.copy()
        # conf_opts.update(other_conf[conf_id])
        conf_opts['save_step'] = conf_opts['dl_steps'] // 10
        if data_type == 'stephen':
            conf_opts['batch_size'] = 4
        if op_af_graph is not None:
            conf_opts['op_affinity_graph'] = op_af_graph

        if len(conf_opts) > 0:
            conf_str = ' -conf_params'
            for k in conf_opts.keys():
                conf_str = '{} {} {} '.format(conf_str, k, conf_opts[k])
        else:
            conf_str = ''

        opt_str = ''
        for k in cmd_opts.keys():
            opt_str = '{} -{} {} '.format(opt_str, k, cmd_opts[k])

        cur_cmd = common_cmd + conf_str + opt_str + end_cmd
        cmd_name = '{}_view{}_{}'.format(exp_name, view, train_type)

        if run_type == 'submit':
            print(cur_cmd)
            print()
            run_jobs(cmd_name, cur_cmd)


##  ###################### GT DBs

def create_gt_db():
    lbl = h5py.File(lbl_file,'r')
    proj_name = apt.read_string(lbl['projname'])
    lbl.close()
    # print(gt_name)
    for view in range(nviews):
        conf = apt.create_conf(gt_lbl, view, 'apt_expt', cache_dir, 'mdn')
        gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}{}.tfrecords'.format(view,gt_name))
        print(gt_file)
        apt.create_tfrecord(conf,False,None,False,True,[gt_file])


## ######################  RESULTS


def get_normal_results(exp_name='apt_expt',  # can be dict of train_type->exp_name
                       train_name='deepnet',
                       use_exp_name_save_file=False,  # if True, include exp_name in savefiles
                       gt_name_use_output=None,  # if supplied, use instead of gt_name
                       last_model_only=False,
                       classify_fcn='classify_db2',
                       classify_return_ims=False,
                       queue='gpu_rtx',
                       dstr = PoseTools.datestr(),
                       force_recomp = False,
                       **kwargs):

    if gt_name_use_output is None:
        gt_name_use_output = gt_name

    all_view = []
    gpu_str = '_tesla' if queue == 'gpu_tesla' else ''
    dstr = '_' + dstr if dstr else ''
    train_name_dstr = train_name + gpu_str + dstr

    for view in range(nviews):
        out_exp = {}

        gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}{}.tfrecords'.format(view,gt_name))

        for train_type in all_models:
            mmstr = f'_{train_type.split("_")[1]}' if train_type.startswith('mmpose') else ''
            tstr = train_name_dstr + mmstr

            exp_name_use = exp_name[train_type] if isinstance(exp_name, dict) else exp_name
            conf = create_conf_help(train_type, view, exp_name_use, **kwargs)
            conf.batch_size = 1
            # compare conf to conf on disk
            cffile = os.path.join(conf.cachedir, 'conf.pickle')
            if os.path.exists(cffile):
                with open(cffile, 'rb') as fh:
                    conf0 = pickle.load(fh,encoding='latin1')
                conf0 = conf0['conf']
            elif train_type == 'deeplabcut_orig':
                conf0 = conf
            else:
                # cffile = os.path.join(conf.cachedir, 'traindata')
                cffile = get_traindata_file_flexible(conf.cachedir, tstr, conf.expname)
                assert os.path.exists(cffile), "Cant find conf on disk"
                with open(cffile, 'rb') as fh:
                    conf0 = pickle.load(fh,encoding='latin1')
                conf0 = conf0[-1]

            print("## View {} Net {} Conf Compare (disk vs now)".format(view,train_type))
            util.dictdiff(vars(conf0), vars(conf))

            assert not conf.normalize_img_mean

            files = get_model_files(conf,tstr,1000,net=train_type)
            print('view {}, net {}. Your models are:'.format(view, train_type))
            print(files)

            out_file = os.path.join(conf.cachedir, tstr + '_results{}.p'.format(gt_name_use_output))
            recomp = do_recompute(out_file,files) or force_recomp
            if recomp:
                if last_model_only:
                    files = files[-1:]
                mt = train_type if not train_type.startswith('mmpose') else 'mmpose'
                mdn_out = apt_expts.classify_db_all(conf,gt_file,files,mt, name=tstr, classify_fcn=classify_fcn, return_ims=classify_return_ims,ignore_hmaps=True)
                with open(out_file,'wb') as f:
                    pickle.dump([mdn_out,files],f)
                print("Wrote {}".format(out_file))
            else:
                A = PoseTools.pickle_load(out_file)
                print("Loaded {}".format(out_file))
                mdn_out = A[0]

            out_exp[train_type] = mdn_out

        ex_im, ex_loc = decode_db(gt_file,conf)
        all_view.append([out_exp,ex_im,ex_loc])

    for ndx,out_exp in enumerate(all_view):
        plot_results(out_exp[0])
        plot_hist(out_exp,ps=[50,75,90,95,98])

        save_file = os.path.join(results_dir, '{}_{}_{}_view{}_time{}'.format(data_type, exp_name, train_name_dstr, ndx, gt_name_use_output)) \
            if use_exp_name_save_file \
            else os.path.join(results_dir, '{}_{}_view{}_time{}'.format(data_type, train_name_dstr, ndx, gt_name_use_output))

        save_filep = save_file+".p"
        with open(save_filep, 'wb') as f:
            pickle.dump(out_exp[0], f)
            print("wrote {}".format(save_filep))
        save_mat(out_exp[0], save_file)


def get_normal_pred_speeds(exp_name='apt_expt',  # can be dict of train_type->exp_name
                           train_name='deepnet',
                           batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
                           **kwargs):

    times_dict = {}
    for view in range(nviews):
        gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}{}.tfrecords'.format(view,gt_name))
        for train_type in all_models:

            exp_name_use = exp_name[train_type] if isinstance(exp_name, dict) else exp_name
            conf = create_conf_help(train_type, view, exp_name_use, **kwargs)

            # compare conf to conf on disk
            cffile = os.path.join(conf.cachedir, 'conf.pickle')
            if os.path.exists(cffile):
                with open(cffile, 'rb') as fh:
                    conf0 = pickle.load(fh,encoding='latin1')
                conf0 = conf0['conf']
            else:
                cffile = os.path.join(conf.cachedir, 'traindata')
                assert os.path.exists(cffile), "Cant find conf on disk"
                with open(cffile, 'rb') as fh:
                    conf0 = pickle.load(fh,encoding='latin1')
                conf0 = conf0[-1]

            print("## View {} Net {} Conf Compare (disk vs now)".format(view,train_type))
            util.dictdiff(vars(conf0), vars(conf))

            assert not conf.normalize_img_mean

            files = get_model_files(conf, train_name)

            print('view {}, net {}. Your models are:'.format(view, train_type))
            print(files)
            afiles = [f.replace('.index', '') for f in files]
            afiles = afiles[-1:]  # last model only

            t_overall = []
            t_read = []
            t_pred = []
            t_pred_inner = []
            dpredmaxes = []
            for ib, bsize in enumerate(batch_sizes):
                conf.batch_size = bsize

                tmr_overall = util.Timer()
                tmr_read = util.Timer()
                tmr_pred = util.Timer()
                tmr_pred_inner = util.Timer()
                with tmr_overall:
                    out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,
                                                    name=train_name,
                                                    classify_fcn='classify_db2',
                                                    timer_read=tmr_read,
                                                    timer_pred=tmr_pred,
                                                    timer_pred_inner=tmr_pred_inner)

                assert(len(out) == 1)
                res = classify_db_all_res_to_dict(out[0])

                if ib == 0:
                    res0 = res
                    dpredmax = 0.
                else:
                    assert ((res['info'] == res0['info']).all())
                    assert ((res['lbl'] == res0['lbl']).all())
                    dpred = res['pred'] - res0['pred']
                    dpredmax = np.max(np.abs(dpred))

                t_overall.append(tmr_overall._recorded_times)
                t_read.append(tmr_read._recorded_times)
                t_pred.append(tmr_pred._recorded_times)
                t_pred_inner.append(tmr_pred_inner._recorded_times)
                dpredmaxes.append(dpredmax)

            if train_type not in times_dict:
                times_dict[train_type] = {}
            times_dict[train_type][view] = {
                't_overall': t_overall,
                't_read': t_read,
                't_pred': t_pred,
                't_pred_inner': t_pred_inner,
                'preddiffmaxs': dpredmaxes}

    return times_dict, batch_sizes


def save_pred_speed_output_to_mat(dt, bsizes, matfile):
    dtsave = {}
    for net in dt:
        dtsave[net] = {}
        for vw in dt[net].keys():
            vwstr = 'vw{}'.format(vw)
            dstats = dt[net][vw]
            FLDSSIMPLE = ['preddiffmaxs', 't_overall']
            FLDSLISTOFARRS = ['t_pred', 't_pred_inner', 't_read']
            for k in FLDSSIMPLE:
                dstats[k] = np.array(dstats[k])
            for k in FLDSLISTOFARRS:
                dstats[k] = [np.array(x) for x in dstats[k]]
            dtsave[net][vwstr] = dstats
    dtsave['bsizes'] = np.array(bsizes)

    hdf5storage.savemat(matfile, dtsave)
    print("Saved {}".format(matfile))


def get_mdn_no_unet_results():
## Normal Training  ------- RESULTS -------
    cache_dir = '/nrs/branson/mayank/apt_cache'
    exp_name = 'apt_expt'

    all_view = []

    for view in range(nviews):
        out_exp = {}

        gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}{}.tfrecords'.format(view,gt_name))
        train_type = 'mdn'
        for train_name in ['deepnet','no_unet']:
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            # if data_type == 'stephen' and train_type == 'mdn':
            #     conf.mdn_use_unet_loss = False
            if op_af_graph is not None:
                conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
            conf.normalize_img_mean = False

            files = get_model_files(conf,train_name,1000)
            out_file = os.path.join(conf.cachedir,'{}_no_unet_results.p'.format(train_name))
            recomp = do_recompute(out_file,files)

            if recomp:
                if train_name == 'deepnet':
                    td_file = os.path.join(conf.cachedir,'traindata')
                else:
                    td_file = os.path.join(conf.cachedir,conf.expname + '_' + train_name + '_traindata')
                td = PoseTools.pickle_load(td_file)
                cur_conf = td[1]
                mdn_out = apt_expts.classify_db_all(cur_conf,gt_file,files,train_type,name=train_name)
                with open(out_file,'w') as f:
                    pickle.dump([mdn_out,files],f)
            else:
                A = PoseTools.pickle_load(out_file)
                mdn_out = A[0]

            out_exp[train_name] = mdn_out

        ex_im, ex_loc = decode_db(gt_file,conf)
        all_view.append([out_exp,ex_im,ex_loc])

    for ndx,out_exp in enumerate(all_view):
        plot_results(out_exp[0])
        plot_hist(out_exp)
        save_mat(out_exp[0],os.path.join(cache_dir,'{}_view{}_no_unet{}'.format(data_type,ndx,gt_name)))


## DLC AUG vs no aug --- RESULTS -----

def get_dlc_results(redo=False):
    cmd_str = ['dlc_aug','dlc_noaug','snapshot']
    # exp_name = 'apt_expt'
    use_round = dlc_aug_use_round
    exp_name = '{}_randsplit_round_{}'.format(data_type,use_round)

    train_type = 'deeplabcut'
    all_view = []
    for view in range(nviews):
        dlc_exp = {}

        gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}.tfrecords'.format(view))

        for conf_id in range(len(cmd_str)):
            train_name=cmd_str[conf_id]
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            conf.normalize_img_mean = False
            if op_af_graph is not None:
                conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
            files = get_model_files(conf,train_name,1000)
            files = files[-1:]
            out_file = os.path.join(conf.cachedir,train_name + '_results.p')
            if redo:
                recomp = True
            else:
                recomp = do_recompute(out_file, files)

            if recomp:
                mdn_out = apt_expts.classify_db_all(conf,gt_file, files,train_type,name=train_name)
                with open(out_file,'w') as f:
                    pickle.dump([mdn_out,files],f)
            else:
                A = PoseTools.pickle_load(out_file)
                mdn_out = A[0]

            dlc_exp[train_name] = mdn_out

        ex_im,ex_loc = decode_db(gt_file,conf)
        all_view.append([dlc_exp,ex_im,ex_loc])
        # all_view.append(dlc_exp)

    for ndx, dlc_exp in enumerate(all_view):
        # plot_results(dlc_exp[0])
        plot_hist(dlc_exp,[50,70,90,95])
        out_file = os.path.join(cache_dir,'{}_view{}_round{}_dlc'.format(data_type,ndx,use_round))
        save_mat(dlc_exp[0],out_file)


def get_leap_results():
    cmd_str = ['deepnet','weights']
    train_type = 'leap'
    for use_round in range(4,8):
        all_view = []

        for view in range(nviews):
            dlc_exp = {}
            gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}.tfrecords'.format(view))

            exp_name = '{}_randsplit_round_{}'.format(data_type, use_round)
            exp_name += '_orig'
            train_name='weights'
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            files = get_model_files(conf,train_name,1000)
            ts = [os.path.getmtime(f) for f in files]
            tmax = ts[-1]-ts[0]

            for conf_id in range(len(cmd_str)):
                exp_name = '{}_randsplit_round_{}'.format(data_type, use_round)
                if cmd_str[conf_id] == 'weights':
                    exp_name += '_orig'
                train_name=cmd_str[conf_id]
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                if op_af_graph is not None:
                    conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
                files = get_model_files(conf,train_name,1000)
                tt = [os.path.getmtime(f) for f in files]
                td = np.array([t-tt[0] for t in tt])
                t_closest = np.argmin(np.abs(td-tmax))
                files = [files[0],files[t_closest]]

                out_file = os.path.join(conf.cachedir,train_name + '_results.p')
                recomp = do_recompute(out_file,files)

                if recomp:
                    afiles = [f.replace('.index', '') for f in files]
                    mdn_out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,name=train_name)
                    with open(out_file,'w') as f:
                        pickle.dump([mdn_out,files],f)
                else:
                    A = PoseTools.pickle_load(out_file)
                    mdn_out = A[0]

                dlc_exp[train_name] = mdn_out
            ex_im,ex_loc = decode_db(gt_file,conf)
            all_view.append([dlc_exp,ex_im,ex_loc])
            # all_view.append(dlc_exp)

        for ndx, dlc_exp in enumerate(all_view):
            # plot_results(dlc_exp[0])
            plot_hist(dlc_exp,[50,70,90,95])
            out_file = os.path.join(cache_dir,'{}_view{}_round{}_leap'.format(data_type,ndx,use_round))
            save_mat(dlc_exp[0],out_file)


## incremental training -- RESULTS ---
def get_incremental_results(dstr=PoseTools.datestr(),queue='gpu_rtx'):
    n_rounds = 8
    all_res = []
    train_name = 'deepnet'
    gpu_str = '_tesla' if queue == 'gpu_tesla' else ''
    train_name_dstr = train_name + gpu_str + '_' + dstr
    all_view = []

    for view in range(nviews):
        out_exp = {}
        gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))
        inc_exp = {}
        for train_type in all_models:
            # if data_type == 'alice' and train_type == 'leap':
            #     continue
            r_files = []
            train_size = []
            mmstr = ('_'+train_type.split('_')[1]) if train_type.startswith('mmpose') else ''
            for ndx in range(n_rounds):
                exp_name = '{}_randsplit_round_{}'.format(data_type, ndx)
                #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                conf = create_conf_help(train_type, view, exp_name)
                conf.batch_size = 1
                split_data = PoseTools.json_load(os.path.join(conf.cachedir,'splitdata.json'))
                train_size.append(len(split_data[0]))
                files = get_model_files(conf,train_name_dstr+mmstr,10)
                aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
                if len(files)>0 and aa[-1]>0:
                    r_files.append(files[-1])
                else:
                    print('MISSING!!!! MISSING!!!! {} {}'.format(train_type,ndx))

            out_file = os.path.join(conf.cachedir,train_name_dstr+mmstr + '_results.p')
            recomp = do_recompute(out_file,r_files)

            if recomp:
                ttype = train_type.split('_')[0] if train_type.startswith('mmpose') else train_type
                mdn_out = apt_expts.classify_db_all(conf,gt_file,r_files,ttype,name=train_name_dstr)
                with open(out_file,'wb') as f:
                    pickle.dump([mdn_out,r_files],f)
            else:
                A = PoseTools.pickle_load(out_file)
                mdn_out = A[0]

            for x, a in enumerate(mdn_out):
                tndx = int(int(re.search('randsplit_round_(\d)',r_files[x])[1]))
                a[-1] = train_size[tndx]
                a[2] = np.array(a[2])
            mdn_out.insert(0,mdn_out[0])
            inc_exp[train_type] = mdn_out
        all_view.append(inc_exp)

    for ndx,ii in enumerate(all_view):
        plot_results(ii,ylim=15)
        save_mat(ii,os.path.join(results_dir,'{}_{}_view{}_trainsize'.format(data_type,train_name_dstr, ndx,)))


def get_no_pretrained_results():
    n_rounds = 8
    all_res = []

    for ndx in range(n_rounds):

        all_view = []
        for view in range(nviews):
            predtrained_exp = {}
            for train_name in ['deepnet', 'no_pretrained']:

                gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))
                train_type = 'mdn'
                train_size = []
                exp_name = '{}_randsplit_round_{}'.format(data_type, ndx)
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                split_data = PoseTools.json_load(os.path.join(conf.cachedir,'splitdata.json'))
                train_size.append(len(split_data[0]))
                if op_af_graph is not None:
                    conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
                files = get_model_files(conf,train_name,10)
                out_file = os.path.join(conf.cachedir,train_name + '_pretraining_results.p')
                recomp = do_recompute(out_file,files)

                if recomp:
                    mdn_out = apt_expts.classify_db_all(conf,gt_file,files,train_type,name=train_name)
                    with open(out_file,'w') as f:
                        pickle.dump([mdn_out,files],f)
                else:
                    A = PoseTools.pickle_load(out_file)
                    mdn_out = A[0]


                # in enumerate(mdn_out):
                #     a[-1] = train_size[x]
                #     a[2] = np.array(a[2])
                # mdn_out.insert(0,mdn_out[0])
                predtrained_exp[train_name] = mdn_out

            ex_im, ex_loc = decode_db(gt_file,conf)
            all_view.append([predtrained_exp, ex_im, ex_loc])

            for ndx_i,ii in enumerate(all_view):
                plot_results(ii[0],ylim=15)
                plot_hist(ii)
                save_mat(ii[0],os.path.join(cache_dir,'{}_view{}_round{}_pretrained'.format(data_type,ndx_i,ndx)))


## CV Results

def get_cv_results(num_splits=None,
                   db_from_mdn_dir=False,
                   exp_name_pfix='',  # prefix for exp_name. can be map from train_type->exp_name_pfix
                   split_idxs=None,
                   queue='gpu_rtx8000',
                   dstr = PoseTools.get_datestr(),
                   ptiles_plot=[50,75,90,95,97]):
    train_name = 'deepnet'
    gpu_str = '_tesla' if 'gpu_tesla' in queue else ''
    train_name = train_name + gpu_str + '_' + dstr
    if num_splits == None:
        print("Reading splits from {}".format(cv_info_file))
        data_info = h5py.File(cv_info_file, 'r')
        cv_info = apt.to_py(data_info['cvi'][()].flatten().astype('int'))
        n_splits = max(cv_info) + 1
        num_splits = n_splits

    if split_idxs is not None:
        assert all([x in range(num_splits) for x in split_idxs])
    else:
        split_idxs = range(num_splits)

    assert gt_lbl is None
    all_view = []
    for view in range(nviews):
        out_exp = {}
        for tndx in range(len(all_models)):
            train_type = all_models[tndx]

            ex_im = None
            ex_loc = None
            out_split = None
            for split in split_idxs:
                pfix_use = exp_name_pfix[train_type] if isinstance(exp_name_pfix, dict) else exp_name_pfix
                exp_name = '{}cv_split_{}'.format(pfix_use, split)
                pfix_use_mdn = exp_name_pfix['mdn'] if isinstance(exp_name_pfix, dict) else exp_name_pfix
                exp_name_mdn = '{}cv_split_{}'.format(pfix_use_mdn, split)

                # confs only used for .cachedir etc; actual conf used in tracking is loaded from
                # traindata. so optional kwargs etc unnec

                #mdn_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
                mdn_conf = create_conf_help('mdn_joint_fpn', view, exp_name_mdn, quiet=True,queue=queue)
                #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                conf = create_conf_help(train_type, view, exp_name, quiet=True,queue=queue)
                conf.batch_size = 1

                mmstr = ''
                ttype = train_type
                if train_type.startswith('mmpose'):
                    mmstr = '_' + train_type.split('_')[1]
                    ttype = 'mmpose'


                files = get_model_files(conf,train_name+mmstr,10)
                files = files[-1:]
                print(files)

                out_file = os.path.join(conf.cachedir,train_name+mmstr + '_results.p')
                recomp = do_recompute(out_file,files)
                if recomp:
                    print("Recomputing {}, vw{}, {}".format(train_type, view, split))
                    db_file = os.path.join(mdn_conf.cachedir,'val_TF.json') if \
                                db_from_mdn_dir else os.path.join(conf.cachedir, 'val_TF.tfrecords')
                    tfile = get_traindata_file_flexible(conf.cachedir, train_name+mmstr,conf.expname)
                    print("Loading traindata file {}".format(tfile))
                    tdata = PoseTools.pickle_load(tfile)
                    # latter case LEAP apparently
                    tdataconf = tdata[1] if isinstance(tdata,list) else tdata
                    util.dictdiff(vars(tdataconf), vars(conf))

                    # mdn_out = apt_expts.classify_db_all(conf,db_file,files,ttype,name=train_name)
                    mdn_out = apt_expts.classify_db_all(conf,db_file,files,ttype,name=train_name)
                    with open(out_file,'wb') as f:
                        pickle.dump([mdn_out,files],f)
                    print("Wrote {}".format(out_file))
                else:
                    A = PoseTools.pickle_load(out_file)
                    mdn_out = A[0]
                    for m in mdn_out:
                        m[2] = np.array(m[2])
                if len(mdn_out) == 1:
                    mdn_out.append(mdn_out[0][:])
                    mdn_out.append(mdn_out[0][:])
                    for ix in range(len(mdn_out)):
                        mdn_out[ix][-1] = ix
                if out_split is None:
                    out_split = mdn_out
                else:
                    for mndx in range(min(len(mdn_out),len(out_split))):
                        out_split[mndx][0] = np.append(out_split[mndx][0],mdn_out[mndx][0],axis=0)
                        out_split[mndx][1] = np.append(out_split[mndx][1],mdn_out[mndx][1],axis=0)
                        out_split[mndx][2] = np.append(out_split[mndx][2],mdn_out[mndx][2],axis=0)

                if ex_im is None:
                    db_file = os.path.join(mdn_conf.cachedir, 'val_TF.tfrecords') if \
                        db_from_mdn_dir else os.path.join(conf.cachedir, 'val_TF.tfrecords')
                    H = multiResData.read_and_decode_without_session(db_file,mdn_conf)
                    ex_ims = np.array(H[0][0])
                    ex_locs = np.array(H[1][0])
            out_exp[train_type] = out_split
        all_view.append([out_exp,ex_ims,ex_locs])

    for ndx,out_exp in enumerate(all_view):
        cmap = PoseTools.get_cmap(5,'cool')
        # cmap = np.array([[0.5200,         0,         0],
        #         [1.0000,    0.5200,         0],
        #         [0.4800,    1.0000,    0.5200],
        #         [0,    0.5200,    1.0000],
        #         [0,         0,    0.5200]])
        plot_hist(out_exp,ps=ptiles_plot,cmap=cmap)
        ax = plt.gca()
        ttl = ax.get_title()
        ttl += ': view{}'.format(ndx)
        ax.set_title(ttl)
        out_mat_file = os.path.join(results_dir,'{}_{}_view{}_cv'.format(data_type,train_name, ndx,))
        save_mat(out_exp[0],out_mat_file)
        # save_mat(out_exp[0],os.path.join(cache_dir,'{}_view{}_cv'.format(data_type,ndx,)))

def get_cv_results_full(db_from_mdn_dir=False,
                        exp_name_pfix='',  # prefix for exp_name. can be map from train_type->exp_name_pfix
                        split_idxs=None,  # FOR DEBUGGING ONLY
                        ):
    '''

    :param db_from_mdn_dir:
    :param exp_name_pfix:
    :return: dict with ret[net][ivw]
    '''

    train_name = 'deepnet'

    print("Reading splits from {}".format(cv_info_file))
    data_info = h5py.File(cv_info_file, 'r')
    cv_info = apt.to_py(data_info['cvi'].value[:, 0].astype('int'))
    n_splits = max(cv_info) + 1
    num_splits = n_splits

    if split_idxs is None:
        split_idxs = range(num_splits)

    assert gt_lbl is None
    out_dict = {}
    for view in range(nviews):
        for tndx in range(len(all_models)):

            train_type = all_models[tndx]

            if view == 0:
                assert train_type not in out_dict
                out_dict[train_type] = {}

            for split in split_idxs:
                pfix_use = exp_name_pfix[train_type] if isinstance(exp_name_pfix, dict) else exp_name_pfix
                exp_name = '{}cv_split_{}'.format(pfix_use, split)
                pfix_use_mdn = exp_name_pfix['mdn'] if isinstance(exp_name_pfix, dict) else exp_name_pfix
                exp_name_mdn = '{}cv_split_{}'.format(pfix_use_mdn, split)

                # confs only used for .cachedir etc; actual conf used in tracking is loaded from
                # traindata. so optional kwargs etc unnec

                #mdn_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
                mdn_conf = create_conf_help('mdn', view, exp_name_mdn, quiet=True)
                #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                conf = create_conf_help(train_type, view, exp_name, quiet=True)
                #if op_af_graph is not None:
                #    conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))

                files = get_model_files(conf, train_name)
                files = files[-1:]
                assert len(files) == 1

                db_file = os.path.join(mdn_conf.cachedir, 'val_TF.tfrecords') if \
                    db_from_mdn_dir else os.path.join(conf.cachedir, 'val_TF.tfrecords')
                tfile = get_traindata_file_flexible(conf.cachedir, train_name)
                tdata = PoseTools.pickle_load(tfile)
                # latter case LEAP apparently
                tdataconf = tdata[1] if isinstance(tdata, list) else tdata

                print("###### {}, vw{}, split{}".format(train_type, view, split))
                print("dbf {}, tdf {}.".format(db_file, tfile))
                print("model {}.".format(files[0]))
                print("###### ")
                print(" ")
                print(" ")
                time.sleep(8)

                models_out = apt_expts.classify_db_all2(tdataconf, db_file, files, train_type,
                                                        name=train_name,
                                                        return_ims=True,
                                                        retrawpred=True
                                                        )
                assert len(models_out) == 1
                ret_dict, lbl_locs, info, mdlfile, mdlts = models_out[0]
                ret_dict['lbl_locs'] = lbl_locs
                ret_dict['info'] = np.array(info)
                ret_dict['mdlfile'] = np.array([mdlfile])
                ret_dict['mdlts'] = np.array([mdlts])

                if split == 0:
                    out_dict[train_type][view] = ret_dict
                else:
                    netvwd = out_dict[train_type][view]
                    for k in ret_dict:
                        assert k in netvwd
                        netvwd[k] = np.append(netvwd[k], ret_dict[k], axis=0)

    return out_dict


## single vs multiple animal ----RESULTS
def get_single_results():
    import multiResData
    assert data_type == 'alice'
    view = 0
    train_type = 'mdn'

    gpu_model = 'GeForceRTX2080Ti'
    sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
    common_conf = {}
    common_conf['rrange'] = 10
    common_conf['trange'] = 5
    common_conf['mdn_use_unet_loss'] = True
    common_conf['dl_steps'] = 60000
    common_conf['decay_steps'] = 20000
    common_conf['save_step'] = 5000
    common_conf['batch_size'] = 8
    common_conf['maxckpt'] = 20

    exp_names = ['single_vs_many_fly1', 'single_vs_many_fly2', 'single_vs_many_other']

    out_exp = {}
    train_name = 'deepnet'
    all_view = []
    for exp_name in exp_names:

        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
        if op_af_graph is not None:
            conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
        files = get_model_files(conf,train_name)
        files = files[-1:]
        out_file = os.path.join(conf.cachedir, train_name + '_results.p')
        recomp = do_recompute(out_file,files)

        db_file = os.path.join(conf.cachedir, 'val_TF.tfrecords')
        H = multiResData.read_and_decode_without_session(db_file,conf,())
        info = np.array(H[2])
        if recomp:
            mdn_out = apt_expts.classify_db_all(conf, db_file, files, train_type, name=train_name)
            with open(out_file, 'w') as f:
                pickle.dump([mdn_out, files], f)
        else:
            A = PoseTools.pickle_load(out_file)
            mdn_out = A[0]
        mdn_out[0].append(info)
        out_exp[exp_name] = mdn_out

    all_view.append(out_exp)

    for ndx, out_exp in enumerate(all_view):
        save_mat(out_exp, os.path.join(cache_dir, '{}_view{}_single_vs_multiple'.format(data_type, ndx, )))

#

    fly1 = [0,3]
    fly2 = [2,8]

    fly1_res = {}
    fly2_res = {}
    other_res = {}

    splits = []
    for exp_name in exp_names:
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
        splitfile = os.path.join(conf.cachedir,'splitdata.json')
        splits.append(PoseTools.json_load(splitfile))


    for exp_name in exp_names:
        mdn_out = out_exp[exp_name][0]
        dd = np.sqrt(np.sum((mdn_out[0]-mdn_out[1])**2,axis=-1))
        nex = mdn_out[0].shape[0]
        fly1_res[exp_name]= []
        fly2_res[exp_name]= []
        other_res[exp_name]= []
        for ndx in range(nex):
            cur_m = mdn_out[-1][ndx][0]
            cur_t = mdn_out[-1][ndx][2]
            cur_i = mdn_out[-1][ndx].tolist()
            if all([cur_i in s[1] for s in splits]):
                if [cur_m,cur_t] == fly1:
                    fly1_res[exp_name].append(dd[ndx,:])
                elif [cur_m,cur_t] ==fly2:
                    fly2_res[exp_name].append(dd[ndx,:])
                else:
                    other_res[exp_name].append(dd[ndx,:])


    percs = [50,75,90,95]
    per_res = []
    all_res = []
    for ndx, exp_name in enumerate(exp_names):
        gg = []
        gg.append(np.percentile(fly1_res[exp_name],percs,axis=0))
        gg.append(np.percentile(fly2_res[exp_name],percs,axis=0))
        gg.append(np.percentile(other_res[exp_name],percs,axis=0))
        per_res.append(np.array(gg))
        gg = []
        gg.append(fly1_res[exp_name])
        gg.append(fly2_res[exp_name])
        gg.append(other_res[exp_name])
        all_res.append(np.array(gg))

    per_res = np.array(per_res)


    #
    f,ax = plt.subplots(1,3,sharex=True,sharey=True)
    ax = ax.flatten()
    trange = np.arange(0,5,0.25)
    for count in range(3):
        if count == 0:
            cur_res = fly1_res
        elif count == 1:
            cur_res =fly2_res
        else:
            cur_res = other_res
        base_res = cur_res[exp_names[count]]
        ad = []
        for exp_name in exp_names:
            dd = np.array(cur_res[exp_name])-np.array(base_res)
            ad = []
            for tr in trange:
                cur_ad = np.count_nonzero(dd.flat>tr) - np.count_nonzero(dd.flat<-tr)
                ad.append(cur_ad/float(dd.flatten().size))
            ax[count].plot(trange,ad)


#
    import hdf5storage
    save_names = [u'train_fly1',u'train_fly2',u'train_others']
    out_dict = {}
    for ndx,exp_name in enumerate(exp_names):
        cur_dict = {}
        cur_dict[u'dist_fly1'] = np.array(fly1_res[exp_name])
        cur_dict[u'dist_fly2'] = np.array(fly2_res[exp_name])
        cur_dict[u'dist_flyothers'] = np.array(other_res[exp_name])
        out_dict[save_names[ndx]] = cur_dict

    hdf5storage.savemat(os.path.join(cache_dir,'alice_single_vs_multiple_results.mat'), out_dict, truncate_existing=True)


def run_active_learning(round_num,add_type='active',view=0):

    assert data_type in ['alice','stephen']
    import random
    import json

    n_add = 20
    exp_name = 'active_round0'
    c_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
    train_type = 'mdn'
    train_name = 'deepnet'
    common_conf['dl_steps'] = 20000
    common_conf['maxckpt'] = 3

    if round_num == 0:
        out_file = os.path.join(c_conf.cachedir,'initital_split.json')

        if add_type == 'active':
            info = get_label_info(c_conf)
            random.shuffle(info)
            train_split = info[:n_add]
            val_split = info[n_add:]
            splits = [train_split, val_split]
            assert not os.path.exists(out_file)
            with open(out_file, 'w') as f:
                json.dump(splits, f)
            exp_name = 'active_round{}'.format(round_num)
        else:
            exp_name = 'random_round{}'.format(round_num)

        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        conf.splitType = 'predefined'
        if 'deeplabcut' in train_type:
            apt.create_deepcut_db(conf, split=True, use_cache=True, split_file=out_file)
        elif 'leap' in train_type:
            apt.create_leap_db(conf, split=True, use_cache=True, split_file=out_file)
        else:
            apt.create_tfrecord(conf, split=True, use_cache=True, split_file=out_file)

        run_trainining(exp_name,train_type,view=view,run_type='submit')

    else:

        if add_type == 'active':
            prev_exp = 'active_round{}'.format(round_num-1)
            exp_name = 'active_round{}'.format(round_num)
        else:
            prev_exp = 'random_round{}'.format(round_num-1)
            exp_name = 'random_round{}'.format(round_num)

        # find the worse validation examples
        prev_conf = apt.create_conf(lbl_file, view, prev_exp, cache_dir, train_type)
        tfile = os.path.join(prev_conf.cachedir,'traindata')
        A = PoseTools.pickle_load(tfile)
        prev_conf = A[1]
        prev_splits = PoseTools.json_load(os.path.join(prev_conf.cachedir,'splitdata.json'))
        if op_af_graph is not None:
            prev_conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
        files = get_model_files(prev_conf,train_name)
        files = files[-1:]
        db_file = os.path.join(prev_conf.cachedir, 'val_TF.tfrecords')
        mdn_out = apt_expts.classify_db_all(prev_conf, db_file, files, train_type, name=train_name)
        res = mdn_out[0]
        val_info = res[2]
        dd = np.sqrt(np.sum((res[0]-res[1])**2,axis=-1))
        tot_dd = np.sum(dd,axis=-1)
        ord_dd = np.argsort(tot_dd)

        if add_type == 'active':
            # add the worst examples to training set
            sel_ex = ord_dd[-n_add:]
        else:
            sel_ex = random.sample(range(len(ord_dd)),n_add)

        train_add = [val_info[ss][0].tolist() for ss in sel_ex]
        gg = range(len(ord_dd))
        val_list = list(set(gg)-set(sel_ex))
        new_val = [val_info[ss][0].tolist() for ss in val_list]
        new_train = prev_splits[0] + train_add
        new_splits = [new_train,new_val]
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        out_split_file = os.path.join(conf.cachedir,'current_split.json')
        with open(out_split_file,'w') as f:
            json.dump(new_splits,f)

        conf.splitType = 'predefined'
        if 'deeplabcut' in train_type:
            apt.create_deepcut_db(conf, split=True, use_cache=True, split_file=out_split_file)
        elif 'leap' in train_type:
            apt.create_leap_db(conf, split=True, use_cache=True, split_file=out_split_file)
        else:
            apt.create_tfrecord(conf, split=True, use_cache=True, split_file=out_split_file)

        run_trainining(exp_name,train_type,view=view,run_type='submit')



def get_active_results(num_rounds=8,view=0):

    assert data_type in ['alice','stephen']
    import random
    import json

    train_type = 'mdn'
    train_name = 'deepnet'
    gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))

    active_exp = {}
    for add_type in ['active','random']:
        afiles = []
        for round_num in range(num_rounds):
            exp_name = '{}_round{}'.format(add_type,round_num)

            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            if op_af_graph is not None:
                conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
            files = get_model_files(conf,train_name)
            files = files[-1]
            afiles.append(files)


        exp_name = '{}_round{}'.format(add_type, 0)
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        out_file = os.path.join(conf.cachedir, train_name + '_results.p')
        recomp = do_recompute(out_file,afiles)

        if recomp:
            tfile = os.path.join(conf.cachedir, 'traindata')
            A = PoseTools.pickle_load(tfile)
            use_conf = A[1]

            mdn_out = apt_expts.classify_db_all(use_conf, gt_file, afiles, train_type, name=train_name)
            with open(out_file, 'w') as f:
                pickle.dump([mdn_out, afiles], f)
        else:
            A = PoseTools.pickle_load(out_file)
            mdn_out = A[0]

        for x, a in enumerate(mdn_out):
            a[-1] = x
        mdn_out.insert(0,mdn_out[0])
        active_exp[add_type] = mdn_out

    plot_results(active_exp)#,ylim=15)
    save_mat(active_exp,os.path.join(cache_dir,'{}_view{}_active'.format(data_type,view)))



def get_individual_animal_results_stephen(dstr):

    info_file = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190416.mat'

    data_info = h5py.File(info_file,'r')
    assert data_type == 'stephen'
    assert trn_flies is not None
    train_type = 'mdn_joint_fpn' #'mdn'

    conf = apt.create_conf(lbl_file,0,'dummy',cache_dir,train_type)
    lbl_movies, _ = multiResData.find_local_dirs(conf)
    in_movies = [PoseTools.read_h5_str(data_info[k]) for k in data_info['trnmovies'][0,:]]
    assert lbl_movies == in_movies

    train_name = 'deepnet_' + dstr
    all_view = []
    for view in range(nviews):
        gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))

        files = []
        for cur_fly in trn_flies:
            exp_name = 'train_fly_{}'.format(cur_fly)
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            cur_files = get_model_files(conf,train_name=train_name,net=train_type)
            files.append(cur_files[-1])

        exp_name = 'train_fly_random'
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        cur_files = get_model_files(conf,train_name=train_name,net=train_type)
        files.append(cur_files[-1])

        if op_af_graph is not None:
            conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))

        out_file = os.path.join(conf.cachedir,train_name + '_results.p')

        recomp = do_recompute(out_file,files)

        if recomp:
            afiles = [f.replace('.index', '') for f in files]
            mdn_out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,name=train_name)
            with open(out_file,'wb') as f:
                pickle.dump([mdn_out,files],f)
        else:
            A = PoseTools.pickle_load(out_file)
            mdn_out = A[0]

        all_view.append(mdn_out)

    for ndx,out_exp in enumerate(all_view):
        save_mat({train_type:out_exp},os.path.join(cache_dir,'{}_view{}_single'.format(data_type,ndx,)))


def do_recompute(out_file,files):
    recomp = False
    if os.path.exists(out_file):
        fts = [os.path.getmtime(glob.glob(f+'*')[0]) for f in files]
        ots = os.path.getmtime(out_file)
        if any([f > ots for f in fts]):
            recomp = True
        else:
            A = PoseTools.pickle_load(out_file)
            old_files = A[1]
            old_files = [f.replace('.index', '') for f in old_files]

            if len(files) != len(old_files) or not all([i == j for i, j in zip(files, old_files)]):
                recomp = True
    else:
        recomp = True
    return recomp


def get_model_files(conf,train_name='deepnet',n_max=10,net='mdn'):
    if net == 'leap_orig':
        train_name = 'deepnet'
    files1 = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
    files2 = glob.glob(os.path.join(conf.cachedir, "{}_202[0-9][0-9][0-9][0-9][0-9]-[0-9]*").format(train_name)) # Dont think  Ineed to worry beyond current decade.
    files = files1 + files2
    files.sort(key=os.path.getmtime)
    files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
    aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
    aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
    if any([a < 0 for a in aa]):
        bb = int(np.where(np.array(aa) < 0)[0][-1]) + 1
        files = files[bb:]
    if len(files) > n_max:
        gg = len(files)
        sel = np.linspace(0, len(files) - 1, n_max).astype('int')
        files = [files[s] for s in sel]
    files = [f.replace('.index', '') for f in files]
    return files



def get_label_info(conf):
    from scipy import io as sio
    local_dirs = multiResData.find_local_dirs(conf.labelfile)
    lbl = h5py.File(conf.labelfile, 'r')

    mov_info = []
    trx_info = []
    n_labeled_frames = 0
    for ndx, dir_name in enumerate(local_dirs):
        if conf.has_trx_file:
            trx_files = multiResData.get_trx_files(lbl, local_dirs)
            trx = sio.loadmat(trx_files[ndx])['trx'][0]
            n_trx = len(trx)
        else:
            n_trx = 1

        cur_mov_info = []
        for trx_ndx in range(n_trx):
            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx)
            mm = [ndx] * frames.size
            tt = [trx_ndx] * frames.size
            cur_trx_info = list(zip(mm, frames.tolist(), tt))
            trx_info.append(cur_trx_info)
            cur_mov_info.extend(cur_trx_info)
            n_labeled_frames += frames.size
        mov_info.append(cur_mov_info)
    lbl.close()

    info = []

    for ndx in range(len(local_dirs)):
        for mndx in range(len(mov_info[ndx])):
            info.extend(mov_info[ndx][mndx:mndx + 1])

    return info


def decode_db(gt_file,conf):
    H = multiResData.read_and_decode_without_session(gt_file, conf)
    ex_im = np.array(H[0][0])[:, :, 0]
    ex_loc = np.array(H[1][0])
    return ex_im, ex_loc

def l2err(pred, lbl, ptiles=(50,90,97)):
    e = np.sqrt( np.sum( (pred-lbl)**2, axis=2 ))
    eptls = np.percentile(e, ptiles, axis=0).T
    return e, eptls


def classify_db_all_res_to_dict(m):
    res = {'pred': m[0],
           'lbl': m[1],
           'info': m[2],
           'mdl': m[3],
           'extra': m[4],
           'ts': m[5]}
    res['err'] = np.sqrt( np.sum( (res['pred']-res['lbl'])**2, axis=2 ))
    return res

def track(movid=0,
               start_ndx=100,
               end_ndx=400,
               trx_ndx=None,
               out_dir='/nrs/branson/mayank/apt_cache/results_trk/',
               train_type='mdn',
               exp_name='apt_expt',
               model_file=None):

    lbl = h5py.File(lbl_file, 'r')

    for view in range(nviews):
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        tdata = PoseTools.pickle_load(os.path.join(conf.cachedir, 'traindata'))
        # Use the saved configuration for tracking.
        local_dirs, _ = multiResData.find_local_dirs(conf)
#        info = get_label_info(conf)

        if type(movid) is int:
            mov = local_dirs[movid]
            mov_str = '{}'.format(movid)
        else:
            mov = movid[view]
            mov_str = os.path.splitext(os.path.basename(mov))[0]

        if conf.has_trx_file:
            if type(movid) is int:
                trx_file = apt.read_string(lbl[lbl['trxFilesAll'][0,movid]])
            else:
                tt = apt.read_string(lbl[lbl['trxFilesAll'][0,0]])
                trx_str = os.path.basename(tt)
                trx_file = os.path.join(os.path.dirname(mov), trx_str)

            if trx_ndx is None:
                from scipy import io as sio
                gg = sio.loadmat(trx_file)
                n_trx = gg['trx'].shape[1]
                trx_ndx = np.array(range(n_trx))

        else:
            trx_file = None

        cur_out = '{}_mov_{}_from_{}_to_{}_view_{}_{}.trk'.format(data_type,mov_str,start_ndx,end_ndx,view,train_type)
        out_file = os.path.join(out_dir,cur_out)

        if type(lbl[lbl['movieFilesAllCropInfo'][0,0]]) != h5py._hl.dataset.Dataset:
            crop_loc = PoseTools.get_crop_loc(lbl, movid, view)
        else:
            crop_loc = None

        if type(tdata) is easydict.EasyDict:
            use_conf = tdata
        else:
            use_conf = tdata[1]

        apt.classify_movie_all(train_type,
                               conf=use_conf,
                               mov_file=mov,
                               trx_file=trx_file,
                               out_file=out_file,
                               start_frame=start_ndx,
                               end_frame=end_ndx,
                               trx_ids=trx_ndx,
                               crop_loc=crop_loc,
                               save_hmaps=False,
                               train_name='deepnet',
                               model_file=model_file
                               )
        tf.reset_default_graph()

def traindatadictdiff(exp1, exp2):
    tdfcn = lambda x: os.path.join(x, 'traindata')
    tdf1 = tdfcn(exp1)
    tdf2 = tdfcn(exp2)
    td1 = PoseTools.pickle_load(tdf1)
    td2 = PoseTools.pickle_load(tdf2)
    c1 = td1[-1]
    c2 = td2[-1]
    util.dictdiff(vars(c1), vars(c2))

    return c1, c2
