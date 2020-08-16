from __future__ import print_function

##  #######################        SETUP


# data_type = 'alice'
# data_type = 'stephen'
# data_type = 'roian'
# data_type = 'brit2'

import APT_interface as apt
import h5py
import PoseTools
import os
import shutil
import subprocess
import time
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import apt_expts
import os
import ast
import apt_expts
import os
import pickle
import multiResData
import multiResData
import random
import json
import tensorflow as tf
import easydict
import sys
import apt_dpk
import util


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

cache_dir = '/nrs/branson/mayank/apt_cache'
all_models = ['mdn', 'deeplabcut', 'unet', 'leap', 'openpose','resnet_unet']
# all_models = ['resnet_unet'] # 20191102. Rerunning experiments with deconvolved upsampling

gpu_model = 'GeForceRTX2080Ti'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
n_splits = 3

dlc_aug_use_round = 0

# common_conf procedure. Always call reload() first to initialize the global common_conf state.
# then setup, then individual actions furthwe tweak conf. Always call reload first!

common_conf = {}
common_conf['rrange'] = 10
common_conf['trange'] = 5
common_conf['brange'] = '\(-0.1,0.1\)'
common_conf['crange'] = '\(0.9,1.1\)'
common_conf['scale_factor_range'] = 1.2
common_conf['mdn_use_unet_loss'] = True
common_conf['dl_steps'] = 40000
common_conf['decay_steps'] = 20000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8
common_conf['normalize_img_mean'] = False
common_conf['adjust_contrast'] = False
common_conf['maxckpt'] = 50


def setup(data_type_in,gpu_device=None):
    global lbl_file, op_af_graph, gt_lbl, data_type, nviews, proj_name, trn_flies, cv_info_file, gt_name, \
        dpk_skel_csv
    data_type = data_type_in
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

    if data_type == 'alice' or data_type=='alice_difficult':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
        op_graph = []
        gt_lbl = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_allGT_stripped.lbl'
        #op_af_graph = '\(0,1\),\(0,2\),\(0,3\),\(0,4\),\(0,5\),\(5,6\),\(5,7\),\(5,9\),\(9,16\),\(9,10\),\(10,15\),\(9,14\),\(7,11\),\(7,8\),\(8,12\),\(7,13\)'
        op_af_graph = '\(0,1\),\(0,2\),\(0,3\),\(0,4\),\(0,5\),\(5,6\),\(5,7\),\(5,9\),\(9,16\),\(9,10\),\(10,15\),\(5,14\),\(7,11\),\(7,8\),\(8,12\),\(5,13\)'
        groups = ['']
        dpk_skel_csv = apt_dpk.skeleton_csvs[data_type]

        if data_type == 'alice_difficult':
            gt_lbl = '/nrs/branson/mayank/apt_cache/multitarget_bubble/multitarget_bubble_expandedbehavior_20180425_allGT_MDNvsDLC_labeled_alMassaged20190809_stripped.lbl'
            gt_name = '_diff'
    elif data_type == 'stephen':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped.lbl'
        gt_lbl = lbl_file
        #op_af_graph = '\(0,2\),\(1,3\),\(1,4\),\(2,4\)'
        # for vw2; who knows vw1
        op_af_graph = '\(0,2\),\(1,3\),\(2,4\),\(3,4\),\(2,3\)'
        dpk_skel_csv = apt_dpk.skeleton_csvs[data_type]

        trn_flies = [212, 216, 219, 229, 230, 234, 235, 241, 244, 245, 251, 254, 341, 359, 382, 417, 714, 719]
        trn_flies = trn_flies[::2]
        common_conf['trange'] = 20

    elif data_type == 'roian':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/roian_apt_dlstripped_newmovielocs.lbl'
        op_af_graph = '\(0,1\),\(0,2\),\(0,3\)'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/RoianTrainCVInfo20190420.mat'
        common_conf['rrange'] = 180
        common_conf['trange'] = 5
    elif data_type == 'brit0':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_0.lbl'
        op_af_graph = '\(0,4\),\(1,4\),\(2,4\),\(3,4\)'
        cv_info_file = '/groups/branson/bransonlab/experiments/data//BSTrainCVInfo20190416.mat'
        common_conf['trange'] = 20
    elif data_type == 'brit1':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_1.lbl'
        op_af_graph = '\(\(0,1\),\)'
        cv_info_file = '/groups/branson/bransonlab/experiments/data/BSTrainCVInfo20190416.mat'
        common_conf['trange'] = 20
    elif data_type == 'brit2':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_2.lbl'
        op_af_graph = '\(2,0\),\(2,1\)'
        cv_info_file = '/groups/branson/bransonlab/experiments/data/BSTrainCVInfo20190416.mat'
        common_conf['trange'] = 20
    elif data_type == 'romain':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/romain_dlstripped_trn1027.mat'
        op_af_graph = '(0,6),(6,12),(3,9),(9,15),(1,7),(7,13),(4,10),(10,16),(5,11),(11,17),(2,8),(8,14),(12,13),(13,14),(14,18),(18,17),(17,16),(16,15)'
        op_af_graph = op_af_graph.replace('(','\(')
        op_af_graph = op_af_graph.replace(')','\)')
        dpk_skel_csv = apt_dpk.skeleton_csvs[data_type]
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/RomainTrainCVInfo20200107.mat'
        common_conf['trange'] = 20
    elif data_type == 'larva':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/larva_dlstripped_20190420.lbl'
        cv_info_file = '/groups/branson/bransonlab/experiments/data/LarvaTrainCVInfo20190419.mat'
        j = tuple(zip(range(27), range(1, 28)))
        op_af_graph = '{}'.format(j)
        op_af_graph = op_af_graph.replace('(','\(')
        op_af_graph = op_af_graph.replace(')','\)')
        op_af_graph = op_af_graph.replace(' ','')
        common_conf['trange'] = 20
        common_conf['rrange'] = 180

    elif data_type == 'carsen':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/carsen_dlstripped_20190501T150134.lbl'
        common_conf['trange'] = 20
        op_af_graph = '\(\(0,1\),\)'
        cv_info_file = '/groups/branson/bransonlab/apt/experiments/data/CarsenTrainCVInfo20190514.mat'
    elif data_type == 'leap_fly':
        lbl_file = '/groups/branson/bransonlab/apt/experiments/data/leap_dataset_gt_stripped.lbl'
        gt_lbl = lbl_file
        gg = np.array(((1,2),(1,3),(1,4),(4,5),(5,6),(7,8),(8,9),(9,10),(11,12),(12,13),(13,14),(15,16),(16,17),(17,18),(19,20),(20,21),(21,22),(23,24),(24,25),(25,26),(27,28),(28,29),(29,30),(4,31),(4,32),(4,7),(4,19),(5,11),(5,15),(5,23),(5,27)))
        gg = gg-1
        op_af_graph = '{}'.format(gg.tolist())
        op_af_graph = op_af_graph.replace('[','\(')
        op_af_graph = op_af_graph.replace(']','\)')
        op_af_graph = op_af_graph.replace(' ','')
    else:
        lbl_file = ''

    lbl = h5py.File(lbl_file,'r')
    proj_name = apt.read_string(lbl['projname'])
    nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
    lbl.close()

def run_jobs(cmd_name,
             cur_cmd,
             redo=False,
             run_dir='/groups/branson/home/leea30/git/apt.dpk1920/deepnet',
             precmd='',
             logdir=sdir):
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
                             gpu_model=gpu_model,
                             run_dir=run_dir,
                             precmd=precmd)
    else:
        print('NOT submitting job {}'.format(cmd_name))


def get_tstr(tin):
    if np.isnan(tin):
        return ' --------- '
    else:
        return time.strftime('%m/%d %H:%M',time.localtime(tin))

def check_train_status(cmd_name, cache_dir, run_name='deepnet'):
    scriptfile = os.path.join(sdir,'opt_' + cmd_name+ '.sh')
    errfile = os.path.join(sdir,'opt_' + cmd_name+ '.err')
    if os.path.exists(scriptfile):
        submit_time = os.path.getmtime(scriptfile)
    else:
        submit_time = np.nan
    if os.path.exists(errfile):
        start_time = os.path.getmtime(errfile)
    else:
        start_time = np.nan
    train_dist = -1
    val_dist = -1

    files = glob.glob(os.path.join(cache_dir, "{}-[0-9]*").format(run_name))
    files.sort(key=os.path.getmtime)
    files = [f for f in files if os.path.splitext(f)[1] in ['.index','']]
    if len(files)>0:
        latest = files[-1]
        latest_model_iter = int(re.search('-(\d*)', latest).groups(0)[0])
        latest_time = os.path.getmtime(latest)
        if latest_time < submit_time:
            latest_time = np.nan
            latest_model_iter = -1
        else:
            if run_name == 'deepnet':
                tfile = 'traindata'
            else:
                tfile = run_name + '_traindata'
            A = PoseTools.pickle_load(os.path.join(cache_dir,tfile))
            if type(A) is list:
                train_dist = A[0]['train_dist'][-1]
                val_dist = A[0]['val_dist'][-1]
    else:
        latest_model_iter = -1
        latest_time = np.nan

    print('{:40s} submit: {}, latest iter: {:06d} at {}. train:{:.2f} val:{:.2f}'.format( cmd_name, get_tstr(submit_time),latest_model_iter, get_tstr(latest_time),train_dist,val_dist))
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
            mm.append(np.percentile(dd, ps, axis=0))
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
        cmap = PoseTools.get_cmap(len(ps),'cool')
    f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
    axx = axx.flat
    for idx,k in enumerate(data_in.keys()):
        o = data_in[k][-1]
        dd = np.sqrt(np.sum((o[0] - o[1]) ** 2, axis=-1))
        mm = np.percentile(dd, ps, axis=0)

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
        ax.set_title(k)
        ax.axis('off')

    f.tight_layout()
    return f


def save_mat(out_exp,out_file):
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
            if type(c[4]) in [list,tuple]:
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

def run_trainining_conf_helper(train_type, view0b, kwargs):
    '''
    Helper function that takes common_conf and further sets up for particular train_type

    :param train_type:
    :param kwargs:
    :return:
    '''

    conf_opts = common_conf.copy()
    # conf_opts.update(other_conf[conf_id])
    conf_opts['save_step'] = conf_opts['dl_steps'] // 10

    if data_type in ['brit0', 'brit1', 'brit2']:
        conf_opts['adjust_contrast'] = True
        if train_type == 'unet':
            conf_opts['batch_size'] = 2
        else:
            conf_opts['batch_size'] = 4

    if data_type in ['romain']:
        if train_type in ['mdn','resnet_unet']:
            conf_opts['batch_size'] = 2
        elif train_type in ['unet']:
            conf_opts['batch_size'] = 2
            conf_opts['rescale'] = 2
        else:
            conf_opts['batch_size'] = 4

    if data_type in ['larva']:
        conf_opts['batch_size'] = 4
        conf_opts['adjust_contrast'] = True
        conf_opts['clahe_grid_size'] = 20
        if train_type in ['unet','resnet_unet','leap']:
            conf_opts['rescale'] = 2
            conf_opts['batch_size'] = 2
        if train_type in ['mdn']:
            conf_opts['batch_size'] = 4
            conf_opts['rescale'] = 2
            conf_opts['mdn_use_unet_loss'] = True
            # conf_opts['mdn_learning_rate'] = 0.0001

    if data_type == 'stephen':
        conf_opts['batch_size'] = 4

    if data_type == 'carsen':
        if train_type in ['mdn','unet','resnet_unet']:
            conf_opts['rescale'] = 2.
        else:
            conf_opts['rescale'] = 1.
        conf_opts['adjust_contrast'] = True
        conf_opts['clahe_grid_size'] = 20
        if train_type in ['unet']:
            conf_opts['batch_size'] = 4
        else:
            conf_opts['batch_size'] = 8

    if op_af_graph is not None:
        conf_opts['op_affinity_graph'] = op_af_graph

    if dpk_skel_csv is not None:
        conf_opts['dpk_skel_csv'] = '\\"' + dpk_skel_csv[view0b] + '\\"'

    for k in kwargs.keys():
        conf_opts[k] = kwargs[k]

    return conf_opts

def cp_exp_bare(src_exp_dir, dst_exp_dir):
    '''
    Copy training dbs etc from existing expdir to new empty expdir
    :param src_exp_dir existing expdir
    :param dst_exp_dir: new expdir, created if nec
    :return:
    '''

    if not os.path.exists(dst_exp_dir):
        os.mkdir(dst_exp_dir)
        print("Created dir {}",format(dst_exp_dir))

    GLOBSPECS = ['*.tfrecords', 'splitdata.json']
    for globspec in GLOBSPECS:
        gs = os.path.join(src_exp_dir, globspec)
        globres = glob.glob(gs)
        for src in globres:
            fileshort = os.path.basename(src)
            dst = os.path.join(dst_exp_dir, fileshort)
            shutil.copyfile(src, dst)
            print("Copied {}->{}".format(src, dst))

def run_trainining(exp_name,train_type,view,run_type,
                   train_name='deepnet',
                   cp_from_existing_exp=None,  # short expname same dir as exp_name
                   exp_note='',
                   **kwargs
                   ):

    common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file, exp_name, cache_dir)
    end_cmd = 'train -skip_db -use_cache'
    cmd_opts = {}
    cmd_opts['type'] = train_type
    cmd_opts['view'] = view + 1
    cmd_opts['train_name'] = train_name

    conf_opts = run_trainining_conf_helper(train_type, view, kwargs)  # this is copied from common_conf
    conf_str = apt.conf_opts_dict2pvargstr(conf_opts)

    opt_str = ''
    for k in cmd_opts.keys():
        opt_str = '{} -{} {} '.format(opt_str, k, cmd_opts[k])

    now_str = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    cur_cmd = common_cmd + conf_str + opt_str + end_cmd
    cmd_name = '{}_view{}_{}_{}_{}_{}'.format(data_type, view, exp_name, train_type, train_name, now_str)
    precmd = 'export PYTHONPATH="{}"'.format(dpk_py_path) if train_type == 'dpk' else ''

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
        run_jobs(cmd_name, cur_cmd, precmd=precmd, logdir=explog_dir)
    elif run_type == 'status':
        #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        conf = create_conf_help(train_type, view, exp_name, quiet=True, **kwargs)
        check_train_status(cmd_name, conf.cachedir)


def create_conf_help(train_type, view, exp_name, quiet=False, **kwargs):
    '''
    Call apt.create_conf after customizing the conf for the given train_type/view/kwargs.
    :param train_type:
    :param view:
    :param exp_name:
    :param kwargs:
    :return:
    '''
    conf_opts = run_trainining_conf_helper(train_type, view, kwargs)
    pvlist = apt.conf_opts_dict2pvargstr(conf_opts)
    pvlist = apt.conf_opts_pvargstr2list(pvlist)
    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type,
                           conf_params=pvlist, quiet=quiet)
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



def create_normal_dbs():
    exp_name = 'apt_expt'
    # assert gt_lbl is not None
    for view in range(nviews):
        for tndx in range(len(all_models)):
            train_type = all_models[tndx]
            conf = create_conf_help(train_type, view, exp_name)
            if train_type == 'deeplabcut':
                apt.create_deepcut_db(conf,split=False,use_cache=True)
            elif train_type == 'leap':
                apt.create_leap_db(conf,split=False,use_cache=True)
            else:
                apt.create_tfrecord(conf,split=False,use_cache=True)


def cv_train_from_mat(skip_db=True, run_type='status', create_splits=False,
                      exp_name_pfix='',  # prefix for exp_name
                      split_idxs=None,  # optional list of split indices to run
                      **kwargs):
    assert data_type in ['romain','larva','roian','carsen']

    data_info = h5py.File(cv_info_file, 'r')
    cv_info = apt.to_py(data_info['cvi'].value[:, 0].astype('int'))
    n_splits = max(cv_info) + 1
    conf = apt.create_conf(lbl_file,0,'cv_dummy',cache_dir,'mdn')
    lbl_movies, _ = multiResData.find_local_dirs(conf)
    in_movies = [PoseTools.read_h5_str(data_info[k]) for k in data_info['movies'][0,:]]
    assert lbl_movies == in_movies or data_type in ['romain','roian']
    label_info = get_label_info(conf)
    fr_info = apt.to_py(data_info['frame'].value[:,0].astype('int'))
    m_info = apt.to_py(data_info['movieidx'].value[:,0].astype('int'))
    if 'target' in data_info.keys():
        t_info = apt.to_py(data_info['target'].value[:,0].astype('int'))
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
            for view in range(nviews):
                for train_type in all_models:
                    conf = create_conf_help(train_type, view, exp_name, **kwargs)
                    #conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                    conf.splitType = 'predefined'
                    if train_type == 'deeplabcut':
                        apt.create_deepcut_db(conf, split=True, split_file=split_file, use_cache=True)
                    elif train_type == 'leap':
                        apt.create_leap_db(conf, split=True, split_file=split_file, use_cache=True)
                    else:
                        apt.create_tfrecord(conf, split=True, split_file=split_file, use_cache=True)


    for view in range(nviews):
        for train_type in all_models:
            for sndx in split_idxs:
                exp_name = '{}cv_split_{}'.format(exp_name_pfix, sndx)
                run_trainining(exp_name,train_type,view,run_type, **kwargs)


def cv_train_britton(skip_db=True, run_type='status',create_splits=False):
    assert data_type[:4] == 'brit'
    britnum = int(data_type[4])
    data_info = h5py.File(cv_info_file, 'r')
    cv_info = apt.to_py(data_info[data_info['cvidx'][britnum,0]].value[:,0].astype('int'))
    n_splits = max(cv_info) + 1

    conf = apt.create_conf(lbl_file,0,'cv_dummy',cache_dir,'mdn')
    lbl_movies, _ = multiResData.find_local_dirs(conf)
    mov_ptr = data_info[data_info['trnmovies'][britnum,0]][0,:]
    in_movies = [PoseTools.read_h5_str(data_info[k]) for k in mov_ptr]
    assert lbl_movies == in_movies
    label_info = get_label_info(conf)
    for sndx in range(max(cv_info)+1):
        val_info = [l for ndx,l in enumerate(label_info) if cv_info[l[0]]==sndx]
        trn_info = list(set(label_info)-set(val_info))
        cur_split = [trn_info,val_info]
        exp_name = 'cv_split_{}'.format(sndx)
        split_file = os.path.join(cache_dir,proj_name,exp_name) + '.json'
        if not skip_db and create_splits:
            assert not os.path.exists(split_file)
            with open(split_file,'w') as f:
                json.dump(cur_split,f)

        # create the dbs
        if not skip_db:
            for view in range(nviews):
                for train_type in all_models:
                    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                    conf.splitType = 'predefined'
                    if train_type == 'deeplabcut':
                        apt.create_deepcut_db(conf, split=True, split_file=split_file, use_cache=True)
                    elif train_type == 'leap':
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
                if train_type == 'deeplabcut':
                    apt.create_deepcut_db(conf,split=True,use_cache=True,split_file=cur_split_file)
                elif train_type == 'leap':
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
    m_ndx = apt.to_py(lbl['preProcData_MD_mov'].value[0, :].astype('int'))
    t_ndx = apt.to_py(lbl['preProcData_MD_iTgt'].value[0, :].astype('int'))
    f_ndx = apt.to_py(lbl['preProcData_MD_frm'].value[0, :].astype('int'))

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
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                mdn_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
                split_file= os.path.join(mdn_conf.cachedir,'splitinfo.json')
                if do_split:
                    assert not os.path.exists(split_file)
                    with open(split_file,'w') as f:
                        json.dump(splits,f)

                conf.splitType = 'predefined'
                if train_type == 'deepla`bcut':
                    apt.create_deepcut_db(conf, split=True, split_file=split_file,use_cache=True)
                elif train_type == 'leap':
                    apt.create_leap_db(conf, split=True, split_file=split_file, use_cache=True)
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


def create_run_individual_animal_dbs_stephen(skip_db = True, run_type='status'):

    info_file = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190416.mat'

    data_info = h5py.File(info_file,'r')
    assert data_type == 'stephen'
    assert trn_flies is not None
    train_type = 'mdn'
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
            run_trainining(exp_name,train_type,view,run_type)

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
        run_trainining(exp_name,train_type,view,run_type)



def run_normal_training(expname = 'apt_expt',
                        run_type = 'status',
                        **kwargs
                        ):

    common_conf['dl_steps'] = 50000
    common_conf['maxckpt'] = 20
    common_conf['save_time'] = 20 # save every 20 min

    results = {}
    for train_type in all_models:
        for view in range(nviews):
            key = "{}_vw{}".format(train_type, view)
            results[key] = run_trainining(expname, train_type, view, run_type, **kwargs)

    return results

## CV Training ---- TRAINING ----

def run_cv_training(run_type='status'):

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
                conf_opts['save_step'] = conf_opts['dl_steps'] / 10
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
            conf_opts['save_step'] = conf_opts['dl_steps']/20
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
    # dlc_cmd = '/groups/branson/bransonlab/mayank/apt_expts/deepcut/pose-tensorflow/train.py'
    dlc_cmd = os.path.join(apt_expts.deepcut_dir,'run_train.py')

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


def run_incremental_training(run_type='status'):
    # Expt where we find out how training error changes with amount of training data

    n_rounds = 8
    info = []
    for view in range(nviews):
        r_info = []
        for ndx in range(n_rounds):
            exp_name = '{}_randsplit_round_{}'.format(data_type,ndx)
            cur_info = {}
            for train_type in all_models:

                common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file,exp_name, cache_dir)
                end_cmd = 'train -skip_db -use_cache'
                cmd_opts = {}
                cmd_opts['type'] = train_type
                cmd_opts['view'] = view + 1
                conf_opts = common_conf.copy()
                # conf_opts.update(other_conf[conf_id])
                conf_opts['save_step'] = conf_opts['dl_steps']/ 10
                if data_type == 'stephen':
                    conf_opts['batch_size'] = 4
                if op_af_graph is not None:
                    conf_opts['op_affinity_graph'] = op_af_graph

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
                cmd_name = '{}_view{}_{}'.format(exp_name,view,train_type)

                if run_type == 'submit':
                    print(cur_cmd)
                    print()
                    run_jobs(cmd_name,cur_cmd)
                elif run_type == 'status':
                    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                    iter = check_train_status(cmd_name,conf.cachedir)
                    cur_info[train_type] = iter
            r_info.append(cur_info)
        info.append(r_info)


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
        conf_opts['save_step'] = conf_opts['dl_steps'] / 10
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


def get_normal_results(exp_name='apt_expt', train_name='deepnet', **kwargs):
## Normal Training  ------- RESULTS -------
    # cache_dir = '/nrs/branson/al/cache'

    all_view = []

    for view in range(nviews):
        out_exp = {}

        gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}{}.tfrecords'.format(view,gt_name))
        for train_type in all_models:

            conf = create_conf_help(train_type, view, exp_name, **kwargs)

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

            files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
            files.sort(key=os.path.getmtime)
            files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
            aa = [int(re.search('-(\d*)',f).groups(0)[0]) for f in files]
            aa = [b-a for a,b in zip(aa[:-1],aa[1:])]
            if any([a<0 for a in aa]):
                bb = int(np.where(np.array(aa)<0)[0])+1
                files = files[bb:]
            # n_max = 10
            # if len(files)> n_max:
            #     gg = len(files)
            #     sel = np.linspace(0,len(files)-1,n_max).astype('int')
            #     files = [files[s] for s in sel]

            print('view {}, net {}. Your models are:'.format(view, train_type))
            print(files)

            out_file = os.path.join(conf.cachedir,train_name + '_results{}.p'.format(gt_name))
            recomp = False
            if os.path.exists(out_file):
                fts = [os.path.getmtime(f) for f in files]
                ots = os.path.getmtime(out_file)
                if any([f > ots for f in fts]):
                    recomp = True
                else:
                    A = PoseTools.pickle_load(out_file)
                    old_files = A[1]
                    if not all([i==j for i,j in zip(files,old_files)]):
                        recomp = True
            else:
                recomp = True

            # recomp = False

            if recomp:
                afiles = [f.replace('.index', '') for f in files]
                mdn_out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,name=train_name)
                with open(out_file,'wb') as f:
                    pickle.dump([mdn_out,files],f)
            else:
                A = PoseTools.pickle_load(out_file)
                mdn_out = A[0]

            out_exp[train_type] = mdn_out

        H = multiResData.read_and_decode_without_session(gt_file, conf)
        ex_im = np.array(H[0][0])[:, :, 0]
        ex_loc = np.array(H[1][0])
        all_view.append([out_exp,ex_im,ex_loc])

    for ndx,out_exp in enumerate(all_view):
        plot_results(out_exp[0])
        plot_hist(out_exp)
        save_mat(out_exp[0],os.path.join(cache_dir,'{}_view{}_time{}'.format(data_type,ndx,gt_name)))

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


            out_file = os.path.join(conf.cachedir,'{}_no_unet_results.p'.format(train_name))
            recomp = False
            if os.path.exists(out_file):
                fts = [os.path.getmtime(f) for f in files]
                ots = os.path.getmtime(out_file)
                if any([f > ots for f in fts]):
                    recomp = True
                else:
                    A = PoseTools.pickle_load(out_file)
                    old_files = A[1]
                    if not all([i==j for i,j in zip(files,old_files)]):
                        recomp = True
            else:
                recomp = True

            # recomp = False

            if recomp:
                afiles = [f.replace('.index', '') for f in files]
                if train_name == 'deepnet':
                    td_file = os.path.join(conf.cachedir,'traindata')
                else:
                    td_file = os.path.join(conf.cachedir,conf.expname + '_' + train_name + '_traindata')
                td = PoseTools.pickle_load(td_file)
                cur_conf = td[1]
                mdn_out = apt_expts.classify_db_all(cur_conf,gt_file,afiles,train_type,name=train_name)
                with open(out_file,'w') as f:
                    pickle.dump([mdn_out,files],f)
            else:
                A = PoseTools.pickle_load(out_file)
                mdn_out = A[0]

            out_exp[train_name] = mdn_out

        H = multiResData.read_and_decode_without_session(gt_file, conf)
        ex_im = np.array(H[0][0])[:, :, 0]
        ex_loc = np.array(H[1][0])
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
            files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
            files.sort(key=os.path.getmtime)
            files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
            aa = [int(re.search('-(\d*)',f).groups(0)[0]) for f in files]
            aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
            if any([a < 0 for a in aa]):
                bb = int(np.where(np.array(aa) < 0)[0]) + 1
                files = files[bb:]

            files = files[-1:]
            out_file = os.path.join(conf.cachedir,train_name + '_results.p')
            recomp = False
            if os.path.exists(out_file):
                fts = [os.path.getmtime(f) for f in files]
                ots = os.path.getmtime(out_file)
                if any([f > ots for f in fts]):
                    recomp = True
                else:
                    A = PoseTools.pickle_load(out_file)
                    old_files = A[1]
                    if not all([i==j for i,j in zip(files,old_files)]):
                        recomp = True
            else:
                recomp = True

            if redo:
                recomp = True

            if recomp:
                afiles = [f.replace('.index', '') for f in files]
#                tdata = PoseTools.pickle_load()
                mdn_out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,name=train_name)
                with open(out_file,'w') as f:
                    pickle.dump([mdn_out,files],f)
            else:
                A = PoseTools.pickle_load(out_file)
                mdn_out = A[0]

            dlc_exp[train_name] = mdn_out
        H = multiResData.read_and_decode_without_session(gt_file, conf)
        ex_im = np.array(H[0][0])[:, :, 0]
        ex_loc = np.array(H[1][0])
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
            files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
            files.sort(key=os.path.getmtime)
            files = [f for f in files if os.path.splitext(f)[1] in ['.index', '','.h5']]
            aa = [int(re.search('-(\d*)',f).groups(0)[0]) for f in files]
            aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
            if any([a < 0 for a in aa]):
                bb = int(np.where(np.array(aa) < 0)[0][-1]) + 1
                files = files[bb:]
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
                files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
                files.sort(key=os.path.getmtime)
                files = [f for f in files if os.path.splitext(f)[1] in ['.index', '','.h5']]
                aa = [int(re.search('-(\d*)',f).groups(0)[0]) for f in files]
                aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
                if any([a < 0 for a in aa]):
                    bb = int(np.where(np.array(aa) < 0)[0][-1]) + 1
                    files = files[bb:]

                tt = [os.path.getmtime(f) for f in files]
                td = np.array([t-tt[0] for t in tt])
                t_closest = np.argmin(np.abs(td-tmax))
                files = [files[0],files[t_closest]]

                out_file = os.path.join(conf.cachedir,train_name + '_results.p')
                recomp = False
                if os.path.exists(out_file):
                    fts = [os.path.getmtime(f) for f in files]
                    ots = os.path.getmtime(out_file)
                    if any([f > ots for f in fts]):
                        recomp = True
                    else:
                        A = PoseTools.pickle_load(out_file)
                        old_files = A[1]
                        if len(files) != len(old_files) or not all([i==j for i,j in zip(files,old_files)]):
                            recomp = True
                else:
                    recomp = True

                if recomp:
                    afiles = [f.replace('.index', '') for f in files]
                    mdn_out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,name=train_name)
                    with open(out_file,'w') as f:
                        pickle.dump([mdn_out,files],f)
                else:
                    A = PoseTools.pickle_load(out_file)
                    mdn_out = A[0]

                dlc_exp[train_name] = mdn_out
            H = multiResData.read_and_decode_without_session(gt_file, conf)
            ex_im = np.array(H[0][0])[:, :, 0]
            ex_loc = np.array(H[1][0])
            all_view.append([dlc_exp,ex_im,ex_loc])
            # all_view.append(dlc_exp)

        for ndx, dlc_exp in enumerate(all_view):
            # plot_results(dlc_exp[0])
            plot_hist(dlc_exp,[50,70,90,95])
            out_file = os.path.join(cache_dir,'{}_view{}_round{}_leap'.format(data_type,ndx,use_round))
            save_mat(dlc_exp[0],out_file)


## incremental training -- RESULTS ---
def get_incremental_results():
    n_rounds = 8
    all_res = []
    train_name = 'deepnet'
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
            for ndx in range(n_rounds):
                exp_name = '{}_randsplit_round_{}'.format(data_type, ndx)
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                split_data = PoseTools.json_load(os.path.join(conf.cachedir,'splitdata.json'))
                train_size.append(len(split_data[0]))
                if op_af_graph is not None:
                    conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
                files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
                files.sort(key=os.path.getmtime)
                files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
                aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
                if len(files)>0 and aa[-1]>0:
                    r_files.append(files[-1])
                else:
                    print('MISSING!!!! MISSING!!!! {} {}'.format(train_type,ndx))

            out_file = os.path.join(conf.cachedir,train_name + '_results.p')
            recomp = False
            if os.path.exists(out_file):
                fts = [os.path.getmtime(f) for f in r_files]
                ots = os.path.getmtime(out_file)
                if any([f > ots for f in fts]):
                    recomp = True
                else:
                    A = PoseTools.pickle_load(out_file)
                    old_files = A[1]
                    if (len(r_files) != len(old_files)) or (not all([i==j for i,j in zip(r_files,old_files)])):
                        recomp = True
            else:
                recomp = True

            if recomp:
                afiles = [f.replace('.index', '') for f in r_files]
                mdn_out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,name=train_name)
                with open(out_file,'w') as f:
                    pickle.dump([mdn_out,r_files],f)
            else:
                A = PoseTools.pickle_load(out_file)
                mdn_out = A[0]

            # mdn_out = apt_expts.classify_db_all(conf, gt_file, r_files, train_type, name=train_name)

            for x, a in enumerate(mdn_out):
                a[-1] = train_size[x]
                a[2] = np.array(a[2])
            mdn_out.insert(0,mdn_out[0])
            inc_exp[train_type] = mdn_out
        all_view.append(inc_exp)

    for ndx,ii in enumerate(all_view):
        plot_results(ii,ylim=15)
        save_mat(ii,os.path.join(cache_dir,'{}_view{}_trainsize'.format(data_type,ndx,)))

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
                files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
                files.sort(key=os.path.getmtime)
                files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
                aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
                aa = [b - a for a, b in zip(aa[:-1], aa[1:])]

                if any([a < 0 for a in aa]):
                    bb = int(np.where(np.array(aa) < 0)[-1]) + 1
                    files = files[bb:]
                n_max = 10
                if len(files) > n_max:
                    gg = len(files)
                    sel = np.linspace(0, len(files) - 1, n_max).astype('int')
                    files = [files[s] for s in sel]
                r_files = files
                out_file = os.path.join(conf.cachedir,train_name + '_pretraining_results.p')
                recomp = False

                if os.path.exists(out_file):
                    fts = [os.path.getmtime(f) for f in r_files]
                    ots = os.path.getmtime(out_file)
                    if any([f > ots for f in fts]):
                        recomp = True
                    else:
                        A = PoseTools.pickle_load(out_file)
                        old_files = A[1]
                        if (len(r_files) != len(old_files)) or (not all([i==j for i,j in zip(r_files,old_files)])):
                            recomp = True
                else:
                    recomp = True

                if recomp:
                    afiles = [f.replace('.index', '') for f in r_files]
                    mdn_out = apt_expts.classify_db_all(conf,gt_file,afiles,train_type,name=train_name)
                    with open(out_file,'w') as f:
                        pickle.dump([mdn_out,r_files],f)
                else:
                    A = PoseTools.pickle_load(out_file)
                    mdn_out = A[0]


                # for x, a 09-Jandhriti
                # in enumerate(mdn_out):
                #     a[-1] = train_size[x]
                #     a[2] = np.array(a[2])
                # mdn_out.insert(0,mdn_out[0])
                predtrained_exp[train_name] = mdn_out

            H = multiResData.read_and_decode_without_session(gt_file, conf)
            ex_im = np.array(H[0][0])[:, :, 0]
            ex_loc = np.array(H[1][0])
            all_view.append([predtrained_exp, ex_im, ex_loc])

            for ndx_i,ii in enumerate(all_view):
                plot_results(ii[0],ylim=15)
                plot_hist(ii)
                save_mat(ii[0],os.path.join(cache_dir,'{}_view{}_round{}_pretrained'.format(data_type,ndx_i,ndx)))



## CV Results

def get_cv_results(num_splits=None):
    train_name = 'deepnet'
    if num_splits == None:
        num_splits = n_splits

    assert gt_lbl is None
    all_view = []
    for view in range(nviews):
        out_exp = {}
        for tndx in range(len(all_models)):
            train_type = all_models[tndx]

            ex_im = None
            ex_loc = None
            out_split = None
            for split in range(num_splits):
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
                    bb = int(np.where(np.array(aa)<0)[0][-1])+1
                    files = files[bb:]
                files = files[-1:]
                # n_max = 10
                # if len(files)> n_max:
                #     gg = len(files)
                #     sel = np.linspace(0,len(files)-1,n_max).astype('int')
                #     files = [files[s] for s in sel]

                out_file = os.path.join(conf.cachedir,train_name + '_results.p')
                recomp = False
                if os.path.exists(out_file):
                    fts = [os.path.getmtime(f) for f in files]
                    ots = os.path.getmtime(out_file)
                    if any([f > ots for f in fts]):
                        recomp = True
                    else:
                        A = PoseTools.pickle_load(out_file)
                        old_files = A[1]
                        if not all([i==j for i,j in zip(files,old_files)]):
                            recomp = True
                else:
                    recomp = True

                if recomp:
                    afiles = [f.replace('.index', '') for f in files]
                    db_file = os.path.join(mdn_conf.cachedir,'val_TF.tfrecords')
                    tdata = PoseTools.pickle_load(os.path.join(conf.cachedir,'traindata'))
                    mdn_out = apt_expts.classify_db_all(tdata[1],db_file,afiles,train_type,name=train_name)
                    with open(out_file,'w') as f:
                        pickle.dump([mdn_out,files],f)
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
                    db_file = os.path.join(mdn_conf.cachedir, 'val_TF.tfrecords')
                    H = multiResData.read_and_decode_without_session(db_file,mdn_conf)
                    ex_ims = np.array(H[0][0])
                    ex_locs = np.array(H[1][0])
            out_exp[train_type] = out_split
        all_view.append([out_exp,ex_ims,ex_locs])

    for ndx,out_exp in enumerate(all_view):
        cmap = np.array([[0.5200,         0,         0],
                [1.0000,    0.5200,         0],
                [0.4800,    1.0000,    0.5200],
                [0,    0.5200,    1.0000],
                [0,         0,    0.5200]])
        plot_hist(out_exp,ps=[50,75,90,95,97],cmap=cmap)
        save_mat(out_exp[0],os.path.join(cache_dir,'{}_view{}_cv'.format(data_type,ndx,)))


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
        files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
        files.sort(key=os.path.getmtime)
        files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
        aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
        aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
        if any([a < 0 for a in aa]):
            bb = int(np.where(np.array(aa) < 0)[0]) + 1
            files = files[bb:]
        files = files[-1:]

        out_file = os.path.join(conf.cachedir, train_name + '_results.p')
        recomp = False
        if os.path.exists(out_file):
            fts = [os.path.getmtime(f) for f in files]
            ots = os.path.getmtime(out_file)
            if any([f > ots for f in fts]):
                recomp = True
            else:
                A = PoseTools.pickle_load(out_file)
                old_files = A[1]
                if not all([i == j for i, j in zip(files, old_files)]):
                    recomp = True
        else:
            recomp = True

        db_file = os.path.join(conf.cachedir, 'val_TF.tfrecords')
        H = multiResData.read_and_decode_without_session(db_file,conf,())
        info = np.array(H[2])
        if recomp:
            afiles = [f.replace('.index', '') for f in files]
            mdn_out = apt_expts.classify_db_all(conf, db_file, afiles, train_type, name=train_name)
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
        if train_type == 'deeplabcut':
            apt.create_deepcut_db(conf, split=True, use_cache=True, split_file=out_file)
        elif train_type == 'leap':
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
        files = glob.glob(os.path.join(prev_conf.cachedir, "{}-[0-9]*").format(train_name))
        files.sort(key=os.path.getmtime)
        files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
        aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
        aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
        if any([a < 0 for a in aa]):
            bb = int(np.where(np.array(aa) < 0)[0]) + 1
            files = files[bb:]
        files = files[-1:]

        afiles = [f.replace('.index', '') for f in files]
        db_file = os.path.join(prev_conf.cachedir, 'val_TF.tfrecords')
        mdn_out = apt_expts.classify_db_all(prev_conf, db_file, afiles, train_type, name=train_name)
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
        if train_type == 'deeplabcut':
            apt.create_deepcut_db(conf, split=True, use_cache=True, split_file=out_split_file)
        elif train_type == 'leap':
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
            files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
            files.sort(key=os.path.getmtime)
            files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
            aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
            aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
            if any([a < 0 for a in aa]):
                bb = int(np.where(np.array(aa) < 0)[0]) + 1
                files = files[bb:]
            files = files[-1]
            afiles.append(files)


        exp_name = '{}_round{}'.format(add_type, 0)
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        out_file = os.path.join(conf.cachedir, train_name + '_results.p')
        recomp = False
        if os.path.exists(out_file):
            fts = [os.path.getmtime(f) for f in afiles]
            ots = os.path.getmtime(out_file)
            if any([f > ots for f in fts]):
                recomp = True
            else:
                A = PoseTools.pickle_load(out_file)
                old_files = A[1]
                if (len(afiles) != len(old_files)) or (not all([i == j for i, j in zip(afiles, old_files)])):
                    recomp = True
        else:
            recomp = True

        if recomp:
            r_files = [f.replace('.index', '') for f in afiles]
            tfile = os.path.join(conf.cachedir, 'traindata')
            A = PoseTools.pickle_load(tfile)
            use_conf = A[1]

            mdn_out = apt_expts.classify_db_all(use_conf, gt_file, r_files, train_type, name=train_name)
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



def get_individual_animal_results_stephen():

    info_file = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190416.mat'

    data_info = h5py.File(info_file,'r')
    assert data_type == 'stephen'
    assert trn_flies is not None
    train_type = 'mdn'

    conf = apt.create_conf(lbl_file,0,'dummy',cache_dir,train_type)
    lbl_movies, _ = multiResData.find_local_dirs(conf)
    in_movies = [PoseTools.read_h5_str(data_info[k]) for k in data_info['trnmovies'][0,:]]
    assert lbl_movies == in_movies

    train_name = 'deepnet'
    all_view = []
    for view in range(nviews):
        gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))

        files = []
        for cur_fly in trn_flies:
            exp_name = 'train_fly_{}'.format(cur_fly)
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            cur_files = get_model_files(conf)
            files.append(cur_files[-1])

        exp_name = 'train_fly_random'
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        cur_files = get_model_files(conf)
        files.append(cur_files[-1])

        if op_af_graph is not None:
            conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))

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

        all_view.append(mdn_out)

    for ndx,out_exp in enumerate(all_view):
        save_mat({'mdn':out_exp},os.path.join(cache_dir,'{}_view{}_single'.format(data_type,ndx,)))


def do_recompute(out_file,files):
    recomp = False
    if os.path.exists(out_file):
        fts = [os.path.getmtime(f) for f in files]
        ots = os.path.getmtime(out_file)
        if any([f > ots for f in fts]):
            recomp = True
        else:
            A = PoseTools.pickle_load(out_file)
            old_files = A[1]
            if not all([i == j for i, j in zip(files, old_files)]):
                recomp = True
    else:
        recomp = True
    return recomp


def get_model_files(conf,train_name='deepnet'):
    files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
    files.sort(key=os.path.getmtime)
    files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
    aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
    aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
    if any([a < 0 for a in aa]):
        bb = int(np.where(np.array(aa) < 0)[0]) + 1
        files = files[bb:]
    return files


def get_label_info(conf):
    from scipy import io as sio
    local_dirs, _ = multiResData.find_local_dirs(conf)
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


def create_mat_file():
    from scipy import io as sio
    exp_name = 'apt_expt'
    train_type = 'mdn'
    for view in range(nviews):
        gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))
        conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
        train_db_file = os.path.join(conf.cachedir,'train_TF.tfrecords')
        H = multiResData.read_and_decode_without_session(train_db_file,conf,())
        ims = np.array(H[0])
        locs = apt.to_mat(np.array(H[1]))
        info = apt.to_mat(np.array(H[2]))
        G = multiResData.read_and_decode_without_session(gt_file,conf,())
        ims_gt = np.array(G[0])
        locs_gt = apt.to_mat(np.array(G[1]))
        info_gt = apt.to_mat(np.array(G[2]))
        out_file = os.path.join(cache_dir,proj_name,'data_view{}.mat'.format(view))
        help_text = '''
        ims: Training images
        locs: Landmark location for the training images
        info: Information about the examples.
        ims_gt: Ground truth images
        locs_gt: Landmark location for the GT images.
        info_gt: Information about the examples.
                    
        '''
        sio.savemat(out_file,{'ims':ims,'locs':locs,'info':info,'ims_gt':ims_gt,'locs_gt':locs_gt,'info_gt':info_gt,'help':help_text})


