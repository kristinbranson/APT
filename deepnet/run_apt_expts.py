
##  #######################        SETUP


data_type = 'alice'
# data_type = 'stephen'

import APT_interface as apt
import h5py
import PoseTools
import os
import time
import glob
import re
import numpy as np

if data_type == 'alice':
    lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
    op_graph = []
elif data_type == 'stephen':
    lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_dlstripped.lbl'
else:
    lbl_file = ''


lbl = h5py.File(lbl_file,'r')
nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
lbl.close()
cache_dir = '/nrs/branson/mayank/apt_cache'
all_models = ['mdn','deeplabcut','unet','openpose','leap']

gpu_model = 'GeForceRTX2080Ti'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

def run_jobs(cmd_name,cur_cmd,redo=False):
    logfile = os.path.join(sdir,'opt_' + cmd_name+ '.log')
    errfile = os.path.join(sdir,'opt_' + cmd_name+ '.err')

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
        PoseTools.submit_job(cmd_name, cur_cmd, sdir, gpu_model=gpu_model)
    else:
        print('NOT submitting job {}'.format(cmd_name))


def get_tstr(tin):
    if np.isnan(tin):
        return '0000000'
    else:
        return time.strftime('%m/%d %H:%M',time.localtime(tin))

def check_train_status(cmd_name, cache_dir, run_name='deepnet'):
    scriptfile = os.path.join(sdir,'opt_' + cmd_name+ '.sh')
    logfile = os.path.join(sdir,'opt_' + cmd_name+ '.log')
    if os.path.exists(scriptfile):
        submit_time = os.path.getmtime(scriptfile)
    else:
        submit_time = np.nan
    if os.path.exists(logfile):
        start_time = os.path.getmtime(logfile)
    else:
        start_time = np.nan

    files = glob.glob(os.path.join(cache_dir, "{}-[0-9]*").format(run_name))
    files.sort(key=os.path.getmtime)
    files = [f for f in files if os.path.splitext(f)[1] in ['.index','']]
    if len(files)>0:
        latest = files[-1]
        latest_model_iter = int(re.search('-(\d*)', latest).groups(0)[0])
        latest_time = os.path.getmtime(latest)
    else:
        latest_model_iter = -1
        latest_time = np.nan

    print('Job:{}, submitted:{}, started:{} latest iter:{} at {}'.format(
          cmd_name, get_tstr(submit_time), get_tstr(start_time),latest_model_iter, get_tstr(latest_time)))


##     ##################        CREATE DBS


## normal dbs

# assert False,'Are you sure?'

exp_name = 'apt_expt'
for view in range(nviews):
    for tndx in range(len(all_models)):
        train_type = all_models[tndx]
        conf = apt.create_conf(lbl_file,view,exp_name,cache_dir,train_type)
        if train_type == 'deeplabcut':
            apt.create_deepcut_db(conf,split=False,use_cache=True)
        elif train_type == 'leap':
            apt.create_leap_db(conf,split=False,use_cache=True)
        else:
            apt.create_tfrecord(conf,split=False,use_cache=True)

## create incremental dbs

# assert False,'Are you sure?'

import json
import os
exp_name = 'db_sz'
lbl = h5py.File(lbl_file,'r')
m_ndx = apt.to_py(lbl['preProcData_MD_mov'].value[0, :].astype('int'))
t_ndx = apt.to_py(lbl['preProcData_MD_iTgt'].value[0, :].astype('int'))
f_ndx = apt.to_py(lbl['preProcData_MD_frm'].value[0, :].astype('int'))

n_mov = lbl['movieFilesAll'].shape[1]
t_arr = []
for ndx in range(n_mov):
    pts = lbl['labeledposTS']
    sz = np.array(lbl[pts[0, ndx]]['size'])[:, 0].astype('int')
    cur_pts = np.zeros(sz).flatten()
    cur_pts[:] = np.nan
    if lbl[pts[0, ndx]]['val'].value.ndim > 1:
        idx = np.array(lbl[pts[0, ndx]]['idx'])[0, :].astype('int') - 1
        val = np.array(lbl[pts[0, ndx]]['val'])[0, :]
        cur_pts[idx] = val
    cur_pts = cur_pts.reshape(np.flipud(sz))
    t_arr.append(cur_pts)

ts_ndx = []
for ndx in range(m_ndx.shape[0]):
    cur_ts = t_arr[m_ndx[ndx]][t_ndx[ndx],f_ndx[ndx],0]
    ts_ndx.append(cur_ts)

ts_ndx = np.array(ts_ndx)

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
            split_file= os.path.join(conf.cachedir,'splitinfo.json')
            with open(split_file,'w') as f:
                json.dump(splits,f)

            conf.splitType = 'predefined'
            if train_type == 'deeplabcut':
                apt.create_deepcut_db(conf, split=True, split_file=split_file,use_cache=True)
            elif train_type == 'leap':
                apt.create_leap_db(conf, split=True, split_file=split_file, use_cache=True)
            else:
                apt.create_tfrecord(conf, split=True, split_file=split_file, use_cache=True)


## create invidual animals dbs


##


##  ###################              RUNNING TRAINING


## NORMAL TRAINING

# assert False,'Are you sure?'
run_type = 'status'

common_conf = {}
common_conf['rrange'] = 10
common_conf['trange'] = 5
common_conf['mdn_use_unet_loss'] = True
common_conf['dl_steps'] = 100000
common_conf['decay_steps'] = 20000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8


for view in range(nviews):

    for train_type in all_models:

        exp_name = 'apt_expt'
        common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file,exp_name, cache_dir)
        end_cmd = 'train -skip_db -use_cache'
        cmd_opts = {}
        cmd_opts['type'] = train_type
        conf_opts = common_conf.copy()
        # conf_opts.update(other_conf[conf_id])
        conf_opts['save_step'] = conf_opts['dl_steps']/20

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
        cmd_name = '{}_view{}_{}'.format(data_type,view,train_type)
        if run_type == 'submit':
            print cur_cmd
            print
            run_jobs(cmd_name,cur_cmd)
        elif run_type == 'status':
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            check_train_status(cmd_name,conf.cachedir)




## DLC augment vs no augment

# assert False,'Are you sure?'

run_type = 'status'
# run_type = 'submit'; redo = False
# gpu_model = 'TeslaV100_SXM2_32GB'
gpu_model = 'GeForceRTX2080Ti'
train_type = 'deeplabcut'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
common_conf = {}
common_conf['rrange'] = 10
common_conf['trange'] = 5
common_conf['mdn_use_unet_loss'] = True
common_conf['dl_steps'] = 100000
common_conf['decay_steps'] = 20000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8

other_conf = [{'dlc_augment':True},{'dlc_augment':False,'dl_steps':300000}]
cmd_str = ['dlc_aug','dlc_noaug']


for view in range(nviews):

    for conf_id in range(len(other_conf)):

        common_cmd = 'APT_interface.py {} -name apt_expt -cache {}'.format(lbl_file,cache_dir)
        end_cmd = 'train -skip_db -use_cache'
        cmd_opts = {}
        cmd_opts['type'] = train_type
        cmd_opts['train_name'] = cmd_str[conf_id]
        conf_opts = common_conf.copy()
        conf_opts.update(other_conf[conf_id])
        conf_opts['save_step'] = conf_opts['dl_steps']/20

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

        cmd_name = '{}_view{}_{}'.format(data_type,view,cmd_str[conf_id])
        if run_type == 'submit':
            print cur_cmd
            print
            run_jobs(cmd_name,cur_cmd,redo)
        elif run_type == 'status':
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
            check_train_status(cmd_name,conf.cachedir,cmd_str[conf_id])



## INCREMENTAL TRAINING

run_type = 'status'
# assert False, 'Are you sure?'
gpu_model = 'GeForceRTX2080Ti'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
common_conf = {}
common_conf['rrange'] = 10
common_conf['trange'] = 5
common_conf['mdn_use_unet_loss'] = True
common_conf['dl_steps'] = 100000
common_conf['decay_steps'] = 20000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8


n_rounds = 8
for ndx in range(n_rounds):
    exp_name = '{}_randsplit_round_{}'.format(data_type,ndx)
    for view in range(nviews):
        for train_type in all_models:

            common_cmd = 'APT_interface.py {} -name {} -cache {}'.format(lbl_file,exp_name, cache_dir)
            end_cmd = 'train -skip_db -use_cache'
            cmd_opts = {}
            cmd_opts['type'] = train_type
            conf_opts = common_conf.copy()
            # conf_opts.update(other_conf[conf_id])
            conf_opts['save_step'] = conf_opts['dl_steps']/20

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
                print cur_cmd
                print
                run_jobs(cmd_name,cur_cmd)
            elif run_type == 'status':
                conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
                check_train_status(cmd_name,conf.cachedir)



##  ###################### GT DBs

for view in range(nviews):
    conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)




## ######################  RESULTS


##



##       #########             EXTRA

import apt_expts
import os

view = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
conf = apt.create_conf(lbl_file, view, 'apt_expt', cache_dir, 'mdn')
# db_file = '/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/apt_expt/train_TF.tfrecords'
db_file = '/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/alice_compare/val_TF.tfrecords'

files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format('deepnet'))
files.sort(key=os.path.getmtime)
files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
if any([a < 0 for a in aa]):
    bb = int(np.where(np.array(aa) < 0)[0]) + 1
    files = files[bb:]
files = [f.replace('.index', '') for f in files]
files = files[-1:]

mdn_out = apt_expts.classify_db_all(conf, db_file, files, 'mdn')

o = mdn_out[0]
dd = np.sqrt(np.sum((o[0] - o[1]) ** 2, axis=-1))
np.percentile(dd,[90,95],axis=0)


files1 = ['/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/alice_compare/bsz_8_lr_100-60000']
import multiResData

tf_iterator = multiResData.tf_reader(conf, db_file, False)
conf.mdn_no_locs_layer = 1
tf_iterator.batch_size = 1
read_fn = tf_iterator.next
pred_fn, close_fn, _ = apt.get_pred_fn('mdn', conf, files1[0])
pred, label, gt_list = apt.classify_db(conf, read_fn, pred_fn, tf_iterator.N)

mdn_out1 = apt_expts.classify_db_all(conf, db_file, files1, 'mdn')

o = mdn_out1[0]
dd1 = np.sqrt(np.sum((o[0] - o[1]) ** 2, axis=-1))
np.percentile(dd,[90,95],axis=0)


##

# CUrrent code on old working stuff

import PoseTools
import re
import tensorflow as tf
import PoseUNet_resnet as PoseURes
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


A = PoseTools.pickle_load('/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/alice_compare/multitarget_bubble_bsz_8_lr_10_traindata')

conf = A[1]
tf.reset_default_graph()
self = PoseURes.PoseUMDN_resnet(conf, name='test_march30')
self.train_data_name = None
self.train_umdn(restore=False)
