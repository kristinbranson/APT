import  matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import os
import APT_interface as apt
import h5py
import subprocess
import yaml
import pickle
import time
import logging
import glob
import scipy.io as sio
import multiResData
import numpy as np
import random
import json
import subprocess
import PoseTools
import stat

methods = ['unet','leap','deeplabcut','openpose']
out_dir = '/groups/branson/bransonlab/mayank/apt_expts/'
nsplits = 3
openpose_dir = '/groups/branson/bransonlab/mayank/apt_expts/open_pose/training'
deepcut_dir = '/groups/branson/bransonlab/mayank/apt_expts/deepcut_orig2'
leap_dir = '/groups/branson/bransonlab/mayank/apt_expts/leap_orig2'
unet_dir = '/groups/branson/home/kabram/PycharmProjects/poseTF'
# deepcut_default_cfg = '/groups/branson/bransonlab/mayank/PoseTF/cache/apt_interface/multitarget_bubble_view0/test_deepcut/pose_cfg.yaml'

deepcut_default_cfg = '/groups/branson/bransonlab/mayank/apt_expts/deepcut_orig2/examples/openfield-Pranav-2018-10-30/config.yaml'
prog_thresholds = [10,100,10]

def create_deepcut_cfg(conf):
    with open(deepcut_default_cfg,'r') as f:
        default_cfg = yaml.load(f)
    default_cfg['dataset'] = 'train_data.p'
    default_cfg['project_path'] = conf.cachedir
    default_cfg['all_joints'] = [[i] for i in range(conf.n_classes)]
    default_cfg['all_joints_names'] = ['part_{}'.format(i) for i in range(conf.n_classes)]
    default_cfg['num_joints'] = conf.n_classes
    default_cfg['init_weights'] = '/groups/branson/bransonlab/mayank/apt_expts/deepcut_orig2/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt'
    with open(os.path.join(conf.cachedir, 'pose_cfg.yaml'), 'w') as f:
        yaml.dump(default_cfg, f)


def get_label_ts(lbl, ndx, trx_ndx, frames):
    pts = np.array(lbl['labeledposTS'])
    sz = np.array(lbl[pts[0, ndx]]['size'])[:, 0].astype('int')
    cur_pts = np.zeros(sz).flatten()
    cur_pts[:] = np.nan
    if lbl[pts[0, ndx]]['val'].value.ndim > 1:
        idx = np.array(lbl[pts[0, ndx]]['idx'])[0, :].astype('int') - 1
        val = np.array(lbl[pts[0, ndx]]['val'])[0, :] - 1
        cur_pts[idx] = val
    cur_pts = cur_pts.reshape(np.flipud(sz))
    if cur_pts.ndim == 4:
        ts = cur_pts[trx_ndx, frames, :, :]
    else:
        ts = cur_pts[frames, :, :]

    return ts


def get_increasing_splits(conf, split_type='random'):
    # creates json files for the xv splits
    local_dirs, _ = multiResData.find_local_dirs(conf)
    lbl = h5py.File(conf.labelfile, 'r')

    info = []
    n_labeled_frames = 0
    for ndx, dir_name in enumerate(local_dirs):
        if conf.has_trx_file:
            trx_files = multiResData.get_trx_files(lbl, local_dirs)
            trx = sio.loadmat(trx_files[ndx])['trx'][0]
            n_trx = len(trx)
        else:
            n_trx = 1

        for trx_ndx in range(n_trx):
            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx)
            ts = get_label_ts(lbl,ndx,trx_ndx,frames)
            mm = [ndx] * frames.size
            tt = [trx_ndx] * frames.size
            cur_trx_info = list(zip(mm, tt, frames.tolist(),ts.tolist()))
            info.extend(cur_trx_info)
            n_labeled_frames += frames.size
    lbl.close()

    if split_type == 'time':
        info = sorted(info, key=lambda x: x[3])
    elif split_type == 'random':
        info = random.shuffle(info)
    else:
        raise ValueError('Incorrect split type for prog')

    return info


def check_db(curm, conf):
    time.sleep(1)
    check_file = os.path.join(conf.cachedir, 'test_train_db.p')
    apt.check_train_db(curm, conf, check_file)
    with open(check_file, 'r') as f:
        F = pickle.load(f)
    out_dir = os.path.join(conf.cachedir, 'train_check')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if F['ims'].max() < 2:
        F['ims'] = F['ims']*255

    for ndx in range(F['ims'].shape[0]):
        plt.close('all')
        f = plt.figure()
        if F['ims'].shape[-1] == 1:
            plt.imshow(F['ims'][ndx,:,:,0].astype('uint8'),'gray')
        else:
            plt.imshow(F['ims'][ndx,:,:,:].astype('uint8'))
        plt.scatter(F['locs'][ndx,:,0],F['locs'][ndx,:,1])
        plt.savefig(os.path.join(out_dir,'train_{}.jpg'.format(ndx)))


def create_db(args):
    H = h5py.File(args.lbl_file,'r')
    nviews = int(apt.read_entry(H['cfg']['NumViews']))
    all_nets = args.nets

    all_split_files = []
    for view in range(nviews):
        if args.split_type is not None and not args.split_type.startswith('prog'):
            cachedir = os.path.join(out_dir, args.name, 'common')
            if not os.path.exists(cachedir):
                os.mkdir(cachedir)
            cachedir = os.path.join(out_dir, args.name, 'common','splits_{}'.format(view))
            if not os.path.exists(cachedir):
                os.mkdir(cachedir)
            conf = apt.create_conf(args.lbl_file, view, args.name, cache_dir=cachedir)
            conf.splitType = args.split_type
            print("Split type is {}".format(conf.splitType))
            if args.do_split:
                train_info, val_info, split_files = apt.create_cv_split_files(conf, nsplits)
            else:
                split_files = [os.path.join(conf.cachedir, 'cv_split_fold_{}.json'.format(ndx)) for ndx in range(nsplits)]
            all_split_files.append(split_files)

    for curm in all_nets:
        for view in range(nviews):

            if args.split_type is None:

                cachedir = os.path.join(out_dir,args.name,'common','{}_view_{}'.format(curm,view),'full')
                conf = apt.create_conf(args.lbl_file, view, args.name, cache_dir=cachedir)
                if not args.only_check:
                    if not os.path.exists(conf.cachedir):
                        os.makedirs(conf.cachedir)
                    if curm == 'unet' or curm == 'openpose':
                        apt.create_tfrecord(conf, False)
                    elif curm == 'leap':
                        apt.create_leap_db(conf, False)
                    elif curm == 'deeplabcut':
                        apt.create_deepcut_db(conf, False)
                        create_deepcut_cfg(conf)
                    else:
                        raise ValueError('Undefined net type: {}'.format(curm))

                check_db(curm, conf)
            elif args.split_type.startswith('prog'):
                split_type = args.split_type[5:]
                all_info = get_increasing_splits(conf, split_type)

                for cur_tr in prog_thresholds:
                    cachedir = os.path.join(out_dir,args.name,'common','{}_view_{}'.format(curm,view),'{}'.format(cur_tr))
                    conf = apt.create_conf(args.lbl_file, view, args.name, cache_dir=cachedir)
                    split_ndx = round(len(all_info)/cur_tr)
                    cur_train = all_info[:split_ndx]
                    cur_val = all_info[split_ndx:]
                    split_file = os.path.join(cachedir,'splitdata.json')
                    with open(split_file,'w') as f:
                        json.dump([cur_train,cur_val],f)
                    if not args.only_check:
                        if curm == 'unet' or curm == 'openpose':
                            apt.create_tfrecord(conf, True, split_file)
                        elif curm == 'leap':
                            apt.create_leap_db(conf, True, split_file)
                        elif curm == 'deeplabcut':
                            apt.create_deepcut_db(conf, True, split_file)
                            create_deepcut_cfg(conf)
                        else:
                            raise ValueError('Undefined net type: {}'.format(curm))
                    check_db(curm, conf)


            else:

                split_files = all_split_files[view]

                for cur_split in range(nsplits):
                    conf.cachedir = os.path.join(out_dir, args.name, 'common', '{}_view_{}'.format(curm,view))
                    if not os.path.exists(conf.cachedir):
                        os.mkdir(conf.cachedir)
                    conf.cachedir = os.path.join(out_dir, args.name, 'common', '{}_view_{}'.format(curm,view), 'cv_{}'.format(cur_split))
                    if not os.path.exists(conf.cachedir):
                        os.mkdir(conf.cachedir)
                    conf.splitType = 'predefined'
                    split_file = split_files[cur_split]
                    if not args.only_check:
                        if curm == 'unet' or curm == 'openpose':
                            apt.create_tfrecord(conf, True, split_file)
                        elif curm == 'leap':
                            apt.create_leap_db(conf, True, split_file)
                        elif curm == 'deeplabcut':
                            apt.create_deepcut_db(conf, True, split_file)
                            create_deepcut_cfg(conf)
                        else:
                            raise ValueError('Undefined net type: {}'.format(curm))
                    check_db(curm, conf)

        base_dir = os.path.join(out_dir, args.name, 'common')
        their_dir = os.path.join(out_dir, args.name, 'theirs')
        our_dir = os.path.join(out_dir, args.name, 'ours')
        our_default_dir = os.path.join(out_dir, args.name, 'ours_default')
        cmd = 'cp -rs {} {}'.format(base_dir, their_dir)
        os.system(cmd)
        cmd = 'cp -rs {} {}'.format(base_dir, our_dir)
        os.system(cmd)
        cmd = 'cp -rs {} {}'.format(base_dir, our_default_dir)
        os.system(cmd)


def train_theirs(args):
    H = h5py.File(args.lbl_file,'r')
    nviews = int(apt.read_entry(H['cfg']['NumViews']))
    all_nets = args.nets

    for curm in all_nets:
        for view in range(nviews):

            if args.split_type is None:

                cachedir = os.path.join(out_dir,args.name,'theirs','{}_view_{}'.format(curm,view),'full')
                singularity_script = os.path.join(cachedir,'singularity.sh')
                singularity_logfile = os.path.join(cachedir,'singularity.log')
                f = open(singularity_script, 'w')
                f.write('#!/bin/bash\n')
                f.write('. /opt/venv/bin/activate\n')

                if curm == 'unet':
                    f.write('cd {}\n'.format(unet_dir))
                    cmd = 'APT_interface.py -view {} -cache {} -type unet {} train -skip_db'.format(view+1, cachedir, args.lbl_file)
                    f.write('python {}'.format(cmd))
                elif curm == 'openpose':
                    f.write('cd {}\n'.format(openpose_dir))
                    cmd = 'train_pose.py {} {} {}'.format(args.lbl_file, cachedir, view)
                    f.write('python {}'.format(cmd))
                elif curm == 'leap':
                    f.write('cd {}\n'.format(leap_dir))
                    data_path = os.path.join(cachedir,'leap_train.h5')
                    cmd = 'leap/training_MK.py {}'.format(data_path)
                    f.write('python {}'.format(cmd))
                elif curm == 'deeplabcut':
                    f.write('cd {}\n'.format(cachedir))
                    cmd = os.path.join(deepcut_dir,'pose-tensorflow','train.py')
                    f.write('python {}'.format(cmd))
                else:
                    raise ValueError('Undefined net type: {}'.format(curm))

                f.close()
                os.chmod(singularity_script, stat.S_IEXEC)
                cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n4 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                    singularity_logfile, singularity_script)  # -n4 because we use 4 preprocessing threads
                subprocess.call(cmd, shell=True)
                print('Submitted job: {}'.format(cmd))

            else:

                for cur_split in range(nsplits):
                    cachedir = os.path.join(out_dir, args.name, '{}_view_{}'.format(curm,view), 'cv_{}'.format(cur_split))
                    singularity_script = os.path.join(cachedir, 'singularity.sh')
                    singularity_logfile = os.path.join(cachedir, 'singularity.log')
                    f = open(singularity_script, 'w')
                    f.write('#!/bin/bash\n')
                    f.write('. /opt/venv/bin/activate\n')

                    args.skip_db = True
                    if curm == 'unet':
                        f.write('cd {}\n'.format(unet_dir))
                        cmd = 'APT_interface_mdn.py {} -view {} -cache {} -type unet train -skip_db'.format(args.lbl_file, view+1,
                                                                                                 cachedir)
                        f.write('python {}'.format(cmd))
                    elif curm == 'openpose':
                        f.write('cd {}\n'.format(openpose_dir))
                        cmd = 'train_pose.py {} {} {}'.format(args.lbl_file, cachedir, view)
                        f.write('python {}'.format(cmd))
                    elif curm == 'leap':
                        f.write('cd {}\n'.format(leap_dir))
                        data_path = os.path.join(cachedir,'leap_train.h5')
                        cmd = 'leap/training_MK.py {}'.format(data_path)
                        f.write('python {}'.format(cmd))
                    elif curm == 'deeplabcut':
                        f.write('cd {}\n'.format(cachedir))
                        cmd = os.path.join(deepcut_dir, 'pose-tensorflow', 'train.py')
                        f.write('python {}'.format(cmd))
                    else:
                        raise ValueError('Undefined net type: {}'.format(curm))

                    f.close()
                    os.chmod(singularity_script, stat.S_IEXEC)
                    cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n4 -gpu "num=1" -q gpu_tesla "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                        singularity_logfile, singularity_script)  # -n4 because we use 4 preprocessing threads
                    subprocess.call(cmd, shell=True)
                    print('Submitted job: {}'.format(cmd))


def train_ours(args):
    H = h5py.File(args.lbl_file,'r')
    nviews = int(apt.read_entry(H['cfg']['NumViews']))
    dir_name = 'ours_default'

    if len(args.nets) == 0:
        all_nets = methods
    else:
        all_nets = args.nets

    for curm in all_nets:
        for view in range(nviews):

            if args.split_type is None:

                cachedir = os.path.join(out_dir,args.name,dir_name,'{}_view_{}'.format(curm,view),'full')
                singularity_script = os.path.join(cachedir,'singularity.sh')
                singularity_logfile = os.path.join(cachedir,'singularity.log')
                f = open(singularity_script, 'w')
                f.write('#!/bin/bash\n')
                f.write('. /opt/venv/bin/activate\n')

                f.write('cd {}\n'.format(unet_dir))
                cmd = 'APT_interface_mdn.py {} -view {} -cache {} -type {} train -skip_db'.format(args.lbl_file, view+1, cachedir, curm)
                if args.whose == 'ours_default':
                    cmd += ' -use_defaults'
                f.write('python {}'.format(cmd))
                f.close()
                os.chmod(singularity_script, 0o755)
                cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n4 -gpu "num=1" -q gpu_tesla "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                    singularity_logfile, singularity_script)  # -n4 because we use 4 preprocessing threads
                subprocess.call(cmd, shell=True)
                print('Submitted job: {}'.format(cmd))

            else:

                for cur_split in range(nsplits):
                    cachedir = os.path.join(out_dir, args.name, '{}_view_{}'.format(curm,view), 'cv_{}'.format(cur_split))
                    singularity_script = os.path.join(cachedir,'singularity.sh')
                    singularity_logfile = os.path.join(cachedir,'singularity.log')
                    f = open(singularity_script, 'w')
                    f.write('#!/bin/bash\n')
                    f.write('. /opt/venv/bin/activate\n')

                    f.write('cd {}\n'.format(unet_dir))
                    cmd = 'APT_interface_mdn.py {} -view {} -cache {} -type {} train -skip_db'.format(args.lbl_file, view+1, cachedir, curm)
                    if args.whose == 'ours_default':
                        cmd += ' -use_defaults'
                    f.write('python {}'.format(cmd))
                    f.close()
                    os.chmod(singularity_script, 0o755)
                    cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n4 -gpu "num=1" -q gpu_tesla "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                        singularity_logfile, singularity_script)  # -n4 because we use 4 preprocessing threads
                    subprocess.call(cmd, shell=True)
                    print('Submitted job: {}'.format(cmd))


def classify_db_all(conf,db_file,model_files,model_type,name='deepnet',distort=False,
                    classify_fcn='classify_db',
                    timer_pred_inner=None,
                    **kwargs):
    cur_out = []
    extra_str = ''
    if model_type not in ['leap', 'openpose', 'sb', 'dpk','leap_orig','mmpose','mdn_joint_fpn'] and not model_type.startswith('mmpose'):
        extra_str = '.index'
    # else:
    #     extra_str = '.h5'
    ts = [os.path.getmtime(f + extra_str) for f in model_files]

    classify_db_fcn = getattr(apt, classify_fcn)

    for mndx, m in enumerate(model_files):
        # pred, label, gt_list = apt.classify_gt_data(conf, curm, out_file, m)
        if db_file.endswith('.json'):
            coco_reader = multiResData.coco_loader(conf, db_file, False, img_dir='')
            read_fn = iter(coco_reader).__next__
            db_len = len(coco_reader)
            conf.img_dim = 3
        else:
            tf_iterator = multiResData.tf_reader(conf, db_file, False)
            tf_iterator.batch_size = 1
            read_fn = tf_iterator.next
            db_len = tf_iterator.N

        pred_fn, close_fn, _ = apt.get_pred_fn(model_type, conf, m,name=name,distort=distort)
                 #                              tmr_pred=timer_pred_inner) # commenting out for now
        ret_list = classify_db_fcn(conf, read_fn, pred_fn, db_len, **kwargs)
        pred, label, gt_list = ret_list[:3]
        if isinstance(pred, dict):
            pred = pred['locs']
        if model_type == 'mdn':
            extra_stuff = ret_list[3:]
        else:
            extra_stuff = ret_list[3:]  # XXX AL but why not
            # extra_stuff = 0
        close_fn()
        gt_list = np.array(gt_list)
        cur_out.append([pred, label, gt_list, m, extra_stuff, ts[mndx]])

    return cur_out

def classify_db_all2(conf, db_file, model_files, model_type,
                     name='deepnet',
                     classify_fcn='classify_db2',
                     timer_pred_inner=None,
                     **kwargs   # fed to classify_db2
                     ):
    cur_out = []
    extra_str = ''
    if model_type not in ['leap', 'openpose', 'sb', 'dpk']:
        extra_str = '.index'
    ts = [os.path.getmtime(f + extra_str) for f in model_files]

    classify_db_fcn = getattr(apt, classify_fcn)

    for mndx, m in enumerate(model_files):
        # pred, label, gt_list = apt.classify_gt_data(conf, curm, out_file, m)
        tf_iterator = multiResData.tf_reader(conf, db_file, False)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        pred_fn, close_fn, _ = apt.get_pred_fn(model_type, conf, m,
                                               name=name,
                                               tmr_pred=timer_pred_inner)
        ret_list = classify_db_fcn(conf, read_fn, pred_fn, tf_iterator.N, **kwargs)
        close_fn()
        ret_list += (m, ts[mndx])
        cur_out.append(ret_list)

    return cur_out

def compute_peformance(args):
    H = h5py.File(args.lbl_file,'r')
    nviews = int(apt.read_entry(H['cfg']['NumViews']))
    dir_name = args.whose

    if len(args.nets) == 0:
        all_nets = methods
    else:
        all_nets = args.nets

    all_preds = {}

    for view in range(nviews):
        db_file = os.path.join(out_dir,args.name, args.gt_name) + '_view{}.tfrecords'.format(view)
        conf = apt.create_conf(args.lbl_file, view, name='a', net_type=all_nets[0], cache_dir=os.path.join(out_dir,args.name,dir_name))
        conf.labelfile = args.gt_lbl
        if not (os.path.exists(db_file) and args.skip_gt_db):
            print('Creating GT DB file {}'.format(db_file))
            apt.create_tfrecord(conf, split=False, on_gt=True, db_files=(db_file,))

    for curm in all_nets:
        all_preds[curm] = []
        for view in range(nviews):
            cur_out = []
            db_file = os.path.join(out_dir, args.name, args.gt_name) + '_view{}.tfrecords'.format(view)
            if args.split_type is None:
                cachedir = os.path.join(out_dir,args.name,dir_name,'{}_view_{}'.format(curm,view),'full')
                conf = apt.create_conf(args.lbl_file, view, name='a',net_type=curm, cache_dir=cachedir)
                model_files, ts = get_model_files(conf, cachedir, curm)
                for mndx, m in enumerate(model_files):
                    out_file = m + '_' + args.gt_name
                    load = False
                    if curm == 'unet' or curm == 'deeplabcut':
                        mm = m + '.index'
                    else:
                        mm = m
                    if os.path.exists(out_file + '.mat') and os.path.getmtime(out_file + '.mat')> os.path.getmtime(mm):
                        load = True

                    if load:
                        H = sio.loadmat(out_file)
                        pred = H['pred_locs'] - 1
                        label = H['labeled_locs'] - 1
                        gt_list = H['list'] - 1
                    else:
                        # pred, label, gt_list = apt.classify_gt_data(conf, curm, out_file, m)
                        tf_iterator = multiResData.tf_reader(conf, db_file, False)
                        tf_iterator.batch_size = 1
                        read_fn = tf_iterator.next
                        pred_fn, close_fn, _ = apt.get_pred_fn(curm, conf, m)
                        pred, label, gt_list = apt.classify_db(conf, read_fn, pred_fn, tf_iterator.N)
                        close_fn()
                        mat_pred_locs = pred+ 1
                        mat_labeled_locs = np.array(label) +1
                        mat_list = gt_list

                        sio.savemat(out_file,{'pred_locs':mat_pred_locs,
                          'labeled_locs': mat_labeled_locs,
                          'list': mat_list})

                    cur_out.append([pred, label, gt_list, m, out_file, view, 0,ts[mndx]])

            else:

                for cur_split in range(nsplits):
                    cachedir = os.path.join(out_dir, args.name, '{}_view_{}'.format(curm,view), 'cv_{}'.format(cur_split))
                    conf = apt.create_conf(args.lbl_file, view, name='a',net_type=curm, cache_dir=cachedir)
                    model_files, ts = get_model_files(conf, cachedir, curm)
                    db_file = os.path.join(cachedir,'val_TF.tfrecords')
                    for mndx, m in enumerate(model_files):
                        out_file = m + '.gt_data'
                        load = False
                        if curm == 'unet' or curm == 'deeplabcut':
                            mm = m + '.index'
                        else:
                            mm = m
                        if os.path.exists(out_file + '.mat') and os.path.getmtime(out_file + '.mat') > os.path.getmtime(mm):
                            load = True

                        if load:
                            H = sio.loadmat(out_file)
                            pred = H['pred_locs'] - 1
                            label = H['labeled_locs'] - 1
                            gt_list = H['list'] - 1
                        else:
                            tf_iterator = multiResData.tf_reader(conf, db_file, False)
                            tf_iterator.batch_size = 1
                            read_fn = tf_iterator.next
                            pred_fn, close_fn, _ = apt.get_pred_fn(curm, conf, m)
                            pred, label, gt_list = apt.classify_db(conf, read_fn, pred_fn, tf_iterator.N)
                            close_fn()
                            mat_pred_locs = pred + 1
                            mat_labeled_locs = np.array(label) + 1
                            mat_list = gt_list

                            sio.savemat(out_file, {'pred_locs': mat_pred_locs,
                                                   'labeled_locs': mat_labeled_locs,
                                                   'list': mat_list})

                        cur_out.append([pred, label, gt_list, m, out_file, view, cur_split,ts[mndx]])

            all_preds[curm].append(cur_out)

    with open(os.path.join(out_dir,args.name,dir_name,args.gt_name + '_results.p'),'w') as f:
        pickle.dump(all_preds,f)


def get_model_files(conf, cache_dir, method):
    if method == 'unet':
#files = glob.glob(os.path.join(cache_dir,"{}_pose_unet-[0-9]*.index").format(conf.expname,conf.view))
        files = glob.glob(os.path.join(cache_dir,"{}_deepnet-[0-9]*.index").format(conf.expname,conf.view))
        files.sort(key=os.path.getmtime)
        ts = [os.path.getmtime(f) for f in files]
        files = [os.path.splitext(f)[0] for f in files]

    elif method == 'deeplabcut':
        files = glob.glob(os.path.join(cache_dir, "snapshot-[0-9]*.index"))
        files.sort(key=os.path.getmtime)
        ts = [os.path.getmtime(f) for f in files]
        files = [os.path.splitext(f)[0] for f in files]

    elif method == 'leap':
        files = glob.glob(os.path.join(cache_dir, "weights-[0-9]*.h5"))
        files.sort(key=os.path.getmtime)
        ts = [os.path.getmtime(f) for f in files]

    elif method == 'openpose':
        files = glob.glob(os.path.join(cache_dir, "weights.[0-9]*.h5"))
        files.sort(key=os.path.getmtime)
        ts = [os.path.getmtime(f) for f in files]

    init_t = ts[0]
    ts = [t-init_t for t in ts]
    return files, ts

def do_opt(lbl_file, cache_dir, param_dict, name, local=True):
    for k in param_dict.keys():
        for ndx, v in enumerate(param_dict[k]):
            exp_name = '{}_{}'.format(k,ndx)
            cmd = 'APT_interface.py {} -name {} -cache {} -exp_name {} -type mdn -conf_params {} {} train -skip_db'.format(lbl_file,name,cache_dir, exp_name, k, v, )
            if local:
                subprocess.call('python',cmd)
            else:
                PoseTools.submit_job(exp_name,cmd,cache_dir)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', dest='type', help='Create DB or train',
                        required=True, choices=['train','create_db','performance'])
    parser.add_argument("-lbl_file",
                        help="path to lbl file", required=True)
    parser.add_argument("-gt_lbl",
                        help="label file with GT data", default=None)
    parser.add_argument('-name', dest='name', help='Name for the setup',
                        required=True)
    parser.add_argument('-split_type', dest='split_type',
                        help='Type of split for CV. If not defined not CV is done', default=None)
    parser.add_argument('-do_split', dest='do_split', help='Create split files for CV', action='store_true')
    parser.add_argument('-whose', dest='whose',
                        help='Use their or our code', required=True, choices=['theirs','ours','our_default'])
    parser.add_argument('-nets', dest='nets', help='Type of nets to run on. Options are unet, openpose, deeplabcut and leap. If not specified run on all nets', default = [], nargs = '*')
    parser.add_argument('-only_check_db', dest='only_check', help='Only check the db and do not regenerate them', action='store_true' )
    parser.add_argument('-gt_name', dest='gt_name', help='Name for GT data', default='gt')
    parser.add_argument('-skip_gt_db', dest='skip_gt_db', help='Skip GT DB if it exists', action='store_true')

    args = parser.parse_args(argv)
    if args.gt_lbl is None:
        args.gt_lbl = args.lbl_file
    log = logging.getLogger()  # root logger
    log.setLevel(logging.ERROR)

    if len(args.nets) == 0:
        args.nets = methods
    if args.type == 'create_db':
        create_db(args)
    elif args.type == 'train':
        if args.whose == 'theirs':
            train_theirs(args)
        else:
            train_ours(args)
    elif args.type == 'performance':
        compute_peformance(args)


if __name__ == "__main__":
    main(sys.argv[1:])
