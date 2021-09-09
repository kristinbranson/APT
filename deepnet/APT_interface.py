from __future__ import division
from __future__ import print_function

import os

os.environ['DLClight'] = 'False'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import logging

ll = logging.getLogger('matplotlib')
ll.setLevel(logging.WARNING)

import shlex
import argparse
import collections
import datetime
import json
import contextlib
import itertools

from os.path import expanduser
from random import sample

# import PoseUNet
import PoseUNet_dataset as PoseUNet
import PoseUNet_resnet as PoseURes
import hdf5storage
import imageio
import multiResData
from multiResData import float_feature, int64_feature, bytes_feature, trx_pts, check_fnum
# from multiResData import *
import leap.training
from leap.training import train as leap_train
# import open_pose as op
# import sb1 as sb
from deeplabcut.pose_estimation_tensorflow.train import train as deepcut_train
import deeplabcut.pose_estimation_tensorflow.train
import ast
import tempfile
import tensorflow

vv = [int(v) for v in tensorflow.__version__.split('.')]
if vv[0] == 1 and vv[1] > 12:
    tf = tensorflow.compat.v1
elif vv[0] == 2:
    tf = tensorflow.compat.v1
    tf.disable_v2_behavior()
else:
    tf = tensorflow

import sys
import h5py
import numpy as np
import os
import movies
import PoseTools
import pickle
import math
import cv2
import re
from scipy import io as sio
# import heatmap
import time  # for timing between writing n frames tracked
import tarfile
import urllib
import getpass
import link_trajectories as lnk
from matplotlib.path import Path
from PoseCommon_pytorch import coco_loader
from tqdm import tqdm
import shapely.geometry
import TrkFile
from scipy.ndimage import uniform_filter
import multiprocessing

ISPY3 = sys.version_info >= (3, 0)
N_TRACKED_WRITE_INTERVAL_SEC = 10  # interval in seconds between writing n frames tracked

try:
    user = getpass.getuser()
except KeyError:
    user = 'err'
if ISPY3 and user != 'ubuntu' and vv[0] == 1:  # AL 20201111 exception for AWS; running on older AMI
    import apt_dpk


def savemat_with_catch_and_pickle(filename, out_dict):
    try:
        # logging.info('Saving to mat file %s using hdf5storage.savemat'%filename)
        # sio.savemat(filename, out_dict, appendmat=False)
        hdf5storage.savemat(filename, out_dict, appendmat=False, truncate_existing=True)
    except Exception as e:
        logging.info('Exception caught saving mat-file {}: {}'.format(filename, e))
        logging.info('Pickling to {}...'.format(filename))
        with open(filename, 'wb') as fh:
            pickle.dump(out_dict, fh)


def loadmat(filename):
    """this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    From: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True, appendmat=False)
    return _check_keys(data)


def _check_keys(dict_in):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict_in:
        if isinstance(dict_in[key], sio.matlab.mio5_params.mat_struct):
            dict_in[key] = _todict(dict_in[key])
    return dict_in


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    cur_dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            cur_dict[strg] = _todict(elem)
        else:
            cur_dict[strg] = elem
    return cur_dict


def h5py_isstring(x):
    if type(x) is h5py._hl.dataset.Dataset:
        is_str1 = x.dtype == np.dtype('uint64') and len(x.shape) == 1 and x.shape[0] > 1
        is_str2 = x.attrs['MATLAB_class'] == b'char'
        return is_str1 or is_str2
    else:
        return False


def read_entry(x):
    if type(x) is h5py._hl.dataset.Dataset:
        return x[0, 0]
    else:
        return x

def read_list(x):
    return x[()][:,0]

def has_entry(x, s):
    if type(x) is h5py._hl.dataset.Dataset:
        return False
    else:
        return s in x.keys()


def read_string(x):
    if type(x) is h5py._hl.dataset.Dataset:
        if len(x) == 2 and x[0] == 0 and x[1] == 0:  # empty ML strings returned this way
            return ''
        else:
            return ''.join(chr(c) for c in x)
    else:
        return x


def has_trx_file(x):
    if type(x) is h5py._hl.dataset.Dataset:
        if len(x.shape) == 1:
            has_trx = False
        else:
            has_trx = True
    else:
        if x.size == 0:
            has_trx = False
        else:
            has_trx = True
    return has_trx


def datetime2matlabdn(dt=datetime.datetime.now()):
    mdn = dt + datetime.timedelta(days=366)
    frac_seconds = (dt - datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds


def convert(in_data, to_python):
    if type(in_data) in [list, tuple]:
        out_data = []
        for i in in_data:
            out_data.append(convert(i, to_python))
    elif type(in_data) is dict:
        out_data = {}
        for i in in_data.keys():
            out_data[i] = convert(in_data[i], to_python)
    elif in_data is None:
        out_data = None
    else:
        offset = -1 if to_python else 1
        out_data = in_data + offset
    return out_data


def parse_frame_arg(framein, nMovies, defaultval):
    if framein == []:
        frameout = [defaultval] * nMovies
    else:
        if not isinstance(framein, list):
            if framein < 0:
                frameout = [np.Inf] * nMovies
            else:
                frameout = [framein] * nMovies
        elif len(framein) == 1:
            frameout = framein * nMovies
        else:
            frameout = list(map(lambda x: np.Inf if (x < 0) else x, framein))
            if len(frameout) < nMovies:
                frameout = frameout + [defaultval] * (nMovies - len(frameout))
    assert len(frameout) == nMovies
    return frameout


def to_py(in_data):
    return convert(in_data, to_python=True)


def to_mat(in_data):
    return convert(in_data, to_python=False)


def tf_serialize(data):
    # serialize data for writing to tf records file.
    frame_in = data['im']
    cur_loc = data['locs']
    info = data['info']
    # frame_in, cur_loc, info = data[:3]
    if 'occ' in data.keys():
        occ = data['occ']
    else:
        occ = np.zeros(cur_loc.shape[:-1])
    if 'roi' in data.keys():
        rois = data['roi']
    else:
        rois = None
    if 'extra_roi' in data.keys():
        erois = data['extra_roi']
        if erois is not None:
            rois = np.concatenate([rois, erois], 0)

    ntgt = cur_loc.shape[0]
    rows, cols, depth = frame_in.shape
    expid, fnum, trxid = info
    image_raw = frame_in.tostring()

    feature = {
        'height': int64_feature(rows),
        'width': int64_feature(cols),
        'depth': int64_feature(depth),
        'trx_ndx': int64_feature(trxid),
        'locs': float_feature(cur_loc.flatten()),
        'expndx': float_feature(expid),
        'ts': float_feature(fnum),
        'image_raw': bytes_feature(image_raw),
        'occ': float_feature(occ.flatten()),
        'ntgt': int64_feature(ntgt)
    }
    if rois is not None:
        mask = create_mask(rois, frame_in.shape[:2])
        feature['mask'] = bytes_feature(mask.tostring())
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def create_tfrecord(conf, split=True, split_file=None, use_cache=True, on_gt=False, db_files=(), max_nsamples=np.Inf, use_gt_cache=False,db_dict=None):
    # function that creates tfrecords using db_from_lbl
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)

    if on_gt:
        train_filename = db_files[0]
        os.makedirs(os.path.dirname(db_files[0]), exist_ok=True)
        env = tf.python_io.TFRecordWriter(train_filename)
        val_env = None
        envs = [env, val_env]
    elif len(db_files) > 1:
        train_filename = db_files[0]
        env = tf.python_io.TFRecordWriter(train_filename)
        val_filename = db_files[1]
        val_env = tf.python_io.TFRecordWriter(val_filename)
        envs = [env, val_env]
    elif len(db_files)==1:
        train_filename = db_files[0]
        env = tf.python_io.TFRecordWriter(train_filename)
        venv = tf.python.io.TFRecordWriter(tempfile.mkstemp()[1])
        envs = [env,venv]
    else:
        try:
            envs = multiResData.create_envs(conf, split)
        except IOError:
            logging.exception('DB_WRITE: Could not write to tfrecord database')
            exit(1)

    out_fns = [lambda data: envs[0].write(tf_serialize(data)),
               lambda data: envs[1].write(tf_serialize(data))]
    if use_cache:
        splits, __ = db_from_cached_lbl(conf, out_fns, split, split_file, on_gt, use_gt_cache=use_gt_cache)
    else:
        splits = db_from_lbl(conf, out_fns, split, split_file, on_gt, max_nsamples=max_nsamples,db_dict=db_dict)

    envs[0].close()
    envs[1].close() if envs[1] is not None else None
    try:
        with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
            json.dump(splits, f)
    except IOError:
        logging.warning('SPLIT_WRITE: Could not output the split data information')


def convert_to_coco(coco_info, ann, data, conf):
    '''
     converts the data as [img,locs,info,occ] into coco compatible format and adds it to ann

    Write im to coco_info['imdir']; add a single image with 1+ corresponding labeled targets
    to ann

    :param coco_info: dict with keys ndx, ann_ndx, imdir. modified in-place
    :param ann:
    :param data: dict with keys im, locs [ntgt x npts x 2], info (mft triplet),
                                occ [ntgt x npts], roi (opt)
    :param conf:
    :return:
    '''

    cur_im = data['im']
    cur_locs = data['locs']  # [ntgt x ...]
    info = data['info']
    cur_occ = data['occ']
    if cur_locs.ndim == 2:
        # for ht trx sort of thing.
        cur_locs = cur_locs[None, ...]
        cur_occ = cur_occ[None, ...]
    # cur_im,cur_locs,info,cur_occ = data[:4]
    if 'roi' in data.keys():
        roi = data['roi']  # [ntgt x ncol] np array
    else:
        roi = None

    if 'extra_roi' in data.keys():
        extra_roi = data['extra_roi']
    else:
        extra_roi = None

    ndx = coco_info['ndx']
    coco_info['ndx'] += 1
    imfile = os.path.join(coco_info['imdir'], '{:08d}.png'.format(ndx))
    if cur_im.shape[2] == 1:
        cv2.imwrite(imfile, cur_im[:, :, 0])
    else:
        cur_im = cv2.cvtColor(cur_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imfile, cur_im)

    ann['images'].append(
        {'id': ndx, 'width': cur_im.shape[1], 'height': cur_im.shape[0], 'file_name': imfile, 'movid': info[0],
         'frm': info[1], 'patch': info[2]})
    for idx in range(cur_locs.shape[0]):
        annid = coco_info['ann_ndx']
        coco_info['ann_ndx'] += 1
        ix = cur_locs[idx, ...]
        if np.all(ix < -1000) or np.all(np.isnan(ix)):
            continue
        occ_coco = 2 - cur_occ[idx, ..., np.newaxis]
        occ_coco[np.isnan(ix[..., 0]), :] = 0
        if roi is None:
            lmin = ix.min(axis=0)
            lmax = ix.max(axis=0)
            w = lmax[0] - lmin[0]
            h = lmax[1] - lmin[1]
            bbox = [lmin[0], lmin[1], w, h]
            segm = [bbox]
        else:
            lmin = roi[idx].min(axis=0).astype('float64')
            lmax = roi[idx].max(axis=0).astype('float64')
            w = lmax[0] - lmin[0]
            h = lmax[1] - lmin[1]
            bbox = [lmin[0], lmin[1], w, h]
            segm = [roi[idx].flatten().tolist()]
        area = w * h
        out_locs = np.concatenate([ix, occ_coco], 1)
        ann['annotations'].append({'iscrowd': 0, 'segmentation': segm, 'area': area, 'image_id': ndx, 'id': annid,
                                   'num_keypoints': cur_locs.shape[1], 'bbox': bbox,
                                   'keypoints': out_locs.flatten().tolist(), 'category_id': 1})

    if extra_roi is not None:
        for cur_roi in extra_roi:
            annid = coco_info['ann_ndx']
            coco_info['ann_ndx'] += 1
            lmin = cur_roi.min(axis=0)
            lmax = cur_roi.max(axis=0)
            w = lmax[0] - lmin[0]
            h = lmax[1] - lmin[1]
            bbox = [lmin[0], lmin[1], w, h]
            segm = [cur_roi.flatten().tolist()]
            if conf.multi_only_ht:
                out_locs = np.zeros([2, 3])
            else:
                out_locs = np.zeros([conf.n_classes, 3])
            ann['annotations'].append({'iscrowd': 1, 'segmentation': segm, 'area': 1, 'image_id': ndx, 'id': annid,
                                       'num_keypoints': conf.n_classes, 'bbox': bbox,
                                       'keypoints': out_locs.flatten().tolist(), 'category_id': 2})


def create_coco_db(conf, split=True, split_file=None, on_gt=False, db_files=(), max_nsamples=np.Inf,use_cache=True,db_dict=None):
    # function that creates tfrecords using db_from_lbl
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)

    if on_gt:
        train_filename = db_files[0]
        os.makedirs(os.path.dirname(db_files[0]), exist_ok=True)
    elif len(db_files) > 1:
        train_filename = db_files[0]
        val_filename = db_files[1]
    else:
        train_filename = os.path.join(conf.cachedir, conf.trainfilename)
        val_filename = os.path.join(conf.cachedir, conf.valfilename)

    skeleton = [[i, i + 1] for i in range(conf.n_classes - 1)]
    names = ['pt_{}'.format(i) for i in range(conf.n_classes)]
    categories = [{'id': 1, 'skeleton': skeleton, 'keypoints': names, 'super_category': 'fly', 'name': 'fly'}, {'id': 2,
                                                                                                                'super_category': 'neg_box',
                                                                                                                'name': 'neg_box'}]

    train_ann = {'images': [], 'info': [], 'annotations': [], 'categories': categories}
    train_info = {'ndx': 0, 'ann_ndx': 0, 'imdir': os.path.join(conf.cachedir, 'train')}
    val_ann = {'images': [], 'info': [], 'annotations': [], 'categories': categories}
    val_info = {'ndx': 0, 'ann_ndx': 0, 'imdir': os.path.join(conf.cachedir, 'val')}
    os.makedirs(os.path.join(conf.cachedir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(conf.cachedir, 'val'), exist_ok=True)

    out_fns = [lambda data: convert_to_coco(train_info, train_ann, data, conf),
               lambda data: convert_to_coco(val_info, val_ann, data, conf)]
    if use_cache:
        splits, __ = db_from_cached_lbl(conf, out_fns, split, split_file, on_gt)
    else:
        splits = db_from_lbl(conf, out_fns, split, split_file, on_gt, max_nsamples=max_nsamples, db_dict=db_dict)

    with open(train_filename + '.json', 'w') as f:
        json.dump(train_ann, f)
    if split or len(splits) > 1:
        with open(val_filename + '.json', 'w') as f:
            json.dump(val_ann, f)

    try:
        with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
            json.dump(splits, f)
    except IOError:
        logging.warning('SPLIT_WRITE: Could not output the split data information')


def to_orig(conf, locs, x, y, theta):
    ''' locs, x and y should be 0-indexed'''

    # tt = -theta - math.pi / 2
    # hsz_p = conf.imsz[0] // 2  # half size for pred
    # r_mat = [[np.cos(tt), -np.sin(tt)], [np.sin(tt), np.cos(tt)]]
    # curlocs = np.dot(locs - [hsz_p, hsz_p], r_mat) + [x, y]

    theta = -theta - math.pi / 2
    psz_x = conf.imsz[1]
    psz_y = conf.imsz[0]

    if conf.trx_align_theta:
        T = np.array([[1, 0, 0], [0, 1, 0],
                      [x - float(psz_x) / 2 + 0.5, y - float(psz_y) / 2 + 0.5, 1]]).astype('float')
        R1 = cv2.getRotationMatrix2D((float(psz_x) / 2 - 0.5, float(psz_y) / 2 - 0.5), theta * 180 / math.pi, 1)
        R = np.eye(3)
        R[:, :2] = R1.T
        A_full = np.matmul(R, T)
    else:
        x = np.round(x)
        y = np.round(y)
        T = np.array([[1, 0, 0], [0, 1, 0],
                      [x - float(psz_x) / 2 + 0.5, y - float(psz_y) / 2 + 0.5, 1]]).astype('float')
        A_full = T

    lr = np.matmul(A_full[:2, :2].T, locs.T) + A_full[2, :2, np.newaxis]
    curlocs = lr.T

    return curlocs


def convert_to_orig(base_locs, conf, fnum, cur_trx, crop_loc):
    '''converts locs in cropped image back to locations in original image. base_locs need to be in 0-indexed py.
    base_locs should be 2 dim.
    crop_loc should be 0-indexed
    fnum should be 0-indexed'''
    if conf.has_trx_file or conf.use_ht_trx or conf.use_bbox_trx:
        trx_fnum = fnum - int(cur_trx['firstframe'][0, 0] - 1)
        x = to_py(cur_trx['x'][0, trx_fnum])
        y = to_py(cur_trx['y'][0, trx_fnum])
        theta = cur_trx['theta'][0, trx_fnum]
        # assert conf.imsz[0] == conf.imsz[1]
        base_locs_orig = to_orig(conf, base_locs, x, y, theta)
    elif crop_loc is not None:
        xlo, xhi, ylo, yhi = crop_loc
        base_locs_orig = base_locs.copy()
        base_locs_orig[:, 0] += xlo
        base_locs_orig[:, 1] += ylo
    else:
        base_locs_orig = base_locs.copy()
    return base_locs_orig


def get_matlab_ts(filename):
    # matlab's time is different from python
    k = datetime.datetime.fromtimestamp(os.path.getmtime(filename))
    return datetime2matlabdn(k)


def convert_unicode(data):
    if (not ISPY3) and isinstance(data, basestring):
        return unicode(data)
    elif ISPY3 and isinstance(data, str):
        return data
    elif isinstance(data, collections.Mapping):
        if ISPY3:
            return dict(map(convert_unicode, data.items()))
        else:
            return dict(map(convert_unicode, data.iteritems()))
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert_unicode, data))
    else:
        return data


def write_hmaps(hmaps, hmaps_dir, trx_ndx, frame_num, extra_str=''):
    ''''trx_ndx and frame_num are 0-indexed'''
    for bpart in range(hmaps.shape[-1]):
        cur_out = os.path.join(hmaps_dir, 'hmap_trx_{}_t_{}_part_{}{}.jpg'.format(trx_ndx + 1, frame_num + 1, bpart + 1,
                                                                                  extra_str))
        cur_im = hmaps[:, :, bpart]
        cur_im = ((np.clip(cur_im, -1 + 1. / 128, 1) * 128) + 127).astype('uint8')
        imageio.imwrite(cur_out, cur_im, 'jpg', quality=75)
        # cur_out_png = os.path.join(hmaps_dir,'hmap_trx_{}_t_{}_part_{}.png'.format(trx_ndx+1,frame_num+1,bpart+1))
        # imageio.imwrite(cur_out_png,cur_im)

    # mat_out = os.path.join(hmaps_dir, 'hmap_trx_{}_t_{}_{}.mat'.format(trx_ndx + 1, frame_num + 1, extra_str))
    # hdf5storage.savemat(mat_out,{'hm':hmaps})


def get_net_type(lbl_file):
    lbl = h5py.File(lbl_file, 'r')
    dt_params_ndx = None
    for ndx in range(lbl['trackerClass'].shape[0]):
        cur_tracker = ''.join([chr(c) for c in lbl[lbl['trackerClass'][ndx][0]]])
        if cur_tracker == 'DeepTracker':
            dt_params_ndx = ndx

    if 'trnNetTypeString' in lbl[lbl['trackerData'][dt_params_ndx][0]].keys():
        return read_string(lbl[lbl['trackerData'][dt_params_ndx][0]]['trnNetTypeString'])

    elif 'sPrm' in lbl[lbl['trackerData'][dt_params_ndx][0]].keys():
        logging.info('Stripped lbl file created using pre-20190214 APT code. Trying to read netType from sPrm')
        dt_params = lbl[lbl['trackerData'][dt_params_ndx][0]]['sPrm']

        if 'netType' in dt_params.keys():
            return read_string(dt_params['netType'])
        else:
            logging.info('Failed to read netType from lbl file')
            return None
    else:
        logging.info('Failed to read netType from lbl file')
        return None


def flatten_dict(din, dout=None, parent_keys=None, sep='_'):
    if dout is None:  # don't use dout={} as a default arg; default args eval'ed only at module load leading to stateful behavior
        dout = {}

    if parent_keys is None:
        parent_keys = []

    for k, v in din.items():
        k0 = k
        if k in dout:
            for i in range(len(parent_keys)):
                k = parent_keys[-i] + sep + k
                if k not in dout:
                    break

        assert k not in dout, "Unable to flatten dict: repeated key {}".format(k)

        try:
            dout = flatten_dict(v, dout=dout, parent_keys=parent_keys + [k0], sep=sep)
        except (AttributeError, TypeError):
            # logging.info('dout[%s]= %s'%(k,str(v)))
            dout[k] = v

    return dout


def conf_opts_dict2pvargstr(conf_opts):
    '''
    Convert a dict of conf opts to a pv-string to be printed to cmdline
    String vals in dict should be double-escaped
    :param conf_opts: dict
    :return: str
    '''

    if len(conf_opts) > 0:
        conf_str = ' -conf_params'
        for k in conf_opts.keys():
            conf_str = '{} {} {} '.format(conf_str, k, conf_opts[k])
    else:
        conf_str = ''

    return conf_str


def conf_opts_pvargstr2list(conf_str):
    '''
    Return pv-list where vals are still strings
    :param conf_str:
    :return:
    '''
    argv = shlex.split(conf_str)
    parser = argparse.ArgumentParser()
    parser.add_argument('-conf_params', default=None, nargs='*')
    args = parser.parse_args(argv)
    confparamslist = args.conf_params
    return confparamslist


def conf_opts_pvargstr2dict(conf_str):
    '''
    Return dict of conf_opts where vals are now true values
    :param conf_str:
    :return:
    '''
    cc = conf_opts_pvargstr2list(conf_str)
    assert len(cc) % 2 == 0, 'Config params should be in pairs of name value'
    props = cc[::2]
    vals = cc[1::2]
    vals = [ast.literal_eval(x) for x in vals]
    conf_opts = dict(zip(props, vals))
    return conf_opts


def create_conf(lbl_file, view, name, cache_dir=None, net_type='unet', conf_params=None, quiet=False, json_trn_file=None,first_stage=False,second_stage=False):

    assert not (first_stage and second_stage), 'Configurations should either for first stage or second stage for multi stage tracking'

    try:
        try:
            lbl = loadmat(lbl_file)
        except (NotImplementedError, ValueError):
            # logging.info('Label file is in v7.3 format. Loading using h5py')
            lbl = h5py.File(lbl_file, 'r')
    except TypeError as e:
        logging.exception('LBL_READ: Could not read the lbl file {}'.format(lbl_file))

    from poseConfig import config
    from poseConfig import parse_aff_graph
    conf = config()
    conf.n_classes = int(read_entry(lbl['cfg']['NumLabelPoints']))
    if lbl['projname'][0] == 0:
        proj_name = 'default'
    else:
        proj_name = read_string(lbl['projname'])

    try:
        proj_file = read_string(lbl['projectFile'])
    except:
        logging.info('Could not read .projectFile from {}'.format(lbl_file))
        proj_file = ''

    conf.view = view
    conf.set_exp_name(proj_name)
    conf.project_file = proj_file
    # conf.cacheDir = read_string(lbl['cachedir'])
    conf.has_trx_file = has_trx_file(lbl[lbl['trxFilesAll'][0, 0]])
    conf.selpts = np.arange(conf.n_classes)
    conf.nviews = int(read_entry(lbl['cfg']['NumViews']))

    if first_stage:
        conf.stage = 'first'
    elif second_stage:
        conf.stage = 'second'
    else:
        conf.stage = None

    # dt_params_ndx = None
    # for ndx in range(lbl['trackerClass'].shape[0]):
    #     cur_tracker = ''.join([chr(c) for c in lbl[lbl['trackerClass'][ndx][0]]])
    #     if cur_tracker == 'DeepTracker':
    #         dt_params_ndx = ndx
    if not (first_stage or second_stage):
        dt_params_ndx = 1
    elif first_stage:
        dt_params_ndx = 0
    elif second_stage:
        dt_params_ndx = 1
    else:
        assert False, 'Unknown stage type'

    # KB 20190214: updates to APT parameter storing
    if 'sPrmAll' in lbl[lbl['trackerData'][dt_params_ndx][0]].keys():
        isModern = True
        dt_params = lbl[lbl['trackerData'][dt_params_ndx][0]]['sPrmAll']['ROOT']
        if second_stage:
            # Find out whether head-tail or bbox detector
            prm = lbl[lbl['trackerData'][0][0]]['sPrmAll']['ROOT']
            if read_entry(flatten_dict(prm)['multi_only_ht']):
                conf.use_ht_trx = True
            else:
                conf.use_bbox_trx = True

    else:
        logging.info('Stripped lbl file created using pre-20190214 APT code. Reading parameters from sPrm')
        dt_params = lbl[lbl['trackerData'][dt_params_ndx][0]]['sPrm']
        isModern = False

    if cache_dir is None:
        if isModern:
            cache_dir = read_string(dt_params['DeepTrack']['Saving']['CacheDir'])
        else:
            cache_dir = read_string(dt_params['CacheDir'])

    conf.cachedir = os.path.join(cache_dir, proj_name, net_type, 'view_{}'.format(view), name)

    if not os.path.exists(conf.cachedir):
        os.makedirs(conf.cachedir)

    # If the project has trx file then we use the crop locs
    # specified by the user. If the project doesnt have trx files
    # then we use the crop size specified by user else use the whole frame.
    if conf.has_trx_file or conf.use_ht_trx or conf.use_bbox_trx:

        if isModern:
            if 'TargetCrop' in dt_params['ImageProcessing']['MultiTarget']:
                width = int(read_entry(dt_params['ImageProcessing']['MultiTarget']['TargetCrop']['Radius'])) * 2 + 1
            else:
                width = int(read_entry(dt_params['MultiAnimal']['TargetCrop']['Radius'])) * 2 + 1
        else:
            # KB 20190212: replaced with preprocessing
            width = int(read_entry(lbl['preProcParams']['TargetCrop']['Radius'])) * 2 + 1
        height = width

        if not isModern:
            if 'sizex' in dt_params:
                oldwidth = int(read_entry(dt_params['sizex']))
                if oldwidth != width:
                    raise ValueError('Tracker parameter sizex does not match preProcParams->TargetCrop->Radius*2 + 1')
                if 'sizey' in dt_params:
                    oldheight = int(read_entry(dt_params['sizey']))
                    if oldheight != height:
                        raise ValueError(
                            'Tracker parameter sizey does not match preProcParams->TargetCrop->Radius*2 + 1')

        conf.imsz = (height, width)
    else:
        if type(lbl[lbl['movieFilesAllCropInfo'][0, 0]]) != h5py._hl.dataset.Dataset:
            # if lbl['cropProjHasCrops'][0, 0] == 1:
            xlo, xhi, ylo, yhi = PoseTools.get_crop_loc(lbl, 0, view)
            conf.imsz = (int(yhi - ylo + 1), int(xhi - xlo + 1))
        else:
            vid_nr = int(read_entry(lbl[lbl['movieInfoAll'][view, 0]]['info']['nr']))
            vid_nc = int(read_entry(lbl[lbl['movieInfoAll'][view, 0]]['info']['nc']))
            conf.imsz = (vid_nr, vid_nc)
    conf.labelfile = lbl_file
    conf.sel_sz = min(conf.imsz)
    if isModern:
        scale = float(read_entry(dt_params['DeepTrack']['ImageProcessing']['scale']))
        conf.adjust_contrast = int(read_entry(dt_params['DeepTrack']['ImageProcessing']['adjustContrast'])) > 0.5
        conf.normalize_img_mean = int(read_entry(dt_params['DeepTrack']['ImageProcessing']['normalize'])) > 0.5
        if 'TargetCrop' in dt_params['ImageProcessing']['MultiTarget']:
            conf.trx_align_theta = bool(read_entry(dt_params['ImageProcessing']['MultiTarget']['TargetCrop']['AlignUsingTrxTheta']))
        elif 'TargetCrop' in dt_params['MultiAnimal']:
            conf.trx_align_theta = bool(read_entry(dt_params['MultiAnimal']['TargetCrop']['AlignUsingTrxTheta']))

        else:
            conf.trx_align_theta = False
    else:
        scale = float(read_entry(dt_params['scale']))
        conf.adjust_contrast = int(read_entry(dt_params['adjustContrast'])) > 0.5
        conf.normalize_img_mean = int(read_entry(dt_params['normalize'])) > 0.5
        conf.trx_align_theta = bool(read_entry(lbl['preProcParams']['TargetCrop']['AlignUsingTrxTheta']))

    conf.rescale = scale

    ex_mov = multiResData.find_local_dirs(conf.labelfile, conf.view)[0][0]

    if 'NumChans' in lbl['cfg'].keys():
        conf.img_dim = int(read_entry(lbl['cfg']['NumChans']))
    else:
        cap = movies.Movie(ex_mov, interactive=False)
        ex_frame = cap.get_frame(0)
        if np.ndim(ex_frame) > 2:
            conf.img_dim = ex_frame[0].shape[2]
        else:
            conf.img_dim = 1
        cap.close()

    try:
        if isModern:
            conf.flipud = int(read_entry(dt_params['DeepTrack']['ImageProcessing']['flipud'])) > 0.5
        else:
            conf.flipud = int(read_entry(dt_params['flipud'])) > 0.5
    except KeyError:
        pass
    try:
        if isModern:
            conf.unet_steps = int(read_entry(dt_params['DeepTrack']['GradientDescent']['dl_steps']))
        else:
            conf.unet_steps = int(read_entry(dt_params['dl_steps']))
        conf.dl_steps = conf.unet_steps
    except KeyError:
        pass
    try:
        if isModern:
            conf.save_td_step = read_entry(dt_params['DeepTrack']['Saving']['display_step'])
        else:
            conf.save_td_step = read_entry(dt_params['display_step'])
    except KeyError:
        pass
    try:
        if isModern:
            bb = read_entry(dt_params['DeepTrack']['DataAugmentation']['brange'])
        else:
            bb = read_entry(dt_params['brange'])
        conf.brange = [-bb, bb]
    except KeyError:
        pass
    try:
        if isModern:
            bb = read_entry(dt_params['DeepTrack']['DataAugmentation']['crange'])
        else:
            bb = read_entry(dt_params['crange'])
        conf.crange = [1 - bb, 1 + bb]
    except KeyError:
        pass
    try:
        if isModern:
            bb = read_entry(dt_params['DeepTrack']['DataAugmentation']['trange'])
        else:
            bb = read_entry(dt_params['trange'])
        conf.trange = bb
    except KeyError:
        pass
    try:
        if isModern:
            bb = read_entry(dt_params['DeepTrack']['DataAugmentation']['rrange'])
        else:
            bb = read_entry(dt_params['rrange'])
        conf.rrange = bb
    except KeyError:
        pass

    # KB 20191218 - use scale_range only if it exists and scale_factor_range does not
    if isModern:
        try:
            conf.use_scale_factor_range = has_entry(dt_params['DeepTrack']['DataAugmentation'],
                                                    'scale_factor_range') or not has_entry(
                dt_params['DeepTrack']['DataAugmentation'], 'scale_range')
        except KeyError:
            pass

    try:
        if isModern and net_type in ['dpk', 'openpose','multi_openpose']:
            try:
                bb = read_string(dt_params['DeepTrack']['OpenPose']['affinity_graph'])
            except ValueError:
                bb = ''
        else:
            bb = ''
        conf.op_affinity_graph = parse_aff_graph(bb) if bb else []
    except KeyError:
        pass
    try:
        bb = read_string(dt_params['DeepTrack']['DataAugmentation']['flipLandmarkMatches'])
        graph = {}
        if bb:
            bb = bb.split(',')
            for b in bb:
                mm = re.search('(\d+)\s+(\d+)', b)
                n1 = int(mm.groups()[0]) - 1
                n2 = int(mm.groups()[1]) - 1
                graph['{}'.format(n1)] = n2
                graph['{}'.format(n2)] = n1
                # The keys have to be strings so that they can be saved in the trk file
        conf.flipLandmarkMatches = graph
    except KeyError:
        pass

    conf.mdn_groups = [(i,) for i in range(conf.n_classes)]

    done_keys = ['CacheDir', 'scale', 'brange', 'crange', 'trange', 'rrange', 'op_affinity_graph', 'flipud', 'dl_steps',
                 'scale', 'adjustContrast', 'normalize', 'sizex', 'sizey', 'flipLandmarkMatches']

    if isModern:
        dt_params_flat = flatten_dict(dt_params)
        # dt_params_flat = flatten_dict(dt_params['DeepTrack'])
    else:
        dt_params_flat = dt_params

    for k in dt_params_flat.keys():
        if k in done_keys:
            continue

        # logging.info('Adding parameter %s'%k)

        if hasattr(conf, k):
            if type(getattr(conf, k)) == str:
                setattr(conf, k, read_string(dt_params_flat[k]))
            else:
                attr_type = type(getattr(conf, k))
                if attr_type == list:
                    setattr(conf,k,attr_type(read_list(dt_params_flat[k])))
                    if k == 'ht_pts':
                        conf.ht_pts = to_py([int(x) for x in conf.ht_pts])
                else:
                    setattr(conf, k, attr_type(read_entry(dt_params_flat[k])))
        else:
            if h5py_isstring(dt_params_flat[k]):
                setattr(conf, k, read_string(dt_params_flat[k]))
            else:
                try:
                    setattr(conf, k, read_entry(dt_params_flat[k]))
                except TypeError:
                    logging.info('Could not parse parameter %s, ignoring' % k)

    conf.json_trn_file = json_trn_file

    if conf_params is not None:
        cc = conf_params
        assert len(cc) % 2 == 0, 'Config params should be in pairs of name value'
        for n, v in zip(cc[0::2], cc[1::2]):
            if not quiet:
                print('Overriding param %s <= ' % n, v)
            setattr(conf, n, ast.literal_eval(v))

    # overrides for each network
    if net_type == 'sb':
        sb.update_conf(conf)
    # elif net_type == 'openpose':
    #     op.update_conf(conf)
    elif net_type == 'dpk':
        if conf.dpk_use_op_affinity_graph:
            apt_dpk.update_conf_dpk_from_affgraph_flm(conf)
        else:
            assert conf.dpk_skel_csv is not None
            apt_dpk.update_conf_dpk_skel_csv(conf, conf.dpk_skel_csv)

    # elif net_type == 'deeplabcut':
    #     conf.batch_size = 1
    elif net_type == 'unet':
        conf.use_pretrained_weights = False

    conf.unet_rescale = conf.rescale
    # conf.op_rescale = conf.rescale  # not used by op4
    # conf.dlc_rescale = conf.rescale
    conf.leap_rescale = conf.rescale

    assert not (
                conf.vert_flip and conf.horz_flip), 'Only one type of flipping, either horizontal or vertical is allowed for augmentation'
    return conf


def test_preproc(lbl_file=None, cachedir=None):
    ''' Compare python preproc pipeline with matlab's'''
    from matplotlib import pyplot as plt
    if lbl_file is None:
        lbl_file = '/home/mayank/temp/apt_cache/multitarget_bubble/20190129T180959_20190129T181147.lbl'
    if cachedir is None:
        cachedir = '/home/mayank/temp/apt_cache'

    conf = create_conf(lbl_file, 0, 'compare_cache', cachedir, 'mdn')

    conf.trainfilename = 'normal'
    n_envs = multiResData.create_envs(conf, False)
    conf.trainfilename = 'cached'
    c_envs = multiResData.create_envs(conf, False)

    n_out_fns = [lambda data: n_envs[0].write(tf_serialize(data)),
                 lambda data: n_envs[1].write(tf_serialize(data))]
    c_out_fns = [lambda data: c_envs[0].write(tf_serialize(data)),
                 lambda data: c_envs[1].write(tf_serialize(data))]

    splits, __ = db_from_cached_lbl(conf, c_out_fns, False, None, False)
    c_envs[0].close()
    splits = db_from_lbl(conf, n_out_fns, False, None, False)
    n_envs[0].close()

    c_file_name = os.path.join(conf.cachedir, 'cached.tfrecords')
    n_file_name = os.path.join(conf.cachedir, 'normal.tfrecords')
    A = []
    A.append(multiResData.read_and_decode_without_session(c_file_name, conf, ()))
    A.append(multiResData.read_and_decode_without_session(n_file_name, conf, ()))

    ims1 = np.array(A[0][0]).astype('float')
    ims2 = np.array(A[1][0]).astype('float')
    locs1 = np.array(A[0][1])
    locs2 = np.array(A[1][1])

    ndx = np.random.choice(ims1.shape[0])
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].imshow(ims1[ndx, :, :, 0], 'gray', vmin=0, vmax=255)
    ax[1].imshow(ims2[ndx, :, :, 0], 'gray', vmin=0, vmax=255)
    ax[0].scatter(locs1[ndx, :, 0], locs1[ndx, :, 1])
    ax[1].scatter(locs2[ndx, :, 0], locs2[ndx, :, 1])


def get_cur_trx(trx_file, trx_ndx):
    if trx_file is None:
        return None, 1
    try:
        trx = sio.loadmat(trx_file)['trx'][0]
        cur_trx = trx[trx_ndx]
        n_trx = len(trx)
    except NotImplementedError:
        # trx file in v7.3 format
        # print('Trx file is in v7.3 format. Loading using h5py')
        trx = h5py.File(trx_file, 'r')['trx']
        cur_trx = {}
        for k in trx.keys():
            cur_trx[k] = np.array(trx[trx[k][trx_ndx, 0]]).T
        n_trx = trx['x'].shape[0]
    return cur_trx, n_trx


def db_from_lbl(conf, out_fns, split=True, split_file=None, on_gt=False, sel=None, max_nsamples=np.Inf,db_dict=None):
    # outputs is a list of functions. The first element writes
    # to the training dataset while the second one write to the validation
    # dataset. If split is False, second element is not used and all data is
    # outputted to training dataset
    # the function will be give a list with:
    # 0: img,
    # 1: locations as a numpy array.
    # 2: information list [expid, frame number, trxid]
    # the function returns a list of [expid, frame_number and trxid] showing
    #  how the data was split between the two datasets.

    # assert not (on_gt and split), 'Cannot split gt data'

    from_list = True if db_dict is not None else False
    if from_list:
        local_dirs = db_dict['moviesFiles']
        trx_files = db_dict['trxFiles']
    else:
        local_dirs = multiResData.find_local_dirs(conf.labelfile, conf.view, on_gt)

    lbl = h5py.File(conf.labelfile, 'r')
    view = conf.view
    flipud = conf.flipud
    occ_as_nan = conf.get('ignore_occluded', False)
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    sel_pts = int(view * npts_per_view) + conf.selpts

    splits = [[], []]
    count = 0
    val_count = 0

    mov_split = None
    predefined = None
    if conf.splitType == 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = PoseTools.json_load(split_file)
    elif conf.splitType == 'movie':
        nexps = len(local_dirs)
        mov_split = sample(list(range(nexps)), int(nexps * conf.valratio))
        predefined = None
    elif conf.splitType == 'trx':
        assert conf.has_trx_file, 'Train/Validation was selected to be trx but the project has no trx files'

    nsamples = 0

    for ndx, dir_name in enumerate(local_dirs):

        if nsamples >= max_nsamples:
            break

        exp_name = conf.getexpname(dir_name)
        if not from_list:
            cur_pts = trx_pts(lbl, ndx, on_gt)
            cur_occ = trx_pts(lbl, ndx, on_gt, field_name='labeledpostag')
            cur_occ = ~np.isnan(cur_occ)
            crop_loc = PoseTools.get_crop_loc(lbl, ndx, view, on_gt)
        else:
            toTrack = db_dict['toTrack']
            cur_sel = [to_py(xx) for xx in toTrack if xx[0]==(ndx-1)]
            nfr = len(cur_sel)
            _, n_trx_list = get_cur_trx(trx_files[ndx])
            cur_pts = np.zeros([n_trx_list,nfr,2,conf.n_classes])
            cur_occ = np.zeros([n_trx_list,nfr,conf.n_classes])
            crop_loc = db_dict['cropLocs'][ndx]

        try:
            cap = movies.Movie(dir_name)
        except ValueError:
            logging.exception('MOVIE_READ: ' + local_dirs[ndx] + ' is missing')
            sys.exit(1)

        if from_list:
            n_trx = n_trx_list
            trx_split = np.random.random(n_trx) < conf.valratio
        elif conf.has_trx_file:
            trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)
            _, n_trx = get_cur_trx(trx_files[ndx], 0)
            trx_split = np.random.random(n_trx) < conf.valratio
        else:
            trx_files = [None, ] * len(local_dirs)
            n_trx = 1
            trx_split = None
            cur_pts = cur_pts[np.newaxis, ...]
            cur_occ = cur_occ[None, ...]

        for trx_ndx in range(n_trx):

            if nsamples >= max_nsamples:
                break

            if from_list:
                frames = [xx[0] for xx in cur_sel if xx[1]==trx_ndx]
            else:
                frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx, on_gt)
            cur_trx, _ = get_cur_trx(trx_files[ndx], trx_ndx)
            for fnum in frames:
                if not check_fnum(fnum, cap, exp_name, ndx):
                    continue

                info = [int(ndx), int(fnum), int(trx_ndx)]
                cur_out = multiResData.get_cur_env(out_fns, split, conf, info, mov_split, trx_split=trx_split, predefined=predefined)

                frame_in, cur_loc = multiResData.get_patch(cap, fnum, conf, cur_pts[trx_ndx, fnum, :, sel_pts], cur_trx=cur_trx, flipud=flipud, crop_loc=crop_loc)

                if occ_as_nan:
                    cur_loc[cur_occ[fnum, :], :] = np.nan

                cur_out({'im':frame_in, 'locs':cur_loc, 'info':info})

                if cur_out is out_fns[1] and split:
                    val_count += 1
                    splits[1].append(info)
                else:
                    count += 1
                    splits[0].append(info)
                nsamples += 1
                if nsamples >= max_nsamples:
                    break

        cap.close()  # close the movie handles
        logging.info('Done %d of %d movies, train count:%d val count:%d' % (ndx + 1, len(local_dirs), count, val_count))

    logging.info('%d,%d number of examples added to the training db and val db' % (count, val_count))
    lbl.close()
    return splits


def setup_ma(conf):
    # setups the crop size. Based on the largets cluster.

    T = PoseTools.json_load(conf.json_trn_file)
    cur_t = T['locdata'][0]
    pack_dir = os.path.split(conf.json_trn_file)[0]
    cur_frame = cv2.imread(os.path.join(pack_dir, cur_t['img'][conf.view]), cv2.IMREAD_UNCHANGED)
    if cur_frame.ndim>2:
        cur_frame = cv2.cvtColor(cur_frame,cv2.COLOR_BGR2RGB)
    fr_sz = cur_frame.shape[:2]
    conf.multi_frame_sz = fr_sz

    max_sz = 0
    for selndx, cur_t in enumerate(T['locdata']):
        ntgt = cur_t['ntgt']
        cur_roi = np.array(cur_t['roi']).reshape([conf.nviews, 2, 4, ntgt])
        cur_roi = np.transpose(cur_roi[conf.view, ...], [2, 1, 0])
        clusters = get_clusters(cur_roi)
        n_cluster = len(np.unique(clusters))
        for cndx in range(n_cluster):
            idx = np.where(clusters == cndx)[0]
            cur_rois = cur_roi[idx, ...]
            x_sz = cur_rois[..., 0].max() - cur_rois[..., 0].min()
            y_sz = cur_rois[..., 1].max() - cur_rois[..., 1].min()
            max_sz = x_sz if x_sz > max_sz else max_sz
            max_sz = y_sz if y_sz > max_sz else max_sz

    max_sz = int(np.ceil((max_sz + 2) / 32)) * 32
    # don't take max in case there is a weird outlier
    conf.multi_max_animal_sz = np.nan

    if not conf.multi_crop_ims:
        conf.imsz = (fr_sz[0], fr_sz[1])
        logging.info(f'--- Not cropping images for multi-animal. Using frame size {fr_sz} as image size ---')
        return
    else:
        logging.info(f'--- Using crops of size {max_sz} for multi-animal training.  ---')
        conf.imsz = (max_sz, max_sz)


def get_clusters(rois):
    # Find bbox to find overlapping clusters
    nlabels = rois.shape[0]
    polys = [shapely.geometry.Polygon(rois[i, ...]) for i in range(nlabels)]
    cluster_ids = np.ones(nlabels, dtype=np.uint) * np.nan
    for ndx in range(nlabels):
        overlap_idx = []
        for ondx in range(nlabels):
            if polys[ndx].intersects(polys[ondx]):
                overlap_idx.append(ondx)

        if len(overlap_idx) > 1:
            clusters = cluster_ids[overlap_idx]
            if np.all(np.isnan(clusters)):
                cluster_ids[overlap_idx] = ndx
            else:
                cid = int(np.nanmin(cluster_ids[overlap_idx]))
                cluster_ids[overlap_idx] = cid
        else:
            # no overlap
            cluster_ids[ndx] = ndx

    # update cluster ids
    cids, indices = np.unique(cluster_ids, return_inverse=True)
    n_clusters = cids.shape[0]
    new_cluster_ids = np.arange(n_clusters)
    cluster_ids = new_cluster_ids[indices]
    return cluster_ids


def create_mask(roi, sz):
    # sz should be h x w (i.e y first then x)
    x, y = np.meshgrid(np.arange(sz[1]), np.arange(sz[0]))
    x = x.flatten()
    y = y.flatten()
    pts = np.vstack((x, y)).T
    grid = None
    for c in roi:
        rr = c.tolist()
        rr.append(rr[0])
        path = Path(rr)
        cgrid = path.contains_points(pts)
        if grid is not None:
            grid = grid | cgrid
        else:
            grid = cgrid

    if grid is None:
        mask = np.zeros(sz) > 0.5
    else:
        mask = grid.reshape(sz)
    return mask


def create_ma_crops(conf, frame, cur_pts, info, occ, roi, extra_roi):
    def random_crop_around_roi(roi_in):
        x_min = np.clip(roi_in[..., 0].min(),0,conf.multi_frame_sz[1])
        y_min = np.clip(roi_in[..., 1].min(),0,conf.multi_frame_sz[0])
        x_max = np.clip(roi_in[..., 0].max(),0,conf.multi_frame_sz[1])
        y_max = np.clip(roi_in[..., 1].max(),0,conf.multi_frame_sz[0])

        d_x = (conf.imsz[1] - (x_max - x_min)) * 0.9
        r_x = (np.random.rand() - 0.5) * d_x
        x_left = int(round((x_max + x_min) / 2 - conf.imsz[1] / 2 + r_x))
        x_left = min(x_left, frame.shape[1] - conf.imsz[1])
        x_left = max(x_left, 0)
        x_right = x_left + conf.imsz[1]

        d_y = (conf.imsz[0] - (y_max - y_min)) * 0.9
        r_y = (np.random.rand() - 0.5) * d_y
        y_top = int(round((y_max + y_min) / 2 - conf.imsz[0] / 2 + r_y))
        y_top = min(y_top, frame.shape[0] - conf.imsz[0])
        y_top = max(y_top, 0)
        y_bottom = y_top + conf.imsz[0]

        assert (y_top-1) <= round(y_min) and (y_bottom+1) >= round(y_max) and (x_left-1) <= round(x_min) and (x_right+1) >= round(x_max), 'Cropping for cluster is improper'
        return x_left, y_top, x_right, y_bottom

    def roi2patch(roi_in, x_left, y_top):
        roi_in = roi_in.copy()
        roi_in[..., 0] = np.clip(roi_in[..., 0] - x_left, 0, conf.imsz[1])
        roi_in[..., 1] = np.clip(roi_in[..., 1] - y_top, 0, conf.imsz[0])
        return roi_in

    def roi_from_patch(roi_in, x_left, y_top):
        roi_in = roi_in.copy()
        roi_in[..., 0] = np.clip(roi_in[..., 0] + x_left, 0, conf.multi_frame_sz[1])
        roi_in[..., 1] = np.clip(roi_in[..., 1] + y_top, 0, conf.multi_frame_sz[0])
        return roi_in

    clusters = get_clusters(roi)
    n_clusters = len(np.unique(clusters))
    all_data = []
    mask_sc = 4
    mask_sz = (conf.multi_frame_sz[0]//mask_sc, conf.multi_frame_sz[1]//mask_sc)
    done_mask = np.zeros(mask_sz) > 1

    roi = roi.copy()
    roi[..., 0] = np.clip(roi[..., 0], 0, conf.multi_frame_sz[1])
    roi[..., 1] = np.clip(roi[..., 1], 0, conf.multi_frame_sz[0])

    if conf.multi_loss_mask or conf.multi_use_mask:
        n_extra_roi = 0 if extra_roi is None else len(extra_roi)
    else:
        n_extra_roi = 1
        extra_roi = np.array([[0,0],[0,conf.multi_frame_sz[0]],[conf.multi_frame_sz[1],conf.multi_frame_sz[0]],[conf.multi_frame_sz[1],0]])[None,...].astype('float')

    if n_extra_roi > 0:
        extra_roi = extra_roi.copy()
        extra_roi[..., 0] = np.clip(extra_roi[..., 0], 0, conf.multi_frame_sz[1])
        extra_roi[..., 1] = np.clip(extra_roi[..., 1], 0, conf.multi_frame_sz[0])

    for cndx in range(n_clusters):
        idx = np.where(clusters == cndx)[0]
        cur_roi = roi[idx, ...].copy()

        x_left, y_top, x_right, y_bottom = random_crop_around_roi(cur_roi)

        curp = frame[y_top:y_bottom, x_left:x_right, :]
        cur_roi[..., 0] -= x_left
        cur_roi[..., 1] -= y_top

        if n_extra_roi > 0:
            cur_eroi = roi2patch(extra_roi, x_left, y_top)
            keep_ndx = ~np.any(np.all(cur_eroi[..., 0:1, :] == cur_eroi, -2), -1)
            cur_eroi = cur_eroi[keep_ndx]
            cur_mask = create_mask(np.concatenate([cur_roi, cur_eroi], 0), [y_bottom - y_top, x_right - x_left])
        else:
            cur_eroi = None
            cur_mask = create_mask(cur_roi, [y_bottom - y_top, x_right - x_left])

        if conf.multi_use_mask:
            curp = curp * cur_mask[..., np.newaxis]

        cur_done_mask = create_mask(roi_from_patch(cur_roi/mask_sc, x_left//mask_sc, y_top//mask_sc), mask_sz)

        if n_extra_roi > 0:
            cur_done_mask = cur_done_mask | create_mask(roi_from_patch(cur_eroi/mask_sc, x_left/mask_sc, y_top/mask_sc), mask_sz)

        done_mask = done_mask | cur_done_mask

        curl = cur_pts[idx, :, :].copy()
        curl[:, :, 0] = curl[:, :, 0] - x_left
        curl[:, :, 1] = curl[:, :, 1] - y_top
        cur_occ = occ[idx, ...]

        all_data.append({'im': curp, 'locs': curl, 'info': [info[0], info[1], cndx], 'occ': cur_occ, 'roi': cur_roi,
                         'extra_roi': cur_eroi, 'x_left': x_left, 'y_top': y_top})

    if n_extra_roi > 0:
        # bkg_sel_rate = conf.background_mask_sel_rate

        done_eroi = np.zeros(n_extra_roi)
        while np.any(done_eroi < 0.5):

            # add examples of background not added earlier.
            for endx in range(n_extra_roi):
                eroi_mask = create_mask(extra_roi[endx:endx + 1, ...]/mask_sc, mask_sz)
                if (eroi_mask & done_mask).sum() / eroi_mask.sum() > 0.5:
                    done_eroi[endx] = 1.
                    continue

                vmsz= [zzz//mask_sc for zzz in conf.imsz]
                valid_locs = uniform_filter((eroi_mask & ~done_mask).astype('float'), size=vmsz, mode='constant')
                yy, xx = np.where(valid_locs == valid_locs.max())
                xx *= mask_sc; yy*=mask_sc
                ix_sel = np.random.choice(len(yy))
                y_top = yy[ix_sel] - conf.imsz[0] // 2
                x_left = xx[ix_sel] - conf.imsz[1] // 2
                y_top = np.clip(y_top, 0, conf.multi_frame_sz[0] - conf.imsz[0])
                x_left = np.clip(x_left, 0, conf.multi_frame_sz[1] - conf.imsz[1])

                cur_eroi = roi2patch(extra_roi, x_left, y_top)

                y_bottom = y_top + conf.imsz[0]
                x_right = x_left + conf.imsz[1]
                curp = frame[y_top:y_bottom, x_left:x_right, :]
                if conf.multi_use_mask:
                    curp = curp * create_mask(cur_eroi, conf.imsz)[..., np.newaxis]

                curl = np.ones_like(cur_pts[0:1, ...]) * np.nan
                cur_occ = np.zeros_like(curl[..., 0])
                cur_roi = np.zeros([1, 4, 2])
                frame_eroi = roi_from_patch(cur_eroi, x_left, y_top)

                if len(cur_eroi) == 0:
                    cur_eroi = None

                done_mask = done_mask | create_mask(frame_eroi/mask_sc, mask_sz)

                all_data.append(
                    {'im': curp, 'locs': curl, 'info': [info[0], info[1], endx], 'occ': cur_occ, 'roi': cur_roi,
                     'extra_roi': cur_eroi, 'x_left': x_left, 'y_top': y_top})

    return all_data


def show_crops(im, all_data, roi, extra_roi, conf):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    plt.ion()

    roim = create_mask(roi, conf.multi_frame_sz)
    eroim = create_mask(extra_roi, conf.multi_frame_sz)
    f = plt.figure()
    plt.imshow(im * (roim | eroim), 'gray')

    f1, ax = plt.subplots(int(np.ceil(len(all_data) / 2)), 2)
    ax = ax.flatten()
    for ndx, a in enumerate(all_data):
        mm = create_mask(a['roi'], conf.imsz)
        if a['extra_roi'] is not None:
            mm = mm | create_mask(a['extra_roi'], conf.imsz)
        ax[ndx].imshow(a['im'] * mm[:, :, None])
        ax[ndx].axis('off')

    plt.figure(f.number)
    plt.axis('off')
    for a in all_data:
        xx = [a['x_left'], a['x_left'], a['x_left'] + conf.imsz[1], a['x_left'] + conf.imsz[1], a['x_left']]
        yy = [a['y_top'], a['y_top'] + conf.imsz[0], a['y_top'] + conf.imsz[0], a['y_top'], a['y_top']]
        plt.plot(xx, yy)


def db_from_trnpack_ht(conf, out_fns, nsamples=None, split=True):
    # TODO: Maybe merge this with db_from_trnpack??
    lbl = h5py.File(conf.labelfile, 'r')
    occ_as_nan = conf.get('ignore_occluded', False)
    T = PoseTools.json_load(conf.json_trn_file)
    nfrms = len(T['locdata'])

    splits = [[] for a in T['splitnames']]
    count = [0 for a in T['splitnames']]

    # KB 20190208: if we only need a few images, don't waste time reading in all of them
    if nsamples is not None:
        T['locdata'] = T['locdata'][np.random.choice(nfrms, nsamples)]
    else:
        sel = np.arange(len(T['locdata']))

    pack_dir = os.path.split(conf.json_trn_file)[0]
    for selndx, cur_t in enumerate(T['locdata']):

        cur_frame = cv2.imread(os.path.join(pack_dir, cur_t['img'][conf.view]), cv2.IMREAD_UNCHANGED)
        if cur_frame.ndim == 2:
            cur_frame = cur_frame[..., np.newaxis]
        else:
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)

        cur_locs = np.array(cur_t['pabs']) - 1
        ntgt = cur_t['ntgt']
        cur_locs = cur_locs.reshape([conf.nviews, 2, conf.n_classes, ntgt])
        cur_locs = np.transpose(cur_locs[conf.view, ...], [2, 1, 0])

        cur_occ = np.array(cur_t['occ'])
        cur_occ = cur_occ.reshape([conf.nviews, conf.n_classes, ntgt])
        cur_occ = cur_occ[conf.view, :]
        cur_occ = np.transpose(cur_occ, [1, 0])
        cur_occ = cur_occ.astype('float')

        cur_roi = np.reshape(cur_t['roi'], [conf.nviews, 2, 4, ntgt])
        cur_roi = np.transpose(cur_roi[conf.view, ...], [2, 1, 0])

        sndx = cur_t['split']
        if type(sndx) == list:
            if len(sndx)<1:
                sndx = 0
            else:
                sndx = sndx[0]
        cur_out = out_fns[sndx]

        for ndx in range(len(cur_locs)):
            info = to_py([cur_t['imov'], cur_t['frm'], ndx + 1])
            if conf.use_ht_trx:
                ht_locs = cur_locs[ndx, conf.ht_pts, :].copy()
                ctr = ht_locs.mean(axis=0)
                theta = np.arctan2(ht_locs[0, 1] - ht_locs[1, 1], ht_locs[0, 0] - ht_locs[1, 0])
            elif conf.use_bbox_trx:
                assert not conf.trx_align_theta, 'Aligning with theta should be off for bbox based top-down tracking'
                ctr = cur_roi[ndx].mean(axis=0)
                theta = 0
            else:
                assert False, 'For top down tracking either head-tail tracking or bbox tracking should be active'

            curl = cur_locs[ndx].copy()
            cur_patch, curl = multiResData.crop_patch_trx(conf, cur_frame, ctr[0], ctr[1], theta, curl)

            if occ_as_nan:
                curl[cur_occ[ndx], :] = np.nan

            data_out = {'im': cur_patch, 'locs': curl, 'info': info, 'occ': cur_occ[ndx,]}
            cur_out(data_out)

            count[sndx] += 1
            splits[sndx].append(info)

            if selndx % 100 == 99 and selndx > 0:
                logging.info('{} number of examples added to the dbs'.format(count))

    logging.info('{} number of examples added to the training dbs'.format(count))
    lbl.close()

    return splits, sel


def db_from_trnpack(conf, out_fns, nsamples=None, split=True,only_ht=True):
    # Creates db from new trnpack format instead of stripped label files.
    # outputs is a list of functions. The first element writes
    # to the training dataset while the second one write to the validation
    # dataset. If split is False, second element is not used and all data is
    # outputted to training dataset
    # the function will be give a list with:
    # 0: img,
    # 1: locations as a numpy array.
    # 2: information list [expid, frame number, trxid]
    # the function returns a list of [expid, frame_number and trxid] showing
    #  how the data was split between the two datasets.

    lbl = h5py.File(conf.labelfile, 'r')
    occ_as_nan = conf.get('ignore_occluded', False)
    T = PoseTools.json_load(conf.json_trn_file)
    nfrms = len(T['locdata'])

    splits = [[] for a in T['splitnames']]
    count = [0 for a in T['splitnames']]

    # KB 20190208: if we only need a few images, don't waste time reading in all of them
    if nsamples is not None:
        T['locdata'] = T['locdata'][np.random.choice(nfrms, nsamples)]
    else:
        sel = np.arange(len(T['locdata']))

    pack_dir = os.path.split(conf.json_trn_file)[0]
    for selndx, cur_t in enumerate(T['locdata']):

        cur_frame = cv2.imread(os.path.join(pack_dir, cur_t['img'][conf.view]), cv2.IMREAD_UNCHANGED)
        if cur_frame.ndim == 2:
            cur_frame = cur_frame[..., np.newaxis]
        else:
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)

        cur_locs = np.array(cur_t['pabs']) - 1
        ntgt = cur_t['ntgt']
        cur_locs = cur_locs.reshape([conf.nviews, 2, conf.n_classes, ntgt])
        cur_locs = np.transpose(cur_locs[conf.view, ...], [2, 1, 0])

        cur_occ = np.array(cur_t['occ'])
        cur_occ = cur_occ.reshape([conf.nviews, conf.n_classes, ntgt])
        cur_occ = cur_occ[conf.view, :]
        cur_occ = np.transpose(cur_occ, [1, 0])

        cur_roi = np.array(cur_t['roi']).reshape([conf.nviews, 2, 4, ntgt])
        cur_roi = np.transpose(cur_roi[conf.view, ...], [2, 1, 0])

        if 'extra_roi' in cur_t.keys():
            extra_roi = np.array(cur_t['extra_roi'],dtype=np.float).reshape([conf.nviews, 2, 4, -1])
            extra_roi = np.transpose(extra_roi[conf.view, ...], [2, 1, 0])
        else:
            extra_roi = None

        if conf.is_multi:
            info = to_py([cur_t['imov'], cur_t['frm'], cur_t['ntgt']])
        else:
            info = to_py([cur_t['imov'], cur_t['frm'], cur_t['itgt']])

        if occ_as_nan:
            cur_locs[cur_occ, :] = np.nan
        cur_occ = cur_occ.astype('float')

        sndx = cur_t['split']
        if type(sndx) == list:
            if len(sndx)<1:
                sndx = 0
            else:
                sndx = sndx[0]
        cur_out = out_fns[sndx]

        if conf.multi_only_ht:
            cur_locs = cur_locs[..., conf.ht_pts, :].copy()
            cur_occ = cur_occ[..., conf.ht_pts].copy()

        if conf.is_multi and conf.multi_crop_ims:
            data_out = create_ma_crops(conf, cur_frame, cur_locs, info, cur_occ, cur_roi, extra_roi)
        else:
            data_out = [{'im': cur_frame, 'locs': cur_locs, 'info': info, 'occ': cur_occ, 'roi': cur_roi,
                         'extra_roi': extra_roi}]
        for curd in data_out:
            cur_out(curd)

        count[sndx] += 1
        splits[sndx].append(info)

        if selndx % 100 == 99 and selndx > 0:
            logging.info('{} number of examples added to the dbs'.format(count))

    logging.info('{} number of examples added to the training dbs'.format(count))
    lbl.close()

    return splits, sel


def db_from_cached_lbl(conf, out_fns, split=True, split_file=None, on_gt=False,sel=None, nsamples=None, use_gt_cache=False):
    # outputs is a list of functions. The first element writes
    # to the training dataset while the second one write to the validation
    # dataset. If split is False, second element is not used and all data is
    # outputted to training dataset
    # the function will be give a list with:
    # 0: img,
    # 1: locations as a numpy array.
    # 2: information list [expid, frame number, trxid]
    # the function returns a list of [expid, frame_number and trxid] showing
    #  how the data was split between the two datasets.
    '''

    :param conf: Note db is created per conf.view
    :param out_fns:
    :param split:
    :param split_file:
    :param on_gt: True when doing gt classify
    :param sel:
    :param nsamples:
    :param use_gt_cache: used when on_gt is True; use separate/dedicated gtcache instead
        of regular
    :return:
    '''
    assert not (on_gt and split), 'Cannot split gt data'

    lbl = h5py.File(conf.labelfile, 'r')
    if not ( ('preProcData_MD_mov' in lbl.keys()) or ('gtcache' in lbl.keys() and 'preProcData_MD_mov' in lbl['gtcache'])):
        if conf.use_ht_trx or conf.use_bbox_trx:
            return db_from_trnpack_ht(conf, out_fns, nsamples=nsamples, split=split)
        else:
            return db_from_trnpack(conf, out_fns, nsamples=nsamples, split=split)

    # npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    if use_gt_cache:
        cachegrp = lbl['gtcache']
    else:
        # local_dirs, _ = multiResData.find_local_dirs(conf, on_gt)
        cachegrp = lbl

    # view = conf.view
    # sel_pts = int(view * npts_per_view) + conf.selpts
    occ_as_nan = conf.get('ignore_occluded', False)

    splits = [[], []]
    count = 0
    val_count = 0

    ims_lbl = True
    # old style stripped label file with cached data.
    # m_ndx = lbl['preProcData_MD_mov'].value[0, :].astype('int')
    # t_ndx = lbl['preProcData_MD_iTgt'].value[0, :].astype('int') - 1
    # f_ndx = lbl['preProcData_MD_frm'].value[0, :].astype('int') - 1
    # occ = lbl['preProcData_MD_tfocc'].value.astype('bool')
    m_ndx = cachegrp['preProcData_MD_mov'].value[0, :].astype('int')
    t_ndx = cachegrp['preProcData_MD_iTgt'].value[0, :].astype('int') - 1
    f_ndx = cachegrp['preProcData_MD_frm'].value[0, :].astype('int') - 1
    occ = cachegrp['preProcData_MD_tfocc'].value.astype('bool')

    mov_split = None
    predefined = None
    if conf.splitType == 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = PoseTools.json_load(split_file)
    elif conf.splitType == 'movie':
        nexps = m_ndx.max() + 1
        mov_split = sample(list(range(nexps)), int(nexps * conf.valratio))
        predefined = None
    elif conf.splitType == 'trx':
        assert conf.has_trx_file, 'Train/Validation was selected to be trx but the project has no trx files'

    if on_gt:
        m_ndx = -m_ndx

    # KB 20190208: if we only need a few images, don't waste time reading in all of them
    if sel is None:
        if nsamples is None:
            sel = np.arange(m_ndx.shape[0])
        else:
            sel = np.random.choice(m_ndx.shape[0], nsamples)
    else:
        sel = np.arange(m_ndx.shape[0])

    for selndx in range(len(sel)):

        ndx = sel[selndx]

        if m_ndx[ndx] < 0:
            continue

        cur_frame = cachegrp[cachegrp['preProcData_I'][conf.view, ndx]].value.copy().T
        if cur_frame.ndim == 2:
            cur_frame = cur_frame[..., np.newaxis]
        cur_locs = to_py(cachegrp['preProcData_P'][:, ndx].copy())
        cur_locs = cur_locs.reshape([2, conf.nviews, conf.n_classes])
        cur_locs = cur_locs[:, conf.view, :].T
        mndx = to_py(m_ndx[ndx])

        cur_occ = occ[:, ndx].copy()
        cur_occ = cur_occ.reshape([conf.nviews, conf.n_classes])
        cur_occ = cur_occ[conf.view, :]

        # For old style code where rotation is done in py look at git history around here to find the code .

        info = [int(mndx), int(f_ndx[ndx]), int(t_ndx[ndx])]

        cur_out = multiResData.get_cur_env(out_fns, split, conf, info, mov_split, trx_split=None, predefined=predefined)
        # when creating from cache, we don't do trx splitting. It should always be predefined

        if occ_as_nan:
            cur_locs[cur_occ] = np.nan
        cur_occ = cur_occ.astype('float')
        data_dict= {'im': cur_frame, 'locs': cur_locs, 'info': info, 'occ': cur_occ}
        if conf.is_multi:
            data_dict['roi'] = []
        cur_out(data_dict)

        if cur_out is out_fns[1] and split:
            val_count += 1
            splits[1].append(info)
        else:
            count += 1
            splits[0].append(info)

        if selndx % 100 == 99 and selndx > 0:
            logging.info('%d,%d number of examples added to the training db and val db' % (count, val_count))

    logging.info('%d,%d number of examples added to the training db and val db' % (count, val_count))
    lbl.close()

    return splits, sel


def create_leap_db(conf, split=False, split_file=None, use_cache=False):
    # function showing how to use db_from_lbl for tfrecords
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)

    train_data = []
    val_data = []

    # collect the images and labels in arrays
    out_fns = [lambda data: train_data.append(data), lambda data: val_data.append(data)]
    if use_cache:
        splits, __ = db_from_cached_lbl(conf, out_fns, split, split_file)
    else:
        splits = db_from_lbl(conf, out_fns, split, split_file)

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

        ims = np.array([i['im'] for i in cur_data])
        locs = np.array([i['locs'] for i in cur_data])
        info = np.array([i['info'] for i in cur_data])
        # hmaps = PoseTools.create_label_images(locs, conf.imsz[:2], 1, conf.label_blur_rad)
        # hmaps = (hmaps + 1) / 2  # brings it back to [0,1]

        if info.size > 0:
            hf = h5py.File(out_file, 'w')
            hf.create_dataset('box', data=np.transpose(ims, (0, 3, 2, 1)))
            # hf.create_dataset('confmaps', data=hmaps)
            hf.create_dataset('joints', data=locs)
            hf.create_dataset('exptID', data=info[:, 0])
            hf.create_dataset('framesIdx', data=info[:, 1])
            hf.create_dataset('trxID', data=info[:, 2])
            hf.close()


def create_deepcut_db(conf, split=False, split_file=None, use_cache=False):
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)

    def deepcut_outfn(data, outdir, count, fis, save_data):
        # pass count as array to pass it by reference.
        if conf.img_dim == 1:
            im = data['im'][:, :, 0]
        else:
            im = data['im']
        img_name = os.path.join(outdir, 'img_{:06d}.png'.format(count[0]))
        imageio.imwrite(img_name, im)
        locs = data['locs']
        bp = conf.n_classes
        for b in range(bp):
            fis[b].write('{}\t{}\t{}\n'.format(count[0], locs[b, 0], locs[b, 1]))
        mod_locs = np.insert(np.array(locs), 0, range(bp), axis=1)
        save_data.append([[img_name], [(3,) + im.shape], [[mod_locs]]])
        count[0] += 1

    bparts = ['part_{}'.format(i) for i in range(conf.n_classes)]
    train_count = [0]
    train_dir = os.path.join(conf.cachedir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_fis = [open(os.path.join(train_dir, b + '.csv'), 'w') for b in bparts]
    train_data = []
    val_count = [0]
    val_dir = os.path.join(conf.cachedir, 'val')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    val_fis = [open(os.path.join(val_dir, b + '.csv'), 'w') for b in bparts]
    val_data = []
    for ndx in range(conf.n_classes):
        train_fis[ndx].write('\tX\tY\n')
        val_fis[ndx].write('\tX\tY\n')

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    def train_out_fn(data):
        deepcut_outfn(data, train_dir, train_count, train_fis, train_data)

    def val_out_fn(data):
        deepcut_outfn(data, val_dir, val_count, val_fis, val_data)

    # collect the images and labels in arrays
    out_fns = [train_out_fn, val_out_fn]
    if use_cache:
        splits, __ = db_from_cached_lbl(conf, out_fns, split, split_file)
    else:
        splits = db_from_lbl(conf, out_fns, split, split_file)
    [f.close() for f in train_fis]
    [f.close() for f in val_fis]
    with open(os.path.join(conf.cachedir, 'train_data.p'), 'wb') as f:
        qq = np.empty([1, len(train_data)], object)
        for i in range(len(train_data)):
            qq[0, i] = train_data[i]
        pickle.dump(qq, f, protocol=2)
    if split:
        with open(os.path.join(conf.cachedir, 'val_data.p'), 'wb') as f:
            qq = np.empty([1, len(val_data)], object)
            for i in range(len(val_data)):
                qq[0, i] = val_data[i]
            pickle.dump(qq, f, protocol=2)

    # save the split data
    try:
        with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
            json.dump(splits, f)
    except IOError:
        logging.warning('SPLIT_WRITE: Could not output the split data information')


def create_cv_split_files(conf, n_splits=3):
    # creates json files for the xv splits
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

    lbls_per_fold = n_labeled_frames / n_splits

    imbalance = True
    for _ in range(10):
        per_fold = np.zeros([n_splits])
        splits = [[] for i in range(n_splits)]

        if conf.splitType == 'movie':
            for ndx in range(len(local_dirs)):
                valid_folds = np.where(per_fold < lbls_per_fold)[0]
                cur_fold = np.random.choice(valid_folds)
                splits[cur_fold].extend(mov_info[ndx])
                per_fold[cur_fold] += len(mov_info[ndx])

        elif conf.splitType == 'trx':
            for tndx in range(len(trx_info)):
                valid_folds = np.where(per_fold < lbls_per_fold)[0]
                cur_fold = np.random.choice(valid_folds)
                splits[cur_fold].extend(trx_info[tndx])
                per_fold[cur_fold] += len(trx_info[tndx])

        elif conf.splitType == 'frame':
            for ndx in range(len(local_dirs)):
                for mndx in range(len(mov_info[ndx])):
                    valid_folds = np.where(per_fold < lbls_per_fold)[0]
                    cur_fold = np.random.choice(valid_folds)
                    splits[cur_fold].extend(mov_info[ndx][mndx:mndx + 1])
                    per_fold[cur_fold] += 1
        else:
            raise ValueError('splitType has to be either movie trx or frame')

        imbalance = (per_fold.max() - per_fold.min()) > float(lbls_per_fold) / 3
        if not imbalance:
            break

    if imbalance:
        print('Couldnt find a valid spilt for split type:{} even after 10 retries.'.format(conf.splitType))
        print('Try changing the split type')
        return None

    all_train = []
    split_files = []
    for ndx in range(n_splits):
        cur_train = []
        for idx, cur_split in enumerate(splits):
            if idx is not ndx:
                cur_train.extend(cur_split)
        all_train.append(cur_train)
        cur_split_file = os.path.join(conf.cachedir, 'cv_split_fold_{}.json'.format(ndx))
        split_files.append(cur_split_file)
        with open(cur_split_file, 'w') as f:
            json.dump([cur_train, splits[ndx]], f)

    return all_train, splits, split_files


def create_batch_ims(to_do_list, conf, cap, flipud, trx, crop_loc,use_bsize=True):
    if use_bsize:
        bsize = conf.batch_size
    else:
        bsize = len(to_do_list)
    all_f = np.zeros((bsize,) + tuple(conf.imsz) + (conf.img_dim,))
    # KB 20200504: sometimes crop_loc might be specified as nans when
    # we want no cropping to happen for reasons. 
    if crop_loc is not None and np.any(np.isnan(np.array(crop_loc))):
        crop_loc = None

    for cur_t in range(len(to_do_list)):
        cur_entry = to_do_list[cur_t]
        trx_ndx = cur_entry[1]
        cur_trx = trx[trx_ndx]
        cur_f = cur_entry[0]

        frame_in, cur_loc = multiResData.get_patch(
            cap, cur_f, conf, np.zeros([conf.n_classes, 2]),
            cur_trx=cur_trx, flipud=flipud, crop_loc=crop_loc)
        all_f[cur_t, ...] = frame_in
    return all_f


def get_trx_info(trx_file, conf, n_frames):
    ''' all returned values are 0-indexed'''
    if conf.has_trx_file:
        trx = sio.loadmat(trx_file)['trx'][0]
        n_trx = len(trx)
        end_frames = np.array([x['endframe'][0, 0] for x in trx])
        first_frames = np.array(
            [x['firstframe'][0, 0] for x in trx]) - 1  # for converting from 1 indexing to 0 indexing
    elif conf.use_ht_trx or conf.use_bbox_trx:
        # convert trk file to trx file format.
        T = h5py.File(trx_file, 'r')
        if T['pTrk'].ndim < 2:
            in_n_trx = 0
        else:
            in_n_trx = T['pTrk'].shape[0]
        trx = []
        end_frames = []
        first_frames = []
        n_trx = 0
        for tndx in range(in_n_trx):
            sframe = T['startframes'][tndx, 0] - 1
            eframe = T['endframes'][tndx, 0]
            if eframe <= sframe:
                continue
            cur_pts = T[T['pTrk'][tndx, 0]][()] + 1
            theta = np.arctan2(cur_pts[..., 1, 0] - cur_pts[..., 1, 1], cur_pts[..., 0, 0] - cur_pts[..., 0, 1])
            ctr = cur_pts.mean(-1)
            end_frames.append(eframe)
            first_frames.append(sframe)
            curtrx = {'x': ctr[None, ..., 0],
                      'y': ctr[None, ..., 1],
                      'firstframe': np.array(sframe + 1).reshape([1, 1]),
                      'endframe': np.array(eframe).reshape([1, 1]),
                      'theta': theta[None, ...]
                      }
            # +1 for firstframe because curtrx is assumed to be in matlab format
            if 'pTrkConf' in T:
                conf = T[T['pTrkConf'][tndx,0]][()]
                curtrx['conf'] = conf[None,...]
            trx.append(curtrx)
            n_trx += 1
        end_frames = np.array(end_frames)
        first_frames = np.array(first_frames)
    else:
        if conf.is_multi:
            trx = [None, ]
            n_trx = conf.max_n_animals
        else:
            trx = [None, ]
            n_trx = 1
        end_frames = np.array([n_frames])
        first_frames = np.array([0])
    return {'trx':trx, 'first_frames':first_frames, 'end_frames':end_frames, 'n_trx':n_trx}


def get_trx_ids(trx_ids_in, n_trx, has_trx_file):
    if has_trx_file:
        if len(trx_ids_in) == 0:
            trx_ids = np.arange(n_trx)
        else:
            trx_ids = np.array(trx_ids_in)
    else:
        trx_ids = np.array([0])
    return trx_ids


def get_augmented_images(conf, out_file, distort=True, on_gt=False, nsamples=None):
    data_in = []
    out_fns = [lambda data: data_in.append(data),
               lambda data: None]

    logging.info('use cache')
    splits, sel = db_from_cached_lbl(conf, out_fns, False, None, on_gt, nsamples=nsamples)
    ims = np.array([d['im'] for d in data_in])
    locs = np.array([d['locs'] for d in data_in])
    logging.info('sel = ' + str(sel))

    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, distort, conf.rescale)

    hdf5storage.savemat(out_file, {'ims': ims, 'locs': locs + 1., 'idx': sel + 1})
    logging.info('Augmented data saved to %s' % out_file)


def convert_to_orig_list(conf, preds, locs, in_list, view, on_gt=False):
    '''convert predicted locs back to original image co-ordinates.
    INCOMPLETE/UNUSED
    '''
    lbl = h5py.File(conf.labelfile, 'r')
    if on_gt:
        local_dirs, _ = multiResData.find_gt_dirs(conf)
    else:
        local_dirs, _ = multiResData.find_local_dirs(conf)

    if conf.has_trx_file:
        trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)
    else:
        trx_files = [None, ] * len(local_dirs)

    for ndx, dir_name in enumerate(local_dirs):
        cur_list = [[l[1], l[2]] for l in in_list if l[0] == (ndx)]
        cur_idx = [i for i, l in enumerate(in_list) if l[0] == (ndx)]
        crop_loc = PoseTools.get_crop_loc(lbl, ndx, view, on_gt)


def classify_list(conf, pred_fn, cap, to_do_list, trx, crop_loc, n_done=0, part_file=None):
    '''

    :param conf:
    :param pred_fn:
    :param cap: Movie object/instance
    :param to_do_list: list of [frm,tgt] sublists (both 0-based) for given movie
    :param trx: trx structure eg first output arg of get_trx_info
    :param crop_loc: crop information
    :param n_done: Number of frames that have been tracked from the list before this call to classify_list. Default = 0.
    :param part_file: File to output number of frames tracked to. If None, number of frames tracked not output. Default = None.
    :return: dict of results. locs are in original coords (independent of crop/roi), but 0based
    '''

    flipud = conf.flipud
    bsize = conf.batch_size
    n_list = len(to_do_list)
    n_batches = int(math.ceil(float(n_list) / bsize))

    ret_dict = {}

    # if part_file is specified, output count of number of frames tracked 
    # (n_done) to part_file every N_TRACKED_WRITE_INTERVAL_SEC seconds
    do_write_n_done = part_file is not None
    if do_write_n_done:
        start_time = time.time()

    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        nrows_pred = min(n_list - cur_start, bsize)
        all_f = create_batch_ims(to_do_list[cur_start:(cur_start + nrows_pred)],
                                 conf, cap, flipud, trx, crop_loc)
        assert all_f.shape[0] == bsize  # dim0 has size bsize but only nrows_pred rows are filled
        ret_dict_b = pred_fn(all_f)

        # py3 and py2 compatible
        for k in ret_dict_b.keys():
            retval = ret_dict_b[k]
            if k not in ret_dict.keys() and (retval.ndim >= 1):
                # again only nrows_pred rows are filled
                assert retval.shape[0] == bsize, \
                    "Unexpected output shape {} for key {}".format(retval.shape, k)
                sz = retval.shape[1:]
                ret_dict[k] = np.zeros((n_list,) + sz)
                ret_dict[k][:] = np.nan

        for cur_t in range(nrows_pred):
            cur_entry = to_do_list[cur_t + cur_start]
            cur_f = cur_entry[0]
            trx_ndx = cur_entry[1]
            cur_trx = trx[trx_ndx]
            for k in ret_dict_b.keys():
                retval = ret_dict_b[k]
                # if retval.ndim == 4:  # hmaps
                #    pass
                if retval.ndim >= 1:
                    cur_orig = retval[cur_t, ...]
                    if k.startswith('locs'):  # transform locs
                        if retval.ndim == 3:
                            cur_orig = convert_to_orig(cur_orig, conf, cur_f, cur_trx, crop_loc)
                        else:
                            # ma
                            # TODO: ma + crops
                            pass
                    ret_dict[k][cur_start + cur_t, ...] = cur_orig
                else:
                    logging.info("Ignoring return value '{}' with shape {}".format(k, retval.shape))
                    # assert False, "Unexpected number of dims in return val"
        # update count of frames tracked
        n_done += nrows_pred
        if do_write_n_done:
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            if elapsed_time >= N_TRACKED_WRITE_INTERVAL_SEC:
                # output n frames tracked to file
                write_n_tracked_part_file(n_done, part_file)
                start_time = curr_time

    return ret_dict


def write_n_tracked_part_file(n_done, part_file):
    '''
    write_n_tracked_part_file(n_done,part_file)
    Output to file part_file the number n_done a string. 
    This is used for communicating progress of tracking to other processes.
    '''
    with open(part_file, 'w') as fh:
        fh.write("{}".format(n_done))


def get_pred_fn(model_type, conf, model_file=None, name='deepnet', distort=False, **kwargs):
    ''' Returns prediction functions and close functions for different network types

    '''
    if model_type == 'dpk':
        pred_fn, close_fn, model_file = apt_dpk.get_pred_fn(conf, model_file, **kwargs)
    # elif model_type == 'openpose':
    #     pred_fn, close_fn, model_file = op.get_pred_fn(conf, model_file,name=name,**kwargs)
    elif model_type == 'sb':
        pred_fn, close_fn, model_file = sb.get_pred_fn(conf, model_file, name=name, **kwargs)
    elif model_type == 'unet':
        pred_fn, close_fn, model_file = get_unet_pred_fn(conf, model_file, name=name, **kwargs)
    elif model_type == 'mdn':
        pred_fn, close_fn, model_file = get_mdn_pred_fn(conf, model_file, name=name, distort=distort, **kwargs)
    elif model_type == 'leap':
        pred_fn, close_fn, model_file = leap.training.get_pred_fn(conf, model_file, name=name, **kwargs)
    elif model_type == 'deeplabcut':
        cfg_dict = create_dlc_cfg_dict(conf, name)
        pred_fn, close_fn, model_file = deeplabcut.pose_estimation_tensorflow.get_pred_fn(cfg_dict, model_file)
    else:
        try:
            module_name = 'Pose_{}'.format(model_type)
            pose_module = __import__(module_name)
            tf.reset_default_graph()
            self = getattr(pose_module, module_name)(conf, name=name)
            pred_fn, close_fn, model_file = self.get_pred_fn(model_file)
        except ImportError:
            raise ImportError('Undefined type of network')

    return pred_fn, close_fn, model_file


def classify_list_all(model_type, conf, in_list, on_gt, model_file,
                      movie_files=None, trx_files=None, crop_locs=None,
                      part_file=None,  # If specified, save intermediate "part" files
                      ):
    '''
    Classifies a list of examples.

    in_list should be of list of type [mov_file, frame_num, trx_ndx]
    everything is 0-indexed

    Movie and trx indices in in_list are dereferenced as follows:
    * In the usual case, movie_files is None and movieFilesAll/trxFilesAll in the
    conf.labelfile are used. If on_gt is True, movieFilesAllGT/etc are used. Crop
    locations are also read from the conf.labelfile (if present).
    * In the externally-specified case, movie_files, trx_files (if appropriate),
    and crop_locs (if appropriate) must be provided.
    '''

    # Possible refactor: factor into i) marshall movs/trxs/crops and ii) track the
    #  'external' list

    is_external_movies = movie_files is not None
    if is_external_movies:
        local_dirs = movie_files

        assert (trx_files is not None and len(trx_files) > 0) == conf.has_trx_file, \
            "Unexpected trx_files specification (length {}), conf.has_trx_file={}.".format(
                len(trx_files), conf.has_trx_file)

        is_external_crop = (crop_locs is not None) and (len(crop_locs) > 0)
        if is_external_crop:
            assert len(crop_locs) == len(local_dirs), \
                "Number of crop_locs ({}) does not match number of movies ({})".format(len(crop_locs), len(local_dirs))
    elif on_gt:
        local_dirs, _ = multiResData.find_gt_dirs(conf)
        # crops fetched from lbl below
    else:
        local_dirs, _ = multiResData.find_local_dirs(conf)
        # crops fetched from lbl below

    lbl = h5py.File(conf.labelfile, 'r')
    view = conf.view

    if conf.has_trx_file:
        if is_external_movies:
            pass  # trx_files provided
        else:
            trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)
    else:
        trx_files = [None, ] * len(local_dirs)

    assert len(trx_files) == len(local_dirs), \
        "Number of trx_files ({}) does not match number of movies ({})".format(len(trx_files), len(local_dirs))

    pred_fn, close_fn, model_file = get_pred_fn(model_type, conf, model_file)

    # pred_locs = np.zeros([len(in_list), conf.n_classes, 2])
    # pred_locs[:] = np.nan
    # pred_conf = np.zeros([len(in_list), conf.n_classes])
    # pred_conf[:] = np.nan
    # pred_crop_locs = np.zeros([len(in_list), 4])
    # pred_crop_locs[:] = np.nan

    nlist = len(in_list)
    ret_dict_all = {}
    ret_dict_all['crop_locs'] = np.zeros([nlist, 4])
    ret_dict_all['crop_locs'][:] = np.nan

    logging.info('Tracking {} rows...'.format(nlist))
    n_done = 0
    start_time = time.time()
    for ndx, dir_name in enumerate(local_dirs):

        cur_list = [[l[1], l[2]] for l in in_list if l[0] == ndx]
        cur_idx = [i for i, l in enumerate(in_list) if l[0] == ndx]
        if is_external_movies:
            if is_external_crop:
                crop_loc = crop_locs[ndx]
            else:
                crop_loc = None
        else:
            # This returns None if proj/lbl doesnt have crops
            crop_loc = PoseTools.get_crop_loc(lbl, ndx, view, on_gt)

        try:
            cap = movies.Movie(dir_name)
        except ValueError:
            logging.exception('MOVIE_READ: ' + local_dirs[ndx] + ' is missing')
            exit(1)

        trx = get_trx_info(trx_files[ndx], conf, 0)['trx']
        ret_dict = classify_list(conf, pred_fn, cap, cur_list, trx, crop_loc, n_done=n_done, part_file=part_file)

        n_cur_list = len(cur_list)  # len of cur_idx; num of rows being processed for curr mov
        for k in ret_dict.keys():
            retval = ret_dict[k]
            if k not in ret_dict_all.keys():
                szval = retval.shape
                assert szval[0] == n_cur_list
                ret_dict_all[k] = np.zeros((nlist,) + szval[1:])
                ret_dict_all[k][:] = np.nan

            ret_dict_all[k][cur_idx, ...] = retval

        # pred_locs[cur_idx, ...] = ret_dict['pred_locs']
        # pred_conf[cur_idx, ...] = ret_dict['pred_conf']
        if crop_loc is not None:
            ret_dict_all['crop_locs'][cur_idx, ...] = crop_loc

        cap.close()  # close the movie handles

        n_done = len([1 for i in in_list if i[0] <= ndx])
        logging.info('Done prediction on {} out of {} GT labeled frames'.format(n_done, len(in_list)))
        if part_file is not None:
            write_n_tracked_part_file(n_done, part_file)

    if 'conf' not in ret_dict_all:
        ret_dict_all['conf'] = vars(conf)

    logging.info('Done prediction on all frames')
    lbl.close()
    close_fn()
    return ret_dict_all


def classify_db(conf, read_fn, pred_fn, n, return_ims=False,
                return_hm=False, hm_dec=100, hm_floor=0.0, hm_nclustermax=1):
    '''Classifies n examples generated by read_fn'''
    bsize = conf.batch_size
    all_f = np.zeros((bsize,) + tuple(conf.imsz) + (conf.img_dim,))
    pred_locs = np.zeros([n, conf.n_classes, 2])
    mdn_locs = np.zeros([n, conf.n_classes, 2])
    unet_locs = np.zeros([n, conf.n_classes, 2])
    mdn_conf = np.zeros([n, conf.n_classes])
    unet_conf = np.zeros([n, conf.n_classes])
    n_batches = int(math.ceil(float(n) / bsize))
    labeled_locs = np.zeros([n, conf.n_classes, 2])

    info = []
    if return_ims:
        all_ims = np.zeros([n, conf.imsz[0], conf.imsz[1], conf.img_dim])
    if return_hm:
        nhm = n // hm_dec + n % hm_dec
        all_hmaps = np.zeros([nhm, conf.imsz[0], conf.imsz[1], conf.n_classes])
        hmap_locs = np.zeros([nhm, conf.n_classes, 2])
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            all_f[ndx, ...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])

        # note all_f[ndx+1:, ...] for the last batch will be cruft

        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']

        for ndx in range(ppe):
            pred_locs[cur_start + ndx, ...] = base_locs[ndx, ...]
            if 'locs_mdn' in ret_dict.keys():
                mdn_locs[cur_start + ndx, ...] = ret_dict['locs_mdn'][ndx, ...]
                unet_locs[cur_start + ndx, ...] = ret_dict['locs_unet'][ndx, ...]
                mdn_conf[cur_start + ndx, ...] = ret_dict['conf'][ndx, ...]
                unet_conf[cur_start + ndx, ...] = ret_dict['conf_unet'][ndx, ...]
            if return_ims:
                all_ims[cur_start + ndx, ...] = all_f[ndx, ...]
            if 'hmaps' in ret_dict and return_hm and \
                    (cur_start + ndx) % hm_dec == 0:
                hmapidx = (cur_start + ndx) // hm_dec
                hmthis = ret_dict['hmaps'][ndx, ...]
                hmthis = hmthis + 1.0
                # all_hmaps[hmapidx, ...] = hmthis
                hmmu = np.zeros((conf.n_classes, 2))
                for ipt in range(conf.n_classes):
                    _, mutmp, _, _ = heatmap.compactify_hmap(hmthis[:, :, ipt],
                                                             floor=hm_floor,
                                                             nclustermax=hm_nclustermax)
                    hmmu[ipt, :] = mutmp[::-1].flatten() - 1.0
                hmap_locs[cur_start + ndx, ...] = hmmu

    if return_ims:
        return pred_locs, labeled_locs, info, all_ims
    else:
        extrastuff = [mdn_locs, unet_locs, mdn_conf, unet_conf]
        if return_hm:
            # extrastuff.append(all_hmaps)
            extrastuff.append(hmap_locs)
        return pred_locs, labeled_locs, info, extrastuff


def classify_db_multi(conf, read_fn, pred_fn, n, return_ims=False,
                      return_hm=False, hm_dec=100, hm_floor=0.0, hm_nclustermax=1):
    '''Classifies n examples generated by read_fn'''
    bsize = conf.batch_size
    max_n = conf.max_n_animals
    all_f = np.zeros((bsize,) + tuple(conf.imsz) + (conf.img_dim,))
    pred_locs = np.zeros([n, max_n, conf.n_classes, 2])
    mdn_locs = np.zeros([n, max_n, conf.n_classes, 2])
    unet_locs = np.zeros([n, max_n, conf.n_classes, 2])
    mdn_conf = np.zeros([n, max_n, conf.n_classes])
    unet_conf = np.zeros([n, max_n, conf.n_classes])
    joint_locs = np.zeros([n, max_n, conf.n_classes, 2])
    n_batches = int(math.ceil(float(n) / bsize))
    labeled_locs = np.zeros([n, max_n, conf.n_classes, 2])

    info = []
    if return_ims:
        all_ims = np.zeros([n, conf.imsz[0], conf.imsz[1], conf.img_dim])
    if return_hm:
        nhm = n // hm_dec + n % hm_dec
        all_hmaps = np.zeros([nhm, conf.imsz[0], conf.imsz[1], conf.n_classes])
        hmap_locs = np.zeros([nhm, conf.n_classes, 2])
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            if type(next_db) == dict:
                all_f[ndx, ...] = next_db['images']
                labeled_locs[cur_start + ndx, ...] = next_db['locs']
                info.append(next_db['info'])
            else:
                all_f[ndx, ...] = next_db[0]
                labeled_locs[cur_start + ndx, ...] = next_db[1]
                info.append(next_db[2])

        # note all_f[ndx+1:, ...] for the last batch will be cruft

        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']

        for ndx in range(ppe):
            pred_locs[cur_start + ndx, ...] = base_locs[ndx, ...]
            if 'locs_mdn' in ret_dict.keys():
                mdn_locs[cur_start + ndx, ...] = ret_dict['locs_mdn'][ndx, ...]
                unet_locs[cur_start + ndx, ...] = ret_dict['locs_unet'][ndx, ...]
                mdn_conf[cur_start + ndx, ...] = ret_dict['conf'][ndx, ...]
                unet_conf[cur_start + ndx, ...] = ret_dict['conf_unet'][ndx, ...]
            if return_ims:
                all_ims[cur_start + ndx, ...] = all_f[ndx, ...]
            if 'hmaps' in ret_dict and return_hm and \
                    (cur_start + ndx) % hm_dec == 0:
                hmapidx = (cur_start + ndx) // hm_dec
                hmthis = ret_dict['hmaps'][ndx, ...]
                hmthis = hmthis + 1.0
                # all_hmaps[hmapidx, ...] = hmthis
                hmmu = np.zeros((conf.n_classes, 2))
                for ipt in range(conf.n_classes):
                    _, mutmp, _, _ = heatmap.compactify_hmap(hmthis[:, :, ipt],
                                                             floor=hm_floor,
                                                             nclustermax=hm_nclustermax)
                    hmmu[ipt, :] = mutmp[::-1].flatten() - 1.0
                hmap_locs[cur_start + ndx, ...] = hmmu
            if 'locs_joint' in ret_dict.keys():
                joint_locs[cur_start + ndx, ...] = ret_dict['locs_joint'][ndx, ...]

    if return_ims:
        return pred_locs, labeled_locs, info, all_ims
    else:
        extrastuff = [mdn_locs, unet_locs, mdn_conf, unet_conf, joint_locs]
        if return_hm:
            # extrastuff.append(all_hmaps)
            extrastuff.append(hmap_locs)
        return pred_locs, labeled_locs, info, extrastuff


def classify_db_multi(conf, read_fn, pred_fn, n, return_ims=False,
                      return_hm=False, hm_dec=100, hm_floor=0.0, hm_nclustermax=1):
    assert False, 'Use classify_db2'
    '''Classifies n examples generated by read_fn'''
    bsize = conf.batch_size
    max_n = conf.max_n_animals
    all_f = np.zeros((bsize,) + tuple(conf.imsz) + (conf.img_dim,))
    pred_locs = np.zeros([n, max_n, conf.n_classes, 2])
    mdn_locs = np.zeros([n, max_n, conf.n_classes, 2])
    unet_locs = np.zeros([n, max_n, conf.n_classes, 2])
    mdn_conf = np.zeros([n, max_n, conf.n_classes])
    unet_conf = np.zeros([n, max_n, conf.n_classes])
    joint_locs = np.zeros([n, max_n, conf.n_classes, 2])
    n_batches = int(math.ceil(float(n) / bsize))
    labeled_locs = np.zeros([n, max_n, conf.n_classes, 2])

    info = []
    if return_ims:
        all_ims = np.zeros([n, conf.imsz[0], conf.imsz[1], conf.img_dim])
    if return_hm:
        nhm = n // hm_dec + n % hm_dec
        all_hmaps = np.zeros([nhm, conf.imsz[0], conf.imsz[1], conf.n_classes])
        hmap_locs = np.zeros([nhm, conf.n_classes, 2])
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            all_f[ndx, ...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])

        # note all_f[ndx+1:, ...] for the last batch will be cruft

        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']

        for ndx in range(ppe):
            pred_locs[cur_start + ndx, ...] = base_locs[ndx, ...]
            if 'locs_mdn' in ret_dict.keys():
                mdn_locs[cur_start + ndx, ...] = ret_dict['locs_mdn'][ndx, ...]
                unet_locs[cur_start + ndx, ...] = ret_dict['locs_unet'][ndx, ...]
                mdn_conf[cur_start + ndx, ...] = ret_dict['conf'][ndx, ...]
                unet_conf[cur_start + ndx, ...] = ret_dict['conf_unet'][ndx, ...]
            if return_ims:
                all_ims[cur_start + ndx, ...] = all_f[ndx, ...]
            if 'hmaps' in ret_dict and return_hm and \
                    (cur_start + ndx) % hm_dec == 0:
                hmapidx = (cur_start + ndx) // hm_dec
                hmthis = ret_dict['hmaps'][ndx, ...]
                hmthis = hmthis + 1.0
                # all_hmaps[hmapidx, ...] = hmthis
                hmmu = np.zeros((conf.n_classes, 2))
                for ipt in range(conf.n_classes):
                    _, mutmp, _, _ = heatmap.compactify_hmap(hmthis[:, :, ipt],
                                                             floor=hm_floor,
                                                             nclustermax=hm_nclustermax)
                    hmmu[ipt, :] = mutmp[::-1].flatten() - 1.0
                hmap_locs[cur_start + ndx, ...] = hmmu
            if 'locs_joint' in ret_dict.keys():
                joint_locs[cur_start + ndx, ...] = ret_dict['locs_joint'][ndx, ...]

    if return_ims:
        return pred_locs, labeled_locs, info, all_ims
    else:
        extrastuff = [mdn_locs, unet_locs, mdn_conf, unet_conf, joint_locs]
        if return_hm:
            # extrastuff.append(all_hmaps)
            extrastuff.append(hmap_locs)
        return pred_locs, labeled_locs, info, extrastuff


def classify_db2(conf, read_fn, pred_fn, n, return_ims=False,
                 timer_read=None, timer_pred=None, ignore_hmaps=False,
                 **kwargs):  # fed to pred_fn
    '''Trying to simplify/generalize classify_db'''

    if timer_read is None:
        timer_read = contextlib.suppress()
    if timer_pred is None:
        timer_pred = contextlib.suppress()

    # logging.info("Ignoring kwargs: {}".format(kwargs.keys()))

    bsize = conf.batch_size
    n_batches = int(math.ceil(float(n) / bsize))

    all_f = np.zeros((bsize,) + tuple(conf.imsz) + (conf.img_dim,))
    if conf.is_multi:
        labeled_locs = np.zeros([n, conf.max_n_animals,conf.n_classes, 2])
    else:
        labeled_locs = np.zeros([n, conf.n_classes, 2])
    info = []
    # pred_locs = np.zeros([n, conf.n_classes, 2])
    # mdn_locs = np.zeros([n, conf.n_classes, 2])
    # unet_locs = np.zeros([n, conf.n_classes, 2])
    # mdn_conf = np.zeros([n, conf.n_classes])
    # unet_conf = np.zeros([n, conf.n_classes])

    ret_dict_all = {}

    for cur_b in tqdm(range(n_batches)):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            with timer_read:
                next_db = read_fn()
            all_f[ndx, ...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])

        # note all_f[ppe+1:, ...] for the last batch will be cruft

        with timer_pred:
            ret_dict = pred_fn(all_f, **kwargs)

        fields = ret_dict.keys()
        if cur_b == 0:
            for k in fields:
                if ignore_hmaps and 'hmap' in k:
                    continue
                val = ret_dict[k]
                valshape = val.shape
                if valshape[0] == bsize:
                    bigvalshape = (n,) + valshape[1:]
                    bigval = np.zeros(bigvalshape)
                    bigval[:] = np.nan
                    ret_dict_all[k] = bigval
                else:
                    logging.warning(
                        "Key {}, value has shape {}. Will not be included in return dict.".format(k, valshape))

            fields_record = list(ret_dict_all.keys())
            logging.warning("Recording these pred fields: {}".format(fields_record))

            if return_ims:
                ret_dict_all['ims_raw'] = np.zeros([n, conf.imsz[0], conf.imsz[1], conf.img_dim])
        else:
            # ret_dict_all, fields_record configured
            pass

        # base_locs = ret_dict['locs']

        for ndx in range(ppe):
            for k in fields_record:
                ret_dict_all[k][cur_start + ndx, ...] = ret_dict[k][ndx, ...]
            if return_ims:
                ret_dict_all['ims_raw'][cur_start + ndx, ...] = all_f[ndx, ...]

    return ret_dict_all, labeled_locs, info


def classify_db_all(model_type, conf, db_file, model_file=None,classify_fcn=None, name='deepnet',fullret=False, img_dir='val',conf2=None,model_type2=None,name2='deepnet', model_file2=None, **kwargs):
    '''
        Classifies examples in DB.

    :param model_type:
    :param conf:
    :param db_file:
    :param model_file:
    :param classify_fcn:
    :param name:
    :param fullret: if True, return raw output of classify_fcn as single arg
    :return:
    '''

    if model_type2 is not None:
        return classify_db_2stage([model_type,model_type2],[conf,conf2],db_file,[model_file,model_file2],name=[name,name2])

    pred_fn, close_fn, model_file = get_pred_fn(model_type, conf, model_file, name=name)

    if classify_fcn is None:
        classify_fcn = classify_db2


    if model_type == 'leap':
        leap_gen, n = leap.training.get_read_fn(conf, db_file)
        ret = classify_fcn(conf, leap_gen, pred_fn, n, **kwargs)
        pred_locs, label_locs, info = ret[:3]
        close_fn()
    elif model_type == 'deeplabcut':
        cfg_dict = create_dlc_cfg_dict(conf)
        [p, d] = os.path.split(db_file)
        cfg_dict['project_path'] = p
        cfg_dict['dataset'] = d
        read_fn, n = deeplabcut.pose_estimation_tensorflow.get_read_fn(cfg_dict)
        ret = classify_fcn(conf, read_fn, pred_fn, n, **kwargs)
        pred_locs, label_locs, info = ret[:3]
        close_fn()
    else:
        if conf.db_format == 'coco':
            coco_reader = multiResData.coco_loader(conf, db_file, False, img_dir=img_dir)
            read_fn = iter(coco_reader).__next__
            db_len = len(coco_reader)
            conf.img_dim = 3
        else:
            is_multi = conf.is_multi
            tf_iterator = multiResData.tf_reader(conf, db_file, False, is_multi=is_multi)
            tf_iterator.batch_size = 1
            read_fn = tf_iterator.next
            db_len = tf_iterator.N
        ret = classify_fcn(conf, read_fn, pred_fn, db_len)
        pred_locs, label_locs, info = ret[:3]
        close_fn()

        # raise ValueError('Undefined model type')

    if fullret:
        return ret
    else:
        return pred_locs, label_locs, info, model_file


def classify_db_2stage(model_type, conf, db_file, model_file = [None,None], name=['deepnet','deepnet'],
                    return_ims=False, img_dir='val',
                    **kwargs
                    ):
    '''
        Classifies examples in DB.

    :param model_type:
    :param conf:
    :param db_file:
    :param model_file:
    :param classify_fcn:
    :param name:
    :return:
    '''

    import copy
    conf1 = copy.deepcopy(conf[0])
    conf1.n_classes = 2
    pred_fn_top, close_fn_top, model_file_top = get_pred_fn(model_type[0], conf1, model_file[0], name=name[0])

    if conf[0].db_format == 'coco':
        coco_reader = multiResData.coco_loader(conf[0], db_file, False, img_dir=img_dir)
        read_fn = iter(coco_reader).__next__
        db_len = len(coco_reader)
    else:
        tf_iterator = multiResData.tf_reader(conf[0], db_file, False, is_multi=True)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        db_len = tf_iterator.N


    bsize = conf[0].batch_size
    max_n = conf[0].max_n_animals
    all_f = np.zeros((bsize,) + tuple(conf[0].imsz) + (conf[0].img_dim,))
    n_batches = int(math.ceil(float(db_len) / bsize))
    ret_dict_all = {}
    labeled_locs = np.zeros([db_len, max_n, conf[0].n_classes, 2])
    info = []
    n = db_len
    single_data = []

    # Predict the top level first
    for cur_b in tqdm(range(n_batches)):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            all_f[ndx, ...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])

        # note all_f[ndx+1:, ...] for the last batch will be cruft

        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn_top(all_f)

        fields = ret_dict.keys()
        if cur_b == 0:
            for k in fields:
                if 'hmap' in k:
                    continue
                val = ret_dict[k]
                valshape = val.shape
                if valshape[0] == bsize:
                    bigvalshape = (n,) + valshape[1:]
                    bigval = np.zeros(bigvalshape)
                    bigval[:] = np.nan
                    ret_dict_all[k+'_top'] = bigval
                else:
                    logging.warning(
                        "Key {}, value has shape {}. Will not be included in return dict.".format(k, valshape))

            fields_record = list(ret_dict_all.keys())
            fields_record = [k.replace('_top','') for k in fields_record]
            logging.warning("Recording these pred fields: {}".format(fields_record))

        base_locs = ret_dict['locs']
        for ndx in range(ppe):
            for k in fields_record:
                ret_dict_all[k+'_top'][cur_start + ndx, ...] = ret_dict[k][ndx, ...]

            # Create the images for single animal predictions
            for curn in range(max_n):
                if np.all(np.isnan(base_locs[ndx,curn])):
                    continue
                ht_locs = base_locs[ndx,curn]
                ctr = ht_locs.mean(axis=0)
                if conf[1].use_ht_trx:
                    theta = np.arctan2(ht_locs[0, 1] - ht_locs[1, 1], ht_locs[0, 0] - ht_locs[1, 0])
                else:
                    theta = 0
                curp, curl = multiResData.crop_patch_trx(conf[1],all_f[ndx],ctr[0],ctr[1],theta,np.zeros([conf[1].n_classes,2]))
                single_data.append([curp,ctr,theta,cur_start+ndx,curn])

    close_fn_top()
    import shutil
    nv_dir = os.path.expanduser('~/.nv')
    if os.path.exists(nv_dir): shutil.rmtree(nv_dir)


    pred_fn_single, close_fn_single, model_file_single = get_pred_fn(model_type[1], conf[1], model_file[1], name=name[1])

    # Predict the single level
    bsize = conf[1].batch_size
    all_fs = np.zeros((conf[1].batch_size,) + tuple(conf[1].imsz) + (conf[1].img_dim,))
    n_batches = int(math.ceil(float(len(single_data)) / bsize))

    for cur_b in tqdm(range(n_batches)):
        cur_start = cur_b * bsize
        ppe = min(len(single_data) - cur_start, bsize)
        for ndx in range(ppe):
            all_fs[ndx, ...] = single_data[cur_start+ndx][0]

        ret_dict = pred_fn_single(all_fs)

        fields = ret_dict.keys()
        if cur_b == 0:
            for k in fields:
                if 'hmap' in k:
                    continue
                val = ret_dict[k]
                valshape = val.shape
                if valshape[0] == bsize:
                    bigvalshape = (n,max_n) + valshape[1:]
                    bigval = np.zeros(bigvalshape)
                    bigval[:] = np.nan
                    ret_dict_all[k] = bigval
                else:
                    logging.warning(
                        "Key {}, value has shape {}. Will not be included in return dict.".format(k, valshape))

            fields_record = list(ret_dict_all.keys())
            fields_record = [k for k in fields_record if not k.endswith('_top')]
            logging.warning("Recording these pred fields for 2nd stage: {}".format(fields_record))

        for ndx in range(ppe):
            cur_in = single_data[cur_start+ndx]
            ctr, theta, im_idx, animal_idx = cur_in[1:5]
            for k in fields_record:
                if 'locs' in k:
                    ret_dict_all[k][im_idx,animal_idx, ...] = to_orig(conf[1],ret_dict[k][ndx, ...],ctr[0],ctr[1],theta)
                else:
                    ret_dict_all[k][im_idx,animal_idx, ...] = ret_dict[k][ndx, ...]


    close_fn_single()

    return ret_dict_all, labeled_locs, info, [model_file_top,model_file_single]


def check_train_db(model_type, conf, out_file):
    ''' Reads db and saves the images and locs to out_file to verify the db'''
    if model_type == 'openpose':
        db_file = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
        print('Checking db from {}'.format(db_file))
        tf_iterator = multiResData.tf_reader(conf, db_file, False)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        n = tf_iterator.N
    elif model_type == 'unet':
        db_file = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
        print('Checking db from {}'.format(db_file))
        tf_iterator = multiResData.tf_reader(conf, db_file, False)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        n = tf_iterator.N
    elif model_type == 'leap':
        db_file = os.path.join(conf.cachedir, 'leap_train.h5')
        print('Checking db from {}'.format(db_file))
        read_fn, n = leap.training.get_read_fn(conf, db_file)
    elif model_type == 'deeplabcut':
        db_file = os.path.join(conf.cachedir, 'train_data.p')
        cfg_dict = create_dlc_cfg_dict(conf)
        [p, d] = os.path.split(db_file)
        cfg_dict['project_path'] = p
        cfg_dict['dataset'] = d
        print('Checking db from {}'.format(db_file))
        read_fn, n = deeplabcut.train.get_read_fn(cfg_dict)
    else:
        db_file = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
        print('Checking db from {}'.format(db_file))
        tf_iterator = multiResData.tf_reader(conf, db_file, False)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        n = tf_iterator.N
        # raise ValueError('Undefined model type')

    n_out = 50
    samples = np.linspace(0, n - 1, n_out).astype('int')
    all_f = np.zeros((n_out,) + conf.imsz + (conf.img_dim,))
    labeled_locs = np.zeros([n_out, conf.n_classes, 2])
    count = 0
    info = []
    for cur_b in range(n):
        next_db = read_fn()
        if cur_b in samples:
            all_f[count, ...] = next_db[0]
            labeled_locs[count, ...] = next_db[1]
            info.append(next_db[2])
            count += 1

    with open(out_file, 'w') as f:
        pickle.dump({'ims': all_f, 'locs': labeled_locs, 'info': np.array(info)}, f, protocol=2)


# KB 20190123: classify a list of movies, targets, and frames
# save results to mat file out_file

def compile_trk_info(conf, model_file, crop_loc, expname=None):
    '''
    Compile classification/predict metadata stored in eg trkfile.trkInfo

    crop_loc should be 0-based as it is converted here with to_mat
    :return:
    '''

    if expname is None:
        expname = os.path.basename(conf.cachedir)

    info = {}  # tracking info. Can be empty.
    info[u'model_file'] = model_file
    modelfilets = model_file + '.index'
    modelfilets = modelfilets if os.path.exists(modelfilets) else model_file
    if not os.path.exists(modelfilets):
        raise ValueError('Files %s and %s do not exist' % (model_file, model_file + '.meta'))
    info[u'trnTS'] = get_matlab_ts(modelfilets)
    info[u'name'] = expname
    param_dict = convert_unicode(conf.__dict__.copy())
    param_dict.pop('cropLoc', None)
    info[u'params'] = param_dict
    if 'flipLandmarkMatches' in param_dict.keys() and not param_dict['flipLandmarkMatches']:
        param_dict['flipLandmarkMatches'] = None
    info[u'crop_loc'] = to_mat(crop_loc)
    info[u'project_file'] = getattr(conf, 'project_file', '')
    info[u'git_commit'] = PoseTools.get_git_commit()

    return info


def to_mat_all_locs_in_dict(ret_dict):
    '''
    Convert all dict vals of 'locs' properties using to_mat.

    Modifies ret_dict in place.

    :param ret_dict:
    :return:
    '''
    for k in ret_dict.keys():
        if k.startswith('locs') or k.endswith('locs'):
            ret_dict[k] = to_mat(ret_dict[k])


def classify_db_stage(args,view,view_ndx,db_file):
    lbl_file = args.lbl_file
    name = args.name
    if args.stage == 'multi':
        conf = create_conf(lbl_file,view,name,args.type,cache_dir=args.cache,conf_params=args.conf_params,first_stage=True)
        conf2 = create_conf(lbl_file,view,name,args.type2,cache_dir=args.cache,conf_params=args.conf_params2,second_stage=True)
        model_file = args.model_file[view_ndx]
        model_file2 = args.model_file2[view_ndx]
        ret_dict = classify_db_all(model_type=args.type,model_type2=args.type2,conf=conf,conf2=conf2,model_file=model_file,model_file2=model_file2,name=args.train_name,img_dir='train')

    else:
        conf = create_conf(lbl_file,view,name,args.type,cache_dir=args.cache,conf_params=args.conf_params)
        model_file = args.model_file
        ret_dict = classify_db_all(model_type=args.type,conf=conf,model_file=model_file,name=args.train_name,img_dir='train')

    return ret_dict


def classify_list_file(args, view, view_ndx=0):
    # ivw is the index into movieFiles, trxFiles. It corresponds to view, but
    # perhaps not absolute view. E.g. if view 1 is the only view being tracked in
    # this call, then movieFiles should have only one movie per movie set, and
    # ivw=0.
    # If views 0 and 1 are tracked in this call to APT_interface, then movieFiles
    # should have two movies per movie set, and ivw=0 corresponds to view 0 and ivw=1
    # corresponds to view 1. 
    success = False
    pred_locs = None
    list_file = args.list_file
    list_fp = open(list_file, 'r')
    out_file = args.out_files[view_ndx]
    part_file = out_file + '.part'  # following classify_movie naming

    if not os.path.isfile(list_file):
        print('File %s does not exist' % list_file)
        return success, pred_locs

    toTrack = json.load(list_fp)

    # minimal checks
    if 'movieFiles' not in toTrack:
        logging.exception('movieFiles not defined in json file %s' % list_file)
        return success, pred_locs
    nMovies = len(toTrack['movieFiles'])
    if 'toTrack' not in toTrack:
        logging.exception('toTrack list not defined in json file %s' % list_file)
        return success, pred_locs

    # select the movies & trxfiles corresponding to current view index
    movieFiles = []
    for movieset in toTrack['movieFiles']:
        if isinstance(movieset, list):
            movieFiles.append(movieset[view_ndx])
        else:
            movieFiles.append(movieset)
    toTrack['movieFiles'] = movieFiles

    hasTrx = 'trxFiles' in toTrack and toTrack['trxFiles']
    trxFiles = []
    if hasTrx:
        nTrx = len(toTrack['trxFiles'])
        if nTrx != nMovies:
            logging.exception('Numbers of movies and trx files do not match')
            return success, pred_locs
        for trxset in toTrack['trxFiles']:
            if isinstance(trxset, list):
                trxFiles.append(trxset[view_ndx])
            else:
                trxFiles.append(trxset)
        toTrack['trxFiles'] = trxFiles
    else:
        toTrack['trxFiles'] = [None,]*len(toTrack['movieFiles'])

    hasCrops = 'cropLocs' in toTrack
    cropLocs = None
    if hasCrops:
        nCrops = len(toTrack['cropLocs'])
        if nCrops != nMovies:
            logging.exception('Number of movie files and cropLocs do not match')
            return success, pred_locs
        cropLocs = toTrack['cropLocs']

    # 1-indexed!
    nToTrack = len(toTrack['toTrack'])
    cur_list = []
    for i in range(nToTrack):
        mov = toTrack['toTrack'][i][0]
        tgt = toTrack['toTrack'][i][1]
        frm = toTrack['toTrack'][i][2]
        if mov <= 0 or mov > nMovies:
            logging.exception('toTrack[%d] has out of range movie index %d' % (i, mov))
            return success, pred_locs
        if tgt <= 0:
            logging.exception('toTrack[%d] has out of range target index %d' % (i, tgt))
            return success, pred_locs
        if isinstance(frm, int) and frm <= 0:
            logging.exception('toTrack[%d] has out of range frame index %d' % (i, frm))
            return success, pred_locs

        if isinstance(frm, int):
            cur_list.append([mov - 1, frm - 1, tgt - 1])
        elif isinstance(frm, list):
            assert len(frm) == 2, 'Invalid frame specification in toTrack[%d]' % (i)
            logging.warning('toTrack[%d] has frm-range specification [%d,%d]. Adding %d frames' % (
            i, frm[0], frm[1], frm[1] - frm[0]))
            for frmreal in range(frm[0], frm[1]):
                cur_list.append([mov - 1, frmreal - 1, tgt - 1])
        else:
            assert False, 'Invalid frame specification in toTrack[%d]' % (i)

    db_dict = {'moviesFiles':toTrack['movieFiles'],'trxFiles':toTrack['trxFiles'],'cropLocs':cropLocs,'toTrack':cur_list}
    db_file = tempfile.mkstemp()[1]
    db_file_val = tempfile.mkstemp()[1]

    lbl_file = args.lbl_file
    name = args.name
    first_stage = args.stage=='multi' or args.stage=='first'
    conf = create_conf(lbl_file,view,name,cache_dir=args.cache, net_type=args.type,conf_params=args.conf_params,first_stage=first_stage)

    if conf.db_format == 'coco':
        create_coco_db(conf,split=False,db_files=(db_file,db_file_val),use_cache=False,db_dict=db_dict)
    else:
        create_tfrecord(conf,split=False,db_files=(db_file,db_file_val),use_cache=False,db_dict=db_dict)

    ret_dict_all = classify_db_stage(args,view,db_file)

    ret_dict_all1 = classify_list_all(args.type, conf, cur_list,
                                     on_gt=False,
                                     model_file=args.model_file[view_ndx],
                                     movie_files=movieFiles,
                                     trx_files=trxFiles,
                                     crop_locs=cropLocs,
                                     part_file=part_file)
    # print('ret_dict_all.keys' + str(ret_dict_all.keys()))

    # this is copy-pasted from write_trk --
    # should probably merge these functions at some point
    ts_shape = ret_dict_all['locs'].shape[:-1]
    ts = np.ones(ts_shape) * datetime2matlabdn()  # time stamp
    tag = np.zeros(ts.shape).astype('bool')  # tag which is always false for now.

    to_mat_all_locs_in_dict(ret_dict_all)
    # output to a temporary file and then rename. 
    # this is because existence of out_file is a flag that tracking is done
    # and file might still be being written when file is discovered
    out_file_tmp = out_file + '.tmp'
    savemat_with_catch_and_pickle(out_file_tmp, {'pred_locs': ret_dict_all['locs'], 'pred_conf': ret_dict_all['conf'], 'pred_ts': ts, 'pred_tag': tag, 'list_file': list_file, 'to_track': toTrack})
    if os.path.exists(out_file_tmp):
        os.rename(out_file_tmp, out_file)
        success = True
    else:
        logging.exception("Did not successfully write output to %s" % out_file_tmp)
        success = False

    return success, pred_locs


def classify_gt_data(args,view,view_ndx):
    ''' Classify GT data in the label file.

    View classified is per conf.view; out_file, model_file should be specified for this view.

    Saved values are 1-indexed.
    '''

    lbl_file = args.lbl_file
    name = args.name
    assert args.stage not in ['first','second'], 'GT classification can not be done in individual stages'
    if args.stage == 'multi':
        conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache, conf_params=args.conf_params,first_stage=True)
        conf2 = create_conf(lbl_file, view, name, net_type=args.type2, cache_dir=args.cache,conf_params=args.conf_params2, second_stage=True)
    else:
        conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache, conf_params=args.conf_params)
        conf2 = None
        model_type = args.type
        model_type2 = None
    # out_file = args.out_files + '_{}.mat'.format(view)

    # create tfr/db
    now_str = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    gttfr = 'gt_{}.tfrecords'.format(now_str)
    gttfr = os.path.join(conf.cachedir, gttfr)
    conf.db_format = 'tfrecord'
    create_tfrecord(conf, split=False, on_gt=True, db_files=(gttfr,), use_gt_cache=True)

    model_file = args.model_file[view_ndx]
    out_file = args.out_files[view_ndx]
    # classify it
    ret = classify_db_all(model_type, conf, gttfr, model_file=model_file, classify_fcn=classify_db2, fullret=True, ignore_hmaps=True, conf2=conf2, model_type2=model_type2)
    ret_dict_all, labeled_locs, info = ret
    info = list(itertools.chain.from_iterable(info))  # info is list-of-list-of-triplets (second list is batches)

    # partfile = out_file + '.part'
    # ret_dict_all = classify_list_all(model_type, conf, cur_list,
    #                                  on_gt=True,
    #                                  model_file=model_file,
    #                                  part_file=partfile)

    ret_dict_all['locs_labeled'] = np.array(labeled_locs)  # AL: already np.array prob dont need copy
    to_mat_all_locs_in_dict(ret_dict_all)
    ret_dict_all['list'] = to_mat(np.array(info,dtype='double'))
    DUMMY_CROP_INFO = []
    ret_dict_all['trkInfo'] = compile_trk_info(conf, model_file, DUMMY_CROP_INFO)

    savemat_with_catch_and_pickle(out_file, ret_dict_all)

    # lbl.close()


def convert_to_mat_trk(pred_locs, conf, start, end, trx_ids):
    ''' Converts predictions to compatible trk format'''
    # pred_locs = in_pred.copy()
    if not conf.is_multi:
        pred_locs = pred_locs[:, trx_ids, ...]
    pred_locs = pred_locs[:(end - start), ...]
    if pred_locs.ndim == 4:
        pred_locs = pred_locs.transpose([2, 3, 0, 1])
    else:
        pred_locs = pred_locs.transpose([2, 0, 1])
    if not (conf.has_trx_file or conf.is_multi or conf.use_ht_trx):
        pred_locs = pred_locs[..., 0]

    ps = np.array(pred_locs.shape)
    idx_f = np.where(~np.isnan(pred_locs.flat))[0]
    vals = pred_locs.flat[idx_f]

    # Convert idx from python's C format to matlab's fortran format
    idx = np.unravel_index(idx_f, ps)
    idx = np.ravel_multi_index(idx[::-1], np.flip(ps))
    idx = to_mat(idx)
    vals = to_mat(vals)
    pred_dict = {'idx': idx, 'val': vals, 'size': ps, 'type': 'nan'}

    return pred_dict


def write_trk(out_file, pred_locs_in, extra_dict, start, end, trx_ids, conf, info, mov_file):
    '''
    pred_locs is the predicted locations of size
    n_frames x n_Trx x n_body_parts x 2
    n_done is the number of frames that have been tracked.
    everything should be 0-indexed
    '''

    locs_lnk = np.transpose(pred_locs_in, [2, 3, 0, 1])

    ts = np.ones_like(locs_lnk[:, 0, ...]) * datetime2matlabdn()
    tag = np.ones(ts.shape)*np.nan  # tag which is always false for now.
    if 'conf' in extra_dict:
        pred_conf = extra_dict['conf']
        locs_conf = np.transpose(pred_conf, [2, 0, 1])
    else:
        locs_conf = None

    if 'occ' in extra_dict:
        pred_occ = extra_dict['occ']
        tag = np.transpose(pred_occ, [2, 0, 1])

    trk = TrkFile.Trk(p=locs_lnk, pTrkTS=ts, pTrkTag=tag, pTrkConf=locs_conf)
    trk.T0 = start
    trk.save(out_file, saveformat='tracklet', trkInfo=info)
    return

    # Old code that saves extra information. Keeping it in for now MK - 20210319
    pred_locs = convert_to_mat_trk(pred_locs_in, conf, start, end, trx_ids)

    tgt = to_mat(np.array(trx_ids))  # target animals that have been tracked.
    # For projects without trx file this is always 1.
    ts_shape = pred_locs['size'][0:1].tolist() + pred_locs['size'][2:].tolist()
    ts = np.ones(ts_shape) * datetime2matlabdn()  # time stamp
    tag = np.zeros(ts.shape).astype('bool')  # tag which is always false for now.
    tracked_shape = pred_locs['size'][2]
    tracked = np.zeros([1,
                        tracked_shape])  # which of the predlocs have been tracked. Mostly to help APT know how much tracking has been done.
    tracked[0, :] = to_mat(np.arange(start, end))

    out_dict = {'pTrk': pred_locs,
                'pTrkTS': ts,
                'expname': mov_file,
                'pTrkiTgt': tgt,
                'pTrkTag': tag,
                'pTrkFrm': tracked,
                'trkInfo': info}
    for k in extra_dict.keys():
        tmp = convert_to_mat_trk(extra_dict[k], conf, start, end, trx_ids)
        # if k.startswith('locs_'):
        #     tmp = to_mat(tmp)
        out_dict['pTrk' + k] = tmp

    # output to a temporary file and then rename to real file name.
    # this is because existence of trk file is a flag that tracking is done for
    # other processes, and writing may still be in progress when file discovered.
    out_file_tmp = out_file + '.tmp'
    savemat_with_catch_and_pickle(out_file_tmp, out_dict)
    if os.path.exists(out_file_tmp):
        os.replace(out_file_tmp, out_file)
    else:
        logging.exception("Did not successfully write output to %s" % out_file_tmp)


def classify_movie(conf, pred_fn, model_type,
                   mov_file='',
                   out_file='',
                   trx_file=None,
                   start_frame=0,
                   end_frame=-1,
                   skip_rate=1,
                   trx_ids=(),
                   model_file='',
                   name='',
                   nskip_partfile=500, 
                   save_hmaps=False,
                   crop_loc=None):
    ''' Classifies frames in a movie. All animals in a frame are classified before moving to the next frame.'''

    logging.info('classify_movie:')
    logging.info('mov_file: %s' % mov_file)
    logging.info('out_file: %s' % out_file)
    logging.info('trx_file: %s' % trx_file)
    logging.info('start_frame: ' + str(start_frame))
    logging.info('end_frame: ' + str(end_frame))
    logging.info('skip_rate: ' + str(skip_rate))
    logging.info('trx_ids: ' + str(trx_ids))
    logging.info('model_file: %s' % model_file)
    logging.info('name: %s' % name)
    logging.info('save_hmaps: ' + str(save_hmaps))
    logging.info('crop_loc: ' + str(crop_loc))

    lnk_cost = conf.get('multi_link_cost', 5)

    cap = movies.Movie(mov_file)
    sz = (cap.get_height(), cap.get_width())
    n_frames = int(cap.get_n_frames())
    trx_dict = get_trx_info(trx_file, conf, n_frames)
    T = trx_dict['trx']; n_trx = trx_dict['n_trx']
    first_frames = trx_dict['first_frames']; end_frames = trx_dict['end_frames']

    # For multi-animal T is [None,] and n_trx is conf.max_n_animals. With this combination, rest of the workflow seems to work. Totally unintentional but I'm not going to update it if it is working. MK 20201111
    has_trx = conf.has_trx_file or conf.use_ht_trx or conf.use_bbox_trx
    trx_ids = get_trx_ids(trx_ids, n_trx, has_trx)
    conf.batch_size = 1 if model_type == 'deeplabcut' else conf.batch_size
    bsize = conf.batch_size
    flipud = conf.flipud

    info = compile_trk_info(conf, model_file, crop_loc, expname=name)

    if end_frames.size==0:
        pred_locs = np.zeros([1,0,conf.n_classes,2])
        write_trk(out_file, pred_locs, {}, 0, 1, [], conf, info, mov_file)
        return

    if end_frame < 0: end_frame = end_frames.max()
    if end_frame > end_frames.max(): end_frame = end_frames.max()
    if start_frame > end_frame: return None

    max_n_frames = end_frame - start_frame
    min_first_frame = start_frame
    pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
    pred_locs[:] = np.nan

    extra_dict = {}

    hmap_out_dir = os.path.splitext(out_file)[0] + '_hmap'
    if (not os.path.exists(hmap_out_dir)) and save_hmaps:
        os.mkdir(hmap_out_dir)

    to_do_list = []
    for cur_f in range(start_frame, end_frame,skip_rate):
        for t in range(n_trx):
            if not np.any(trx_ids == t):
                continue
            if (end_frames[t] > cur_f) and (first_frames[t] <= cur_f):
                if T[t] is None or (not np.isnan(T[t]['x'][0, cur_f - first_frames[t]])):
                    to_do_list.append([cur_f, t])

    # TODO: this stuff is really similar to classify_list, some refactor
    # likely useful

    n_list = len(to_do_list)
    n_batches = int(math.ceil(float(n_list) / bsize))
    for cur_b in tqdm(range(n_batches)):
        cur_start = cur_b * bsize
        ppe = min(n_list - cur_start, bsize)
        all_f = create_batch_ims(to_do_list[cur_start:(cur_start + ppe)], conf, cap, flipud, T, crop_loc)

        ret_dict = pred_fn(all_f)
        base_locs = ret_dict.pop('locs')
        # hmaps = ret_dict.pop('hmaps')

        assert not save_hmaps
        # if save_hmaps:
        # mat_out = os.path.join(hmap_out_dir, 'hmap_batch_{}.mat'.format(cur_b+1))
        # hdf5storage.savemat(mat_out,{'hm':hmaps,'startframe1b':to_do_list[cur_start][0]+1})

        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            trx_ndx = cur_entry[1]
            cur_trx = T[trx_ndx]
            cur_f = cur_entry[0]
            base_locs_orig = convert_to_orig(base_locs[cur_t, ...], conf, cur_f, cur_trx, crop_loc)
            if conf.is_multi:
                # doing only this seems to work
                pred_locs[cur_f - min_first_frame, :, :, :] = base_locs_orig[...]
            else:
                pred_locs[cur_f - min_first_frame, trx_ndx, :, :] = base_locs_orig[...]

            # if save_hmaps:
            #    write_hmaps(hmaps[cur_t, ...], hmap_out_dir, trx_ndx, cur_f)

            # for everything else that is returned..
            for k in ret_dict.keys():

                if (ret_dict[k].ndim == 4 and (not conf.is_multi)) or ret_dict[k].ndim == 5:  # hmaps
                    # if save_hmaps:
                    #    cur_hmap = ret_dict[k]
                    #    write_hmaps(cur_hmap[cur_t, ...], hmap_out_dir, trx_ndx, cur_f, k[5:])
                    pass
                else:
                    cur_v = ret_dict[k]
                    # py3 and py2 compatible
                    if k not in extra_dict:
                        sz = cur_v.shape[1:]
                        if conf.is_multi:
                            extra_dict[k] = np.zeros((max_n_frames,) + sz)
                        else:
                            extra_dict[k] = np.zeros((max_n_frames, n_trx) + sz)

                    if k.startswith('locs'):  # transform locs
                        cur_orig = convert_to_orig(cur_v[cur_t, ...], conf, cur_f, cur_trx, crop_loc)
                    else:
                        cur_orig = cur_v[cur_t, ...]

                    if conf.is_multi:
                        extra_dict[k][cur_f - min_first_frame, ...] = cur_orig
                    else:
                        extra_dict[k][cur_f - min_first_frame, trx_ndx, ...] = cur_orig

        # if cur_b % 20 == 19:
        #     sys.stdout.write('.')
        if (cur_b % nskip_partfile == 0) & (cur_b > 0):
            # sys.stdout.write('\n')
            T1 = to_do_list[cur_start][0]
            write_trk(out_file + '.part', pred_locs, extra_dict, start_frame, T1, trx_ids, conf, info, mov_file)

    # Get the animal confidences for 2 stage tracking
    pred_animal_conf = None
    if conf.use_bbox_trx or conf.use_ht_trx:
        pred_animal_conf = np.ones([max_n_frames, n_trx,conf.n_classes])*-1
        if 'conf' in T[0]:
            for ix in range(n_trx):
                for cur_f in range(start_frame, end_frame):
                    if (end_frames[ix] > cur_f) and (first_frames[ix] <= cur_f):
                        pred_animal_conf[ cur_f- min_first_frame,ix,:] = T[ix]['conf'][0,cur_f-first_frames[ix],0:1]


    if (conf.is_multi and (conf.stage==None)) or (conf.stage==conf.multi_link_stage):
        # write out raw results before linking.
        pre_fix, ext = os.path.splitext(out_file)
        raw_file = pre_fix + '_raw' + ext
        write_trk(raw_file, pred_locs, extra_dict, start_frame, end_frame, trx_ids, conf, info, mov_file)
        pred_conf = extra_dict['conf'] if 'conf' in extra_dict else None
        trk = lnk.link_trklets(pred_locs, conf, mov_file, out_file,pred_conf=pred_conf, pred_animal_conf=pred_animal_conf)
        trk.T0 = start_frame
        out_file_tracklet = out_file
        trk.save(out_file_tracklet, saveformat='tracklet', trkInfo=info)
    else:
        write_trk(out_file, pred_locs, extra_dict, start_frame, end_frame, trx_ids, conf, info, mov_file)

    if os.path.exists(out_file + '.part') and not conf.is_multi:
        os.remove(out_file + '.part')
    cap.close()
    tf.reset_default_graph()
    return pred_locs


def get_unet_pred_fn(conf, model_file=None, name='deepnet'):
    ''' Prediction function for UNet network'''
    tf.reset_default_graph()
    self = PoseUNet.PoseUNet(conf, name=name)
    if name == 'deepnet':
        self.train_data_name = 'traindata'
    return self.get_pred_fn(model_file)


def get_mdn_pred_fn(conf, model_file=None, name='deepnet', distort=False, **kwargs):
    tf.reset_default_graph()
    self = PoseURes.PoseUMDN_resnet(conf, name=name)
    if name == 'deepnet':
        self.train_data_name = 'traindata'
    else:
        self.train_data_name = None

    return self.get_pred_fn(model_file, distort=distort, **kwargs)


def get_latest_model_files(conf, net_type='mdn', name='deepnet'):
    if net_type == 'mdn':
        self = PoseURes.PoseUMDN_resnet(conf, name=name)
        if name == 'deepnet':
            self.train_data_name = 'traindata'
        files = self.model_files()
    elif net_type == 'unet':
        self = PoseUNet.PoseUNet(conf, name=name)
        if name == 'deepnet':
            self.train_data_name = 'traindata'
        files = self.model_files()
    elif net_type == 'leap':
        files = leap.training.model_files(conf, name)
    elif net_type == 'openpose' or net_type == 'sb':
        files = op.model_files(conf, name)
    elif net_type == 'deeplabcut':
        files = deeplabcut.pose_estimation_tensorflow.model_files(create_dlc_cfg_dict(conf, name), name)
    else:
        assert False, 'Undefined Net Type'

    for f in files:
        assert os.path.exists(f), 'Model file {} does not exist'.format(f)

    return files


def classify_movie_all(model_type, **kwargs):
    ''' Classify movie wrapper'''
    conf = kwargs['conf']
    model_file = kwargs['model_file']
    train_name = kwargs['train_name']
    del kwargs['model_file'], kwargs['conf'], kwargs['train_name']
    if conf.stage == 'first':
        conf.n_classes = 2
    pred_fn, close_fn, model_file = get_pred_fn(model_type, conf, model_file, name=train_name)
    logging.info('Saving hmaps') if kwargs['save_hmaps'] else logging.info('NOT saving hmaps')
    try:
        classify_movie(conf, pred_fn, model_type, model_file=model_file, **kwargs)
    except (IOError, ValueError) as e:
        close_fn()
        logging.exception('Could not track movie')
    close_fn()


def train_unet(conf, args, restore, split, split_file=None):
    if not args.skip_db:
        create_tfrecord(conf, split=split, use_cache=args.use_cache, split_file=split_file)
    tf.reset_default_graph()
    self = PoseUNet.PoseUNet(conf, name=args.train_name)
    if args.train_name == 'deepnet':
        self.train_data_name = 'traindata'
    else:
        self.train_data_name = None
    self.train_unet(restore=restore)


def train_mdn(conf, args, restore, split, split_file=None, model_file=None):
    if not args.skip_db:
        create_tfrecord(conf, split=split, use_cache=args.use_cache, split_file=split_file)
    tf.reset_default_graph()
    self = PoseURes.PoseUMDN_resnet(conf, name=args.train_name)
    if args.train_name == 'deepnet':
        self.train_data_name = 'traindata'
    else:
        self.train_data_name = None
    self.train_umdn(restore=restore, model_file=model_file)
    tf.reset_default_graph()


def train_leap(conf, args, split, split_file=None):
    assert (
                conf.dl_steps % conf.display_step == 0), 'Number of training iterations must be a multiple of display steps for LEAP'

    if not args.skip_db:
        create_leap_db(conf, split=split, use_cache=args.use_cache, split_file=split_file)

    leap_train(data_path=os.path.join(conf.cachedir, 'leap_train.h5'),
               base_output_path=conf.cachedir,
               run_name=args.train_name,
               net_name=conf.leap_net_name,
               box_dset="box",
               confmap_dset="joints",
               val_size=conf.leap_val_size,
               preshuffle=conf.leap_preshuffle,
               filters=int(conf.leap_filters),
               rotate_angle=conf.rrange,
               epochs=conf.dl_steps // conf.display_step,
               batch_size=conf.batch_size,
               batches_per_epoch=conf.display_step,
               val_batches_per_epoch=conf.leap_val_batches_per_epoch,
               reduce_lr_factor=conf.leap_reduce_lr_factor,
               reduce_lr_patience=conf.leap_reduce_lr_patience,
               reduce_lr_min_delta=conf.leap_reduce_lr_min_delta,
               reduce_lr_cooldown=conf.leap_reduce_lr_cooldown,
               reduce_lr_min_lr=conf.leap_reduce_lr_min_lr,
               amsgrad=conf.leap_amsgrad,
               upsampling_layers=conf.leap_upsampling,
               conf=conf)

    tf.reset_default_graph()


def train_openpose(conf, args, split, split_file=None):
    if not args.skip_db:
        create_tfrecord(conf, split=split, use_cache=args.use_cache, split_file=split_file)

    nodes = []
    graph = conf.op_affinity_graph
    _ = [nodes.extend(n) for n in graph]
    # assert len(graph) == (conf.n_classes - 1) and len(
    # set(nodes)) == conf.n_classes, 'Affinity Graph for open pose is not a complete tree'

    op.training(conf, name=args.train_name)
    tf.reset_default_graph()


def train_sb(conf, args, split, split_file=None):
    if not args.skip_db:
        create_tfrecord(conf, split=split, use_cache=args.use_cache, split_file=split_file)
    sb.training(conf, name=args.train_name)
    tf.reset_default_graph()


def train_deepcut(conf, args, split_file=None, model_file=None):
    if not args.skip_db:
        create_deepcut_db(conf, False, use_cache=args.use_cache, split_file=split_file)

    cfg_dict = create_dlc_cfg_dict(conf, args.train_name)
    if model_file is not None:
        cfg_dict.init_weights = model_file
    deepcut_train(cfg_dict,
                  displayiters=conf.display_step,
                  saveiters=conf.save_step,
                  maxiters=cfg_dict['dlc_train_steps'],
                  max_to_keep=conf.maxckpt)
    tf.reset_default_graph()


def train_dpk(conf, args, split, split_file=None):
    if not args.skip_db:
        create_tfrecord(conf,
                        split=split,
                        use_cache=args.use_cache,
                        split_file=split_file)

    tf.reset_default_graph()
    apt_dpk.train(conf)


def create_dlc_cfg_dict(conf, train_name='deepnet'):
    url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    wt_dir = os.path.join(script_dir, 'pretrained')
    wt_file = os.path.join(wt_dir, 'resnet_v1_50.ckpt')
    if hasattr(conf, 'tfactor_range'):
        minsz = max(conf.imsz) / 2 * (1 - conf.tfactor_range)
        wd_x = conf.imsz[1] * conf.tfactor_range / 2
        wd_y = conf.imsz[0] * conf.tfactor_range / 2
    else:
        minsz = max(conf.imsz) / 2 - conf.trange / 2
        wd_x = conf.trange / 2
        wd_y = conf.trange / 2
    if not os.path.exists(wt_file):
        print('Downloading pretrained weights..')
        if not os.path.exists(wt_dir):
            os.makedirs(wt_dir)
        sname, header = urllib.request.urlretrieve(url)
        tar = tarfile.open(sname, "r:gz")
        print('Extracting pretrained weights..')
        tar.extractall(path=wt_dir)
    pretrained_weights = os.path.join(wt_dir, 'resnet_v1_50.ckpt')

    symmetric_joints = list(range(conf.n_classes))
    for k in conf.flipLandmarkMatches.keys():
        symmetric_joints[int(k)] = int(conf.flipLandmarkMatches[k])
    cfg_dict = {'snapshot_prefix': os.path.join(conf.cachedir, train_name),
                'project_path': conf.cachedir,
                'dataset': conf.dlc_train_data_file,
                'init_weights': pretrained_weights,
                'dlc_train_steps': conf.dl_steps,
                'symmetric_joints': symmetric_joints,
                'num_joints': conf.n_classes,
                'all_joints': [[i] for i in range(conf.n_classes)],
                'all_joints_names': ['part_{}'.format(i) for i in range(conf.n_classes)],
                'dataset_type': 'default',  # conf.dlc_augmentation_type,
                'global_scale': 1. / conf.rescale,
                'scale_jitter_lo': min(1 / conf.scale_factor_range, conf.scale_factor_range),
                'scale_jitter_up': max(1 / conf.scale_factor_range, conf.scale_factor_range),
                'net_type': 'resnet_50',
                'pos_dist_thresh': 17,
                'intermediate_supervision': getattr(conf, 'dlc_intermediate_supervision', False),  # False
                'intermediate_supervision_layer': getattr(conf, 'dlc_intermediate_supervision_layer', 12),  # 12,
                'location_refinement': getattr(conf, 'dlc_location_refinement', True),  # True,
                'locref_huber_loss': getattr(conf, 'dlc_locref_huber_loss', True),  # True,
                'locref_loss_weight': getattr(conf, 'dlc_locref_loss_weight', 0.05),  # 0.05,
                'locref_stdev': getattr(conf, 'dlc_locref_stdev', 7.2801),  # 7.2801,
                'img_dim': conf.img_dim,
                'dlc_use_apt_preprocess': getattr(conf, 'dlc_use_apt_preprocess', True),
                'scale_factor_range': conf.scale_factor_range,
                'brange': conf.brange,
                'crange': conf.crange,
                'trange': conf.trange,
                'rrange': conf.rrange,
                'horz_flip': conf.horz_flip,
                'vert_flip': conf.vert_flip,
                'adjust_contrast': conf.adjust_contrast,
                'flipLandmarkMatches': conf.flipLandmarkMatches,
                'normalize_img_mean': conf.normalize_img_mean,
                'use_scale_factor_range': conf.use_scale_factor_range,
                'imsz': conf.imsz,
                'imax': conf.imax,
                'rescale': conf.rescale,
                'clahe_grid_size': conf.clahe_grid_size,
                'check_bounds_distort': conf.check_bounds_distort,
                'expname': conf.expname,
                'rot_prob': conf.rot_prob,

                # 'minsize':minsz,
                #              'leftwidth':wd_x,
                #              'rightwidth':wd_x,
                #              'topheight':wd_y,
                #              'bottomheight': wd_y,
                'mirror': False  # switch to cfg.horz_flip if dataset is imgaug
                }
    return cfg_dict


def train_multi_stage(args, nviews):
    name = args.name
    lbl_file = args.lbl_file
    if args.stage == 'multi':
        # if not args.debug:
        #     args1 = {'lbl_file':lbl_file,'nviews':nviews,'name':name,'args':args,'first_stage':True}
        #     p1 = multiprocessing.Process(target=train,args=args1)
        #     p1.start()
        #     p1.join()
        #
        #     args.type = args.type2
        #     args.conf_params2 = args.conf_params
        #     args2 = {'lbl_file':args.lbl_file2,'nviews':nviews,'name':name,'args':args,'second_stage':True}
        #     p2 = multiprocessing.Process(target=train,args=args2)
        #     p2.start()
        #     p2.join()
        # else:
        train(lbl_file, nviews, name, args, first_stage=True)
        args.type = args.type2
        args.conf_params = args.conf_params2
        args.model_file = args.model_file2
        train(lbl_file, nviews, name, args, second_stage=True)
    elif args.stage == 'first':
        train(lbl_file, nviews, name, args, first_stage=True)
    elif args.stage == 'second':
        train(lbl_file, nviews, name, args, second_stage=True)
    else:
        train(lbl_file, nviews, name, args)


def train(lblfile, nviews, name, args,first_stage=False,second_stage=False):
    ''' Creates training db and calls the appropriate network's training function '''

    view = args.view
    net_type = args.type
    restore = args.restore
    if view is None:
        views = range(nviews)
    else:
        views = [view]

    for view_ndx, cur_view in enumerate(views):
        conf = create_conf(lblfile, cur_view, name, net_type=net_type, cache_dir=args.cache,conf_params=args.conf_params, json_trn_file=args.json_trn_file,first_stage=first_stage,second_stage=second_stage)

        conf.view = cur_view
        model_file = args.model_file[view_ndx]
        if args.split_file is not None:
            assert (os.path.exists(args.split_file))
            in_data = PoseTools.json_load(args.split_file)
            out_data = []
            for d in in_data:
                out_data.append((np.array(d) - 1).tolist())
            # tempfile.tempdir returns None for bsub/sing
            # t_file = os.path.join(tempfile.tempdir,next(tempfile._get_candidate_names()))
            t_file = args.split_file + ".temp0b"
            with open(t_file, 'w') as f:
                json.dump(out_data, f)
            conf.splitType = 'predefined'
            split = True
            split_file = t_file
        else:
            split = False
            split_file = None

        try:
            if net_type == 'unet':
                train_unet(conf, args, restore, split, split_file=split_file)
            elif net_type == 'mdn':
                train_mdn(conf, args, restore, split, split_file=split_file, model_file=model_file)
            # elif net_type == 'openpose':
            #     if args.use_defaults:
            #         op.set_openpose_defaults(conf)
            #     train_openpose(conf, args, split, split_file=split_file)
            elif net_type == 'sb':
                assert not args.use_defaults
                train_sb(conf, args, split, split_file=split_file)
            elif net_type == 'leap':
                if args.use_defaults:
                    leap.training.set_leap_defaults(conf)
                train_leap(conf, args, split, split_file=split_file)
            elif net_type == 'deeplabcut':
                if args.use_defaults:
                    deeplabcut.train.set_deepcut_defaults(conf)
                train_deepcut(conf, args, split_file=split_file, model_file=model_file)
            elif net_type == 'dpk':
                train_dpk(conf, args, split, split_file=split_file)

            else:
                if conf.is_multi:
                    setup_ma(conf)
                if not args.skip_db:
                    if conf.db_format == 'coco':
                        create_coco_db(conf, split=split, split_file=split_file)
                    else:
                        create_tfrecord(conf, split=split, use_cache=args.use_cache, split_file=split_file)

                if conf.multi_only_ht:
                    assert conf.stage!= 'second', 'multi_ony_ht should be True only for the first stage'
                    conf.n_classes = 2
                    conf.flipLandmarkMatches = {}

                module_name = 'Pose_{}'.format(net_type)
                pose_module = __import__(module_name)
                tf.reset_default_graph()
                self = getattr(pose_module, module_name)(conf, name=args.train_name)
                # self.name = args.train_name
                self.train_wrapper(restore=restore, model_file=model_file)

        except tf.errors.InternalError as e:
            logging.exception(
                'Could not create a tf session. Probably because the CUDA_VISIBLE_DEVICES is not set properly')
            exit(1)
        except tf.errors.ResourceExhaustedError as e:
            logging.exception('Out of GPU Memory. Either reduce the batch size or scale down the image')
            exit(1)

        # Disabling this because it is hardly used MK 20210712
        # if args.classify_val:
        #     val_filename = get_valfilename(conf, net_type)
        #     db_file = os.path.join(conf.cachedir, val_filename)
        #     logging.info("Classifying {}... ".format(db_file))
        #     preds, locs, info, model_file = classify_db_all(net_type, conf, db_file)
        #     preds = to_mat(preds)
        #     locs = to_mat(locs)
        #     info = to_mat(info)
        #     out_file = args.classify_val_out
        #     logging.info("... done classifying, used model {}. Saving to {}".format(model_file, out_file))
        #     out_dict = {'preds': preds, 'locs': locs, 'info': info, 'model_file': model_file}
        #     hdf5storage.savemat(out_file, out_dict, appendmat=False, truncate_existing=True)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("lbl_file", help="path to lbl file")
    parser.add_argument("-lbl_file2", help="path to lbl file for second stage")
    parser.add_argument('-json_trn_file', dest='json_trn_file', help='Json file containing label information',
                        default=None)
    parser.add_argument('-name', dest='name', help='Name for the run. Default - apt', default='apt')
    parser.add_argument('-view', dest='view', help='Run only for this view. If not specified, run for all views',
                        default=None, type=int)
    parser.add_argument('-model_files', dest='model_file', help='Use this model file. For tracking this overrides the latest model file. For training this will be used for initialization', default=None, nargs='*')
    parser.add_argument('-model_files2', dest='model_file2', help='Use this model file for second stage. For tracking this overrides the latest model file. For training this will be used for initialization', default=None, nargs='*')
    parser.add_argument('-cache', dest='cache', help='Override cachedir in lbl file', default=None)
    parser.add_argument('-debug', dest='debug', help='Print debug messages', action='store_true')
    parser.add_argument('-no_except', dest='no_except', help='Dont catch exception. Useful for debugging',
                        action='store_true')
    parser.add_argument('-train_name', dest='train_name', help='Training name', default='deepnet')
    parser.add_argument('-err_file', dest='err_file', help='Err file', default=None)
    parser.add_argument('-log_file', dest='log_file', help='Err file', default=None)
    parser.add_argument('-conf_params', dest='conf_params',
                        help='conf params. These will override params from lbl file', default=None, nargs='*')
    parser.add_argument('-conf_params2', dest='conf_params2',
                        help='conf params for 2nd stage. These will override params from lbl file', default=None, nargs='*')
    parser.add_argument('-type', dest='type', help='Network type', default=None)
    parser.add_argument('-type2', dest='type2', help='Network type for second stage', default=None)
    parser.add_argument('-stage', dest='stage', help='Stage for multi-stage tracking. Options are multi, first, second or None (default)', default=None)
    subparsers = parser.add_subparsers(help='train or track or gt_classify', dest='sub_name')

    parser_train = subparsers.add_parser('train', help='Train the detector')
    parser_train.add_argument('-skip_db', dest='skip_db', help='Skip creating the data base', action='store_true')
    parser_train.add_argument('-use_defaults', dest='use_defaults', action='store_true',
                              help='Use default settings of openpose, deeplabcut or leap')
    parser_train.add_argument('-use_cache', dest='use_cache', action='store_true', help='Use cached images in the label file to generate the training data.')
    parser_train.add_argument('-continue', dest='restore', action='store_true',
                              help='Continue from previously unfinished traning. Only for unet')
    parser_train.add_argument('-split_file', dest='split_file',
                              help='Split file to split data for train and validation', default=None)
    # parser_train.add_argument('-classify_val', dest='classify_val',
    #                           help='Apply trained model to val db', action='store_true')
    # parser_train.add_argument('-classify_val_out', dest='classify_val_out',
    #                           help='Store results of classify_val in this file (specified as a full path).',
    #                           default=None)

    # parser_train.add_argument('-cache',dest='cache_dir',
    #                           help='cache dir for training')

    parser_classify = subparsers.add_parser('track', help='Track a movie')
    parser_classify.add_argument("-mov", dest="mov",
                                 help="movie(s) to track",
                                 nargs='+')  # KB 20190123 removed required because list_file does not require mov
    parser_classify.add_argument("-trx", dest="trx",
                                 help='trx file for above movie', default=None, nargs='*')
    parser_classify.add_argument('-start_frame', dest='start_frame', help='start tracking from this frame', nargs='*', type=int, default=1)
    parser_classify.add_argument('-end_frame', dest='end_frame', help='end frame for tracking', nargs='*', type=int, default=-1)
    parser_classify.add_argument('-skip_rate', dest='skip', help='frames to skip while tracking', default=1, type=int)
    parser_classify.add_argument('-out', dest='out_files', help='file to save tracking results to', required=True, nargs='+')
    parser_classify.add_argument('-trx_ids', dest='trx_ids', help='only track these animals', nargs='*', type=int, default=[], action='append')
    parser_classify.add_argument('-hmaps', dest='hmaps', help='generate heatmpas', action='store_true')
    parser_classify.add_argument('-crop_loc', dest='crop_loc', help='crop location given xlo xhi ylo yhi', nargs='*', type=int, default=None)
    parser_classify.add_argument('-list_file', dest='list_file', help='JSON file with list of movies, targets and frames to track', default=None)
    parser_classify.add_argument('-use_cache', dest='use_cache', action='store_true', help='Use cached images in the label file to generate the database for list file.')


    parser_gt = subparsers.add_parser('gt_classify', help='Classify GT labeled frames')
    parser_gt.add_argument('-out', dest='out_files', help='Mat file (full path with .mat extension) where GT output will be saved', nargs='+', required=True)

    parser_aug = subparsers.add_parser('data_aug', help='get the augmented images')
    parser_aug.add_argument('-no_aug', dest='no_aug', help='dont augment the images. Return the original images', default=False)
    parser_aug.add_argument('-out', dest='out_files', help='Destination to save the images', required=True)
    parser_aug.add_argument('-use_cache', dest='use_cache', action='store_true', help='Use cached images in the label file to generate the augmented images')
    parser_aug.add_argument('-nsamples', dest='nsamples', default=None, help='Number of examples to be generated', type=int)


    parser_db = subparsers.add_parser('classify', help='Classify validation data. Not supported completely')
    parser_db.add_argument('-out', dest='out_files', help='Destination to save the output', required=True)
    parser_db.add_argument('-db_file', dest='db_file', help='Validation data set to classify', default=None)

    parser_model = subparsers.add_parser('model_files', help='prints the list of model files')

    parser_test = subparsers.add_parser('test', help='Perform tests')
    parser_test.add_argument('testrun', choices=['hello'], help="Test to run")

    print(argv)
    args = parser.parse_args(argv)
    if args.view is not None:
        args.view = convert(args.view, to_python=True)
    if args.sub_name == 'track' and args.mov is not None:
        nmov = len(args.mov)
        args.trx_ids = parse_trx_ids_arg(args.trx_ids, nmov)
        if type(args.start_frame) == list:
            assert all([a > 0 for a in args.start_frame]), 'Start frames must be positive integers'
        else:
            assert args.start_frame > 0, 'Start frames must be positive integers'
        args.start_frame = to_py(args.start_frame)
        args.start_frame = parse_frame_arg(args.start_frame, nmov, 0)
        args.end_frame = parse_frame_arg(args.end_frame, nmov, np.Inf)

        # logging.info('trx_ids = ' + str(args.trx_ids))
        # logging.info('start_frame = ' + str(args.start_frame))
        # logging.info('end_frame = ' + str(args.start_frame))

        args.crop_loc = to_py(args.crop_loc)

    if args.sub_name != 'test':
        net_type = get_net_type(args.lbl_file)
        # command line has precedence over the one in label file.
        if args.type is None and net_type is not None:
            # AL20190719: don't understand this, in this branch the net_type was found in the lbl file?
            # Shouldn't we be using/assigning to net_type here.
            logging.info("No network type specified on command line or in the lbl file. Selecting MDN")
            args.type = 'mdn'
    return args


def parse_trx_ids_arg(trx_ids, nmovies):
    logging.info('PARSE_TRX_IDS: Input nmovies = %d, trx_ids = ' % nmovies + str(trx_ids))
    # make trx_ids a list of lists, with one element for each movie
    if trx_ids == []:  # not input
        trx_ids_out = [[]] * nmovies
    elif isinstance(trx_ids, int):  # one id
        trx_ids_out = [[to_py(trx_ids)]] * nmovies
    elif isinstance(trx_ids, list):
        if isinstance(trx_ids[0], int):
            trx_ids_out = [to_py(trx_ids)] * nmovies
        else:  # should be a list
            trx_ids_out = list(map(to_py, trx_ids))
            if len(trx_ids_out) < nmovies:
                trx_ids_out = trx_ids_out + [] * (nmovies - len(trx_ids_out))
    else:
        raise ValueError('Type for trx_ids is not handled')

    logging.info('Output trx_ids = ' + str(trx_ids_out))
    return trx_ids_out


def get_valfilename(conf, nettype):
    if nettype in ['mdn', 'unet', 'openpose']:
        val_filename = conf.valfilename + '.tfrecords'
    elif nettype == 'leap':
        val_filename = 'leap_val.h5'
    elif nettype == 'deeplabcut':
        val_filename = 'val_data.p'
    else:
        raise ValueError('Unrecognized net type')
    return val_filename

def track_multi_stage(args,arg_ndx,view,crop_loc):
    name = args.name
    lbl_file = args.lbl_file
    if args.stage == 'multi':
        out_files = args.out_files
        args.out_files = args.trx
        track_view_mov(lbl_file, view, arg_ndx, name, args, first_stage=True, crop_loc=crop_loc)
        args.out_files = out_files
        args.type = args.type2
        args.conf_params = args.conf_params2
        track_view_mov(lbl_file, view, arg_ndx, name, args, second_stage=True, crop_loc=crop_loc)

    elif args.stage == 'first':
        track_view_mov(lbl_file, view, arg_ndx, name, args, first_stage=True, crop_loc=crop_loc)
    elif args.stage == 'second':
        track_view_mov(lbl_file, view, arg_ndx, name, args, second_stage=True, crop_loc=crop_loc)
    else:
        track_view_mov(lbl_file, view, arg_ndx, name, args, crop_loc=crop_loc)


def track_view_mov(lbl_file,view_ndx,arg_ndx,name,args,crop_loc,first_stage=False,second_stage=False):
    conf = create_conf(lbl_file, view_ndx,name, net_type=args.type, cache_dir=args.cache, conf_params=args.conf_params,first_stage=first_stage,second_stage=second_stage)

    classify_movie_all(args.type,
                       conf=conf,
                       mov_file=args.mov[arg_ndx],
                       trx_file=args.trx[arg_ndx],
                       out_file=args.out_files[arg_ndx],
                       start_frame=args.start_frame[arg_ndx],
                       end_frame=args.end_frame[arg_ndx],
                       skip_rate=args.skip,
                       trx_ids=args.trx_ids[arg_ndx],
                       name=name,
                       save_hmaps=args.hmaps,
                       crop_loc=crop_loc[arg_ndx],
                       model_file=args.model_file[arg_ndx],
                       train_name=args.train_name
                       )


def check_args(args,nviews):
    # Does a check on on arguments and converts the arguments to appropriate format
    view = args.view
    if view is None:
        views = range(nviews)
    else:
        views = [view]
    nviews = len(views)

    if args.sub_name == 'train':
        if args.model_file is None:
            args.model_file = [None]*nviews
        else:
            assert len(args.model_file) == nviews, 'Number of model files should be same as number of views for training'
        if args.model_file2 is None:
            args.model_file2 = [None]*nviews
        else:
            assert len(args.model_file2) == nviews, 'Number of model files should be same as number of views for training'

    elif args.sub_name == 'track' and args.list_file is not None:
        # KB 20190123: added list_file input option
        assert args.mov is None, 'Input list_file should specify movie files'
        assert args.trx is None, 'Input list_file should specify trx files'
        assert args.crop_loc is None, 'Input list_file should specify crop locations'

        assert len(args.out_files) == nviews, 'Number of out files should be same as number of views to track (%d)' % nviews
        if args.model_file is None:
            args.model_file = [None] * nviews
        else:
            assert len(args.model_file) == nviews, 'Number of model files should be same as the number of views to track (%d)' % nviews
        if args.model_file2 is None:
            args.model_file2 = [None] * nviews
        else:
            assert len(args.model_file2) == nviews, 'Number of model files should be same as the number of views to track (%d)' % nviews

    elif args.sub_name == 'track':

        if args.view is None:
            assert len(args.mov) == nviews, 'Number of movie files should be same number of views'
            assert len(args.out_files) == nviews, 'Number of out files should be same as number of views'
            if args.trx is None:
                args.trx = [None] * nviews
            else:
                assert len(args.trx) == nviews, 'Number of movie files should be same as the number of trx files'
            if args.model_file is None:
                args.model_file = [None] * nviews
            else:
                assert len(args.model_file) == nviews, 'Number of movie files should be same as the number of trx files'
            if args.model_file2 is None:
                args.model_file2 = [None] * nviews
            else:
                assert len(args.model_file2) == nviews, 'Number of movie files should be same as the number of trx files'
            if args.crop_loc is not None:
                assert len(args.crop_loc) == 4 * nviews, 'cropping location should be specified as xlo xhi ylo yhi for all the views'
        else:
            nmov = len(args.mov)

            def checklen(x, varstr):
                assert len(x) == nmov, "Number of {} ({}) must match number of movies ({})".format(varstr, len(x), nmov)

            if args.trx is None:
                args.trx = [None] * nmov
            else:
                checklen(args.trx, 'trx')
            if args.model_file is None:
                args.model_file = [None] * nmov
            elif len(args.model_file) == 1:
                args.model_file = args.model_file * nmov
            else:
                checklen(args.model_file, 'model files')
            checklen(args.out_files, 'output files')


    elif args.sub_name == 'gt_classify':

        if args.model_file is None:
            args.model_file = [None] * len(views)
        else:
            assert len(args.model_file) == len(views), 'Number of model files specified must match number of views to be processed'
        if args.model_file2 is None:
            args.model_file2 = [None] * len(views)
        else:
            assert len(args.model_file2) == len(views), 'Number of model files specified must match number of views to be processed'

        assert args.out_files is not None

        assert len(args.out_files) == len(views), 'Number of gt output files must match number of views to be processed'

    elif args.sub_name == 'classify':
        if args.view is None:
            if args.model_file is None:
                args.model_file = [None] * nviews
            else:
                assert len(args.model_file) == nviews, 'Number of movie files should be same as the number of trx files'
            if args.model_file2 is None:
                args.model_file2 = [None] * nviews
            else:
                assert len(
                    args.model_file2) == nviews, 'Number of movie files should be same as the number of trx files'
        else:
            if args.model_file is None:
                args.model_file = [None]* nviews
            if args.model_file2 is None:
                args.model_file2 = [None]* nviews



def run(args):
    name = args.name

    lbl_file = args.lbl_file
    try:
        try:
            H = loadmat(lbl_file)
        except NotImplementedError:
            logging.info('Label file is in v7.3 format. Loading using h5py')
            H = h5py.File(lbl_file, 'r')
    except TypeError as e:
        logging.exception('LBL_READ: Could not read the lbl file {}'.format(lbl_file))
        exit(1)

    # raise ValueError('I am an error')

    nviews = int(read_entry(H['cfg']['NumViews']))
    view = args.view
    if view is None:
        views = range(nviews)
    else:
        views = [view]
    nviews = len(views)
    check_args(args,nviews)

    if args.sub_name == 'train':
        train_multi_stage(args,nviews)

    elif args.sub_name == 'track' and args.list_file is not None:
        # KB 20190123: added list_file input option

        for view_ndx, view in enumerate(views):
            # conf = create_conf(lbl_file, view, name, net_type=args.type,cache_dir=args.cache, conf_params=args.conf_params)
            # maybe this should be in create_conf
            # conf.batch_size = 1 if args.type == 'deeplabcut' else conf.batch_size
            success, pred_locs = classify_list_file(args,view=view,view_ndx=view_ndx)
            assert success, 'Error classifying list_file ' + args.list_file + 'view ' + str(view)

    elif args.sub_name == 'track':

        if args.view is None:
            for view_ndx, view in enumerate(views):
                if args.crop_loc is not None:
                    crop_loc = [int(x) for x in args.crop_loc]
                    # crop_loc = np.array(crop_loc).reshape([len(views), 4])[view_ndx, :] - 1
                    # KB 20190123: crop_loc was being decremented twice, removed one
                    crop_loc = np.array(crop_loc).reshape([len(views), 4])
                else:
                    crop_loc = [None]*nviews
                track_multi_stage(args,view_ndx,view,crop_loc)
                # conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache,conf_params=args.conf_params)
                # if args.crop_loc is not None:
                #     crop_loc = [int(x) for x in args.crop_loc]
                #     # crop_loc = np.array(crop_loc).reshape([len(views), 4])[view_ndx, :] - 1
                #     # KB 20190123: crop_loc was being decremented twice, removed one
                #     crop_loc = np.array(crop_loc).reshape([len(views), 4])[view_ndx, :]
                # else:
                #     crop_loc = None
                #
                # classify_movie_all(args.type,
                #                    conf=conf,
                #                    mov_file=args.mov[view_ndx],
                #                    trx_file=args.trx[view_ndx],
                #                    out_file=args.out_files[view_ndx],
                #                    start_frame=args.start_frame[view_ndx],
                #                    end_frame=args.end_frame[view_ndx],
                #                    skip_rate=args.skip,
                #                    trx_ids=args.trx_ids[view_ndx],
                #                    name=name,
                #                    save_hmaps=args.hmaps,
                #                    crop_loc=crop_loc,
                #                    model_file=args.model_file[view_ndx],
                #                    train_name=args.train_name
                #                    )
        else:
            nmov = len(args.mov)
            if args.crop_loc is None:
                crop_loc_list = [None] * nmov
            else:
                assert len(args.crop_loc) == 4 * nmov, 'cropping location should be specified (for given view) as xlo1 xhi1 ylo1 yhi1 xlo2 xhi2 ...'
                args.crop_loc = [int(x) for x in args.crop_loc]
                crop_loc_list = []
                for imov in range(nmov):
                    croparr = np.array(args.crop_loc[imov * 4:(imov + 1) * 4])
                    crop_loc_list.append(croparr)

            # AL for now require single view spec to not break bkwds
            # args.view

            for ndx in range(nmov):
                track_multi_stage(args,ndx,args.view,crop_loc_list)
                # conf = create_conf(lbl_file, args.view, name,
                #                    net_type=args.type,
                #                    cache_dir=args.cache,
                #                    conf_params=args.conf_params)
                # logging.info('Classifying movie {}: {}'.format(ndx, args.mov[ndx]))
                # classify_movie_all(args.type,
                #                    conf=conf,
                #                    mov_file=args.mov[ndx],
                #                    trx_file=args.trx[ndx],
                #                    out_file=args.out_files[ndx],
                #                    start_frame=args.start_frame[ndx],
                #                    end_frame=args.end_frame[ndx],
                #                    skip_rate=args.skip,
                #                    trx_ids=args.trx_ids[ndx],
                #                    name=name,
                #                    save_hmaps=args.hmaps,
                #                    crop_loc=crop_loc_list[ndx],
                #                    model_file=args.model_file[ndx],
                #                    train_name=args.train_name
                #                    )

    elif args.sub_name == 'gt_classify':

        for view_ndx, view in enumerate(views):
            classify_gt_data(args, view,view_ndx)

    elif args.sub_name == 'data_aug':

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache,conf_params=args.conf_params)
            out_file = args.out_files + '_{}.mat'.format(view)
            distort = not args.no_aug
            get_augmented_images(conf, out_file, distort, nsamples=args.nsamples)

    elif args.sub_name == 'classify':

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache,conf_params=args.conf_params)
            out_file = args.out_files + '_{}.mat'.format(view)
            if args.db_file is not None:
                db_file = args.db_file
            else:
                val_filename = get_valfilename(conf, args.type)
                db_file = os.path.join(conf.cachedir, val_filename)
            preds, locs, info, _ = classify_db_all(args.type, conf, db_file, model_file=args.model_file[view_ndx])
            # A = convert_to_orig_list(conf,preds,locs, info)
            info = to_mat(info)
            preds = to_mat(preds)
            locs = to_mat(locs)
            # preds, locs = to_mat(A)
            hdf5storage.savemat(out_file, {'pred_locs': preds, 'labeled_locs': locs, 'list': info}, appendmat=False,
                                truncate_existing=True)

    elif args.sub_name == 'model_files':
        m_files = []
        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache, conf_params=args.conf_params)
            m_files.append(get_latest_model_files(conf, net_type=args.type, name=args.train_name))
        print(m_files)


def main(argv):
    args = parse_args(argv)

    if args.sub_name == 'test':
        print("Hello this is APT!")
        return

    log_formatter = logging.Formatter('%(asctime)s %(pathname)s %(funcName)s [%(levelname)-5.5s] %(message)s')

    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    # add err logging
    if args.err_file is None:
        err_file = os.path.join(expanduser("~"), '{}.err'.format(args.name))
    else:
        err_file = args.err_file
    errh = logging.FileHandler(err_file, 'w')
    errh.setLevel(logging.ERROR)
    errh.setFormatter(log_formatter)

    if args.log_file is None:
        # output to console if no log file is specified
        logh = logging.StreamHandler()
    else:
        logh = logging.FileHandler(args.log_file, 'w')

    if args.debug:
        logh.setLevel(logging.DEBUG)
    else:
        logh.setLevel(logging.INFO)
    logh.setFormatter(log_formatter)

    log.addHandler(errh)
    log.addHandler(logh)
    log.setLevel(logging.DEBUG)

    repo_info = PoseTools.get_git_commit()
    logging.info('Git Commit: {}'.format(repo_info))
    logging.info('Args: {}'.format(argv))

    if args.no_except:
        run(args)
    else:
        try:
            run(args)
        except Exception as e:
            logging.exception('UNKNOWN: APT_interface errored')


if __name__ == "__main__":
    main(sys.argv[1:])

# Legacy Code
#
# def train_unet_nodataset(conf, args, restore):
#     ''' O'''
#     if not args.skip_db:
#         create_tfrecord(conf, False, use_cache=args.use_cache)
#     tf.reset_default_graph()
#     self = PoseUNet.PoseUNet(conf, name='deepnet')
#     self.train_data_name = 'traindata'
#     self.train_unet(restore=restore, train_type=1)
