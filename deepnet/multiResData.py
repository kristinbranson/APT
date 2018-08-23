from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import chr
from builtins import range
import scipy.io as sio
import os
import sys
import myutils
import re
import numpy as np
import cv2
from cvc import cvc
import math
from random import randint, sample
import pickle
import h5py
import errno
import PoseTools
import tensorflow as tf
import movies
import json


def find_local_dirs(conf, on_gt=False):
    lbl = h5py.File(conf.labelfile, 'r')
    if on_gt:
        exp_list = lbl['movieFilesAllGT'][conf.view,:]
    else:
        exp_list = lbl['movieFilesAll'][conf.view,:]
    local_dirs = [u''.join(chr(c) for c in lbl[jj]) for jj in exp_list]
    # local_dirs = [u''.join(chr(c) for c in lbl[jj]) for jj in conf.getexplist(lbl)]
    sel_dirs = [True] * len(local_dirs)
    try:
        for k in lbl['projMacros'].keys():
            r_dir = u''.join(chr(c) for c in lbl['projMacros'][k])
            local_dirs = [s.replace('${}'.format(k), r_dir) for s in local_dirs]
    except:
        pass
    lbl.close()
    return local_dirs, sel_dirs


def find_gt_dirs(conf):
    lbl = h5py.File(conf.labelfile, 'r')
    exp_list = lbl['movieFilesAllGT'][conf.view,:]
    local_dirs = [u''.join(chr(c) for c in lbl[jj]) for jj in exp_list]
    sel_dirs = [True] * len(local_dirs)
    try:
        for k in lbl['projMacros'].keys():
            r_dir = u''.join(chr(c) for c in lbl['projMacros'][k])
            local_dirs = [s.replace('${}'.format(k), r_dir) for s in local_dirs]
    except:
        pass
    lbl.close()
    return local_dirs, sel_dirs


def get_trx_files(lbl, local_dirs, on_gt=False):
    if on_gt:
        trx_files = [u''.join(chr(c) for c in lbl[jj]) for jj in lbl['trxFilesAllGT'][0]]
    else:
        trx_files = [u''.join(chr(c) for c in lbl[jj]) for jj in lbl['trxFilesAll'][0]]
    movdir = [os.path.dirname(a) for a in local_dirs]
    trx_files = [s.replace('$movdir', m) for (s, m) in zip(trx_files, movdir)]
    try:
        for k in lbl['projMacros'].keys():
            r_dir = u''.join(chr(c) for c in lbl['projMacros'][k])
            trx_files = [s.replace('${}'.format(k), r_dir) for s in trx_files]
    except:
        pass
    return trx_files


def create_val_data(conf, force=False):
    outfile = os.path.join(conf.cachedir, conf.valdatafilename)
    if ~force & os.path.isfile(outfile):
        return

    print('Creating val data %s!' % outfile)
    localdirs, seldirs = find_local_dirs(conf)
    nexps = len(seldirs)
    isval = []
    if (not hasattr(conf, 'splitType')) or (conf.splitType is 'exp'):
        isval = sample(list(range(nexps)), int(nexps * conf.valratio))

    try:
        os.makedirs(conf.cachedir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    with open(outfile, 'w') as f:
        pickle.dump([isval, localdirs, seldirs], f)


def load_val_data(conf):
    outfile = os.path.join(conf.cachedir, conf.valdatafilename)
    assert os.path.isfile(outfile), "valdatafile {} doesn't exist".format(outfile)

    with open(outfile, 'rb') as f:
        if sys.version_info.major == 3:
            is_val, local_dirs, sel_dirs = pickle.load(f, encoding='latin1')
        else:
            is_val, local_dirs, sel_dirs = pickle.load(f)
    return is_val, local_dirs, sel_dirs


def get_movie_lists(conf):
    is_val, local_dirs, sel_dirs = load_val_data(conf)
    trainexps = []
    valexps = []
    for ndx in range(len(local_dirs)):
        if not sel_dirs[ndx]:
            continue
        if is_val.count(ndx):
            valexps.append(local_dirs[ndx])
        else:
            trainexps.append(local_dirs[ndx])

    return trainexps, valexps


def create_id(exp_name, cur_loc, f_num, im_sz):
    for x in cur_loc:
        assert x[0] >= 0, "x value %d is less than 0" % x[0]
        assert x[1] >= 0, "y value %d is less than 0" % x[1]
        assert x[0] < im_sz[1], "x value %d is greater than imsz %d" % (x[0], im_sz[1])
        assert x[1] < im_sz[0], "y value %d is greater than imsz %d" % (x[1], im_sz[0])

    x_str = '_'.join([str(x[0]) for x in cur_loc])
    y_str = '_'.join([str(x[1]) for x in cur_loc])

    str_id = '{:08d}:{}:x{}:y{}:t{:d}'.format(randint(0, 1e8),
                                              exp_name, x_str, y_str, f_num)
    return str_id


def decode_id(key_str):
    vv = re.findall('(\d+):(.*):x(.*):y(.*):t(\d+)', key_str)[0]
    x_locs = [int(x) for x in vv[2].split('_')]
    y_locs = [int(x) for x in vv[3].split('_')]
    locs = list(zip(x_locs, y_locs))
    return vv[1], locs, int(vv[4])


def sanitize_locs(locs):
    n_locs = np.array(locs).astype('float')
    n_locs[n_locs < 0] = np.nan
    return n_locs


def int64_feature(value):
    if not isinstance(value, (list, np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    if not isinstance(value, (list, np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_record(conf):
    lbl = h5py.File(conf.labelfile, 'r')
    pts = np.array(lbl['pts'])
    ts = np.array(lbl['ts']).squeeze().astype('int')
    expid = np.array(lbl['expidx']).squeeze().astype('int')
    view = conf.view
    count = 0
    valcount = 0

    create_val_data(conf)
    isval, localdirs, seldirs = load_val_data(conf)

    trainfilename = os.path.join(conf.cachedir, conf.trainfilename)
    valfilename = os.path.join(conf.cachedir, conf.valfilename)

    env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
    valenv = tf.python_io.TFRecordWriter(valfilename + '.tfrecords')

    for ndx, dirname in enumerate(localdirs):
        if not seldirs[ndx]:
            continue

        expname = conf.getexpname(dirname)
        frames = np.where(expid == (ndx + 1))[0]
        cap = cv2.VideoCapture(localdirs[ndx])

        curenv = valenv if isval.count(ndx) else env

        for curl in frames:

            fnum = ts[curl]
            if fnum > cap.get(cvc.FRAME_COUNT):
                if fnum > cap.get(cvc.FRAME_COUNT) + 1:
                    raise ValueError('Accessing frames beyond ' +
                                     'the length of the video for' +
                                     ' {} expid {:d} '.format(expname, ndx) +
                                     ' at t {:d}'.format(fnum)
                                     )
                continue
            framein = myutils.readframe(cap, fnum - 1)
            cloc = conf.cropLoc[tuple(framein.shape[0:2])]
            framein = PoseTools.crop_images(framein, conf)
            framein = framein[:, :, 0:1]

            curloc = np.round(pts[curl, :, view, :]).astype('int')
            curloc[:, 0] = curloc[:, 0] - cloc[1]  # ugh, the nasty x-y business.
            curloc[:, 1] = curloc[:, 1] - cloc[0]
            curloc = curloc.clip(min=1)

            rows = framein.shape[0]
            cols = framein.shape[1]
            if np.ndim(framein) > 2:
                depth = framein.shape[2]
            else:
                depth = 1

            image_raw = framein.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': int64_feature(rows),
                'width': int64_feature(cols),
                'depth': int64_feature(depth),
                'locs': float_feature(curloc.flatten()),
                'expndx': float_feature(ndx),
                'ts': float_feature(curl),
                'image_raw': bytes_feature(image_raw)}))
            curenv.write(example.SerializeToString())

            if isval.count(ndx):
                valcount += 1
            else:
                count += 1

        cap.release()  # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx, len(localdirs), count, valcount))
    env.close()  # close the database
    valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' % (count, valcount))


def create_full_tf_record(conf):
    lbl = h5py.File(conf.labelfile, 'r')
    pts = np.array(lbl['pts'])
    ts = np.array(lbl['ts']).squeeze().astype('int')
    exp_id = np.array(lbl['expidx']).squeeze().astype('int')
    view = conf.view
    count = 0
    val_count = 0

    create_val_data(conf)
    is_val, local_dirs, sel_dirs = load_val_data(conf)

    train_filename = os.path.join(conf.cachedir, conf.fulltrainfilename)

    env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')

    for ndx, dirname in enumerate(local_dirs):
        if not sel_dirs[ndx]:
            continue

        exp_name = conf.getexpname(dirname)
        frames = np.where(exp_id == (ndx + 1))[0]
        cap = cv2.VideoCapture(local_dirs[ndx])

        cur_env = env

        for curl in frames:

            fnum = ts[curl]
            if fnum > cap.get(cvc.FRAME_COUNT):
                if fnum > cap.get(cvc.FRAME_COUNT) + 1:
                    raise ValueError('Accessing frames beyond ' +
                                     'the length of the video for' +
                                     ' {} expid {:d} '.format(exp_name, ndx) +
                                     ' at t {:d}'.format(fnum)
                                     )
                continue
            frame_in = myutils.readframe(cap, fnum - 1)
            c_loc = conf.cropLoc[tuple(frame_in.shape[0:2])]
            frame_in = PoseTools.crop_images(frame_in, conf)
            frame_in = frame_in[:, :, 0:1]

            cur_loc = np.round(pts[curl, :, view, :]).astype('int')
            cur_loc[:, 0] = cur_loc[:, 0] - c_loc[1]  # ugh, the nasty x-y business.
            cur_loc[:, 1] = cur_loc[:, 1] - c_loc[0]
            cur_loc = cur_loc.clip(min=0.1)

            rows = frame_in.shape[0]
            cols = frame_in.shape[1]
            if np.ndim(frame_in) > 2:
                depth = frame_in.shape[2]
            else:
                depth = 1

            image_raw = frame_in.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': int64_feature(rows),
                'width': int64_feature(cols),
                'depth': int64_feature(depth),
                'locs': float_feature(cur_loc.flatten()),
                'expndx': float_feature(ndx),
                'ts': float_feature(curl),
                'image_raw': bytes_feature(image_raw)}))
            cur_env.write(example.SerializeToString())

            count += 1

        cap.release()  # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx, len(local_dirs), count, val_count))
    env.close()  # close the database
    print('%d,%d number of pos examples added to the db and val-db' % (count, val_count))

def get_labeled_frames(lbl,ndx ,trx_ndx=None, on_gt=False):
    cur_pts = trx_pts(lbl, ndx, on_gt)
    if cur_pts.ndim == 4:
        frames = np.where(np.invert(np.all(np.isnan(cur_pts[trx_ndx, :, :, :]), axis=(1, 2))))[0]
    else:
        frames = np.where(np.invert(np.all(np.isnan(cur_pts[:, :, :]), axis=(1, 2))))[0]

    return frames


def create_tf_record_from_lbl(conf, split=True, split_file=None):
    lbl = h5py.File(conf.labelfile, 'r')

    create_val_data(conf,True)
    is_val, local_dirs, sel_dirs = load_val_data(conf)

    env, val_env = create_envs(conf, split)
    view = conf.view
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    sel_pts = int(view * npts_per_view) + conf.selpts

    splits = [[],[]]
    count = 0
    val_count = 0

    if conf.splitType is 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = PoseTools.json_load(split_file)
    else:
        predefined = None

    for ndx, dir_name in enumerate(local_dirs):
        if not sel_dirs[ndx]:
            continue

        exp_name = conf.getexpname(dir_name)
        cur_pts = trx_pts(lbl, ndx)
        frames = get_labeled_frames(lbl, ndx, None)
        cap = movies.Movie(local_dirs[ndx])

        for fnum in frames:

            if not check_fnum(fnum, cap, exp_name, ndx):
                continue

            cur_env = get_cur_env([env, val_env], split, conf, [ndx, fnum, 0], is_val, trx_split=None, predefined=predefined)
            frame_in, cur_loc = get_patch(cap, fnum, conf, cur_pts[fnum,:,sel_pts])

            rows = frame_in.shape[0]
            cols = frame_in.shape[1]
            depth = frame_in.shape[2] if frame_in.ndim > 2 else 1

            image_raw = frame_in.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': int64_feature(rows),
                'width': int64_feature(cols),
                'depth': int64_feature(depth),
                'trx_ndx': int64_feature(0),
                'locs': float_feature(cur_loc.flatten()),
                'expndx': float_feature(ndx),
                'ts': float_feature(fnum),
                'image_raw': bytes_feature(image_raw)}))
            cur_env.write(example.SerializeToString())

            if cur_env is val_env and split:
                val_count += 1
                splits[1].append([ndx, fnum, 0])
            else:
                count += 1
                splits[0].append([ndx, fnum, 0])

        cap.close()  # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx + 1, len(local_dirs), count, val_count))
    env.close()  # close the database
    if split:
        val_env.close()
    print('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
    with open(os.path.join(conf.cachedir,'splitdata.json'),'w') as f:
        json.dump(splits, f)


def get_patch(cap, fnum, conf, locs, offset=0, stationary=True, cur_trx=None, flipud=False, crop_loc=None):
    # fnum is the frame number
    # cur_trx == None indicates that the project doesnt have trx file.
    # offset is used for multiframe
    # stationary is also used for multiframe.
    # crop_loc is the cropping location.

    if cur_trx is not None: # when there are trx
        return get_patch_trx(cap, cur_trx, fnum, conf, locs, offset, stationary,flipud)
    else:
        frame_in, _, _, _ = read_frame(cap,fnum,cur_trx,flipud=flipud)
        frame_in = frame_in[:,:,0:conf.imgDim]
        if crop_loc is not None:
            xlo, xhi, ylo, yhi = crop_loc
            xhi += 1; yhi += 1
        else:
            xlo = 0; ylo = 0
            yhi, xhi = frame_in.shape[0:2]

            # convert grayscale to color if the conf says so.
        #c_loc = conf.cropLoc[tuple(frame_in.shape[0:2])]
        #frame_in = PoseTools.crop_images(frame_in, conf)
        frame_in = frame_in[ylo:yhi,xlo:xhi,:]
        cur_loc = locs.copy()
        cur_loc[:, 0] = cur_loc[:, 0] - xlo   # ugh, the nasty x-y business.
        cur_loc[:, 1] = cur_loc[:, 1] - ylo
        # -1 because matlab is 1-indexed
        cur_loc = cur_loc.clip(min=0, max=[(xhi-xlo) + 7, (yhi-ylo) + 7])
        return  frame_in, cur_loc


def create_envs(conf, split, db_type=None):
    if db_type is 'rnn':
        if split:
            train_filename = os.path.join(conf.cachedir, conf.trainfilename_rnn)
            val_filename = os.path.join(conf.cachedir, conf.valfilename_rnn)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = tf.python_io.TFRecordWriter(val_filename + '.tfrecords')
        else:
            train_filename = os.path.join(conf.cachedir, conf.trainfilename_rnn)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = None
        return env, val_env
    elif db_type is not None:
        if split:
            train_filename = os.path.join(conf.cachedir, conf.trainfilename + '_' + db_type)
            val_filename = os.path.join(conf.cachedir, conf.valfilename + '_' + db_type)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = tf.python_io.TFRecordWriter(val_filename + '.tfrecords')
        else:
            train_filename = os.path.join(conf.cachedir, conf.trainfilename + '_' + db_type)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = None
        return env, val_env
    else:
        if split:
            train_filename = os.path.join(conf.cachedir, conf.trainfilename)
            val_filename = os.path.join(conf.cachedir, conf.valfilename)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = tf.python_io.TFRecordWriter(val_filename + '.tfrecords')
        else:
            train_filename = os.path.join(conf.cachedir, conf.trainfilename)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = None
        return env, val_env


def trx_pts(lbl, ndx, on_gt = False):
    # new styled sparse labeledpos
    if on_gt:
        pts = np.array(lbl['labeledposGT'])
    else:
        pts = np.array(lbl['labeledpos'])
    try:
        sz = np.array(lbl[pts[0, ndx]]['size'])[:, 0].astype('int')
        cur_pts = np.zeros(sz).flatten()
        cur_pts[:] = np.nan
        if lbl[pts[0,ndx]]['val'].value.ndim > 1:
            idx = np.array(lbl[pts[0, ndx]]['idx'])[0, :].astype('int') - 1
            val = np.array(lbl[pts[0, ndx]]['val'])[0, :] - 1
            cur_pts[idx] = val
        cur_pts = cur_pts.reshape(np.flipud(sz))
    except ValueError:
        cur_pts = np.array(lbl[pts[0,ndx]]) - 1
    return cur_pts


def get_cur_env(envs, split, conf, info, mov_split, trx_split, predefined=None):
    env, val_env = envs
    mov_ndx, frame_ndx, trx_ndx = info
    if split:
        if hasattr(conf, 'splitType'):
            if conf.splitType is 'frame':
                cur_env = val_env if np.random.random() < conf.valratio \
                    else env
            elif conf.splitType is 'trx':
                cur_env = val_env if trx_split[trx_ndx] else env
            elif conf.splitType is 'predefined':
                cur_env = val_env if predefined[1].count(info) > 0 else env
            else:
                cur_env = val_env if mov_split.count(mov_ndx) else env
        else:
            cur_env = val_env if mov_split.count(mov_ndx) and split else env
    else:
        cur_env = env
    return cur_env


def check_fnum(fnum, cap, expname, ndx):
    if fnum > cap.get_n_frames():  # get(cvc.FRAME_COUNT):
        if fnum > cap.get_n_frames() + 1:  # get(cvc.FRAME_COUNT) + 1:
            raise ValueError('Accessing frames beyond ' +
                             'the length of the video for' +
                             ' {} expid {:d} '.format(expname, ndx) +
                             ' at t {:d}'.format(fnum)
                             )
        return False
    else:
        return True


def read_trx(cur_trx, fnum):
    if cur_trx is None:
        return None,None,None
    trx_fnum = fnum - cur_trx['firstframe'][0, 0] + 1
    x = int(round(cur_trx['x'][0, trx_fnum])) - 1
    y = int(round(cur_trx['y'][0, trx_fnum])) - 1
    # -1 for 1-indexing in matlab and 0-indexing in python
    theta = cur_trx['theta'][0, trx_fnum]
    return x, y, theta


def read_frame(cap, fnum, cur_trx, offset=0, stationary=True,flipud=False):
    # stationary means that fly will always be in the center of the frame
    if not check_fnum(fnum, cap, 0, 0):
        return None, None, None, None
    o_fnum = fnum + offset

    if cur_trx is not None:
        if o_fnum > cur_trx['endframe'][0, 0] - 1:
            o_fnum = cur_trx['endframe'][0, 0] - 1
        if o_fnum < cur_trx['firstframe'][0, 0] - 1:
            o_fnum = cur_trx['firstframe'][0, 0] - 1
    else:
        o_fnum = 0 if o_fnum < 0 else o_fnum
        o_fnum = cap.get_n_frames()-1 if o_fnum > cap.get_n_frames() else o_fnum

    framein = cap.get_frame(o_fnum)[0]
    if flipud:
        framein = np.flipud(framein)
    if framein.ndim == 2:
        framein = framein[:, :, np.newaxis]

    if stationary:
        x, y, theta = read_trx(cur_trx, o_fnum)
    else:
        x, y, theta = read_trx(cur_trx, fnum)
    return framein, x, y, theta


def get_patch_trx(cap, cur_trx, fnum, conf, locs, offset=0, stationary=True,flipud=False):
    # assert conf.imsz[0] == conf.imsz[1]
    psz = max(conf.imsz)
    im, x, y, theta = read_frame(cap, fnum, cur_trx, offset, stationary,flipud)

    im = im.copy()
    theta = theta + math.pi / 2
    if im.ndim == 2:
        pad_im = np.pad(im, [psz, psz], 'constant')
        patch = pad_im[y:y + 2 * psz, x:x + 2 * psz]
        rot_mat = cv2.getRotationMatrix2D((psz, psz), theta * 180 / math.pi, 1)
        rpatch = cv2.warpAffine(patch, rot_mat, (2 * psz, 2 * psz))
        rpatch = rpatch[psz / 2:-psz / 2, psz / 2:-psz / 2]
    else:
        pad_im = np.pad(im, [[psz, psz], [psz, psz], [0, 0]], 'constant')
        patch = pad_im[y:y + 2 * psz, x:x + 2 * psz, :]
        rot_mat = cv2.getRotationMatrix2D((psz, psz), theta * 180 / math.pi, 1)
        rpatch = cv2.warpAffine(patch, rot_mat, (2 * psz, 2 * psz))
        if rpatch.ndim == 2:
            rpatch = rpatch[:, :, np.newaxis]
        rpatch = rpatch[psz / 2:-psz / 2, psz / 2:-psz / 2, :]
    ll = locs.copy()
    ll = ll - [x, y]
    rot = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    lr = np.dot(ll, rot) + [psz / 2, psz / 2]

    if conf.imsz[0] < conf.imsz[1]:
        extra = (psz-conf.imsz[0])/2
        rpatch = rpatch[extra:-extra,...]
        lr[:,1] -= extra
    elif conf.imsz[1] < conf.imsz[0]:
        extra = (psz-conf.imsz[1])/2
        rpatch = rpatch[:,extra:-extra,...]
        lr[:,0] -= extra

    rpatch = rpatch[:,:,:conf.imgDim]
    return rpatch, lr


def create_tf_record_from_lbl_with_trx(conf, split=True, split_file=None):
    create_val_data(conf)
    is_val, local_dirs, _ = load_val_data(conf)

    lbl = h5py.File(conf.labelfile, 'r')
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    trx_files = get_trx_files(lbl, local_dirs)

    envs = create_envs(conf, split)
    view = conf.view
    count = 0
    val_count = 0
    sel_pts = int(view * npts_per_view) + conf.selpts

    if conf.splitType is 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = json.load(split_file)
    else:
        predefined = None

    splits = [[],[]]
    for ndx, dir_name in enumerate(local_dirs):

        exp_name = conf.getexpname(dir_name)
        trx = sio.loadmat(trx_files[ndx])['trx'][0]
        n_trx = len(trx)

        curpts = trx_pts(lbl, ndx)
        trx_split = np.random.random(n_trx) < conf.valratio
        cap = movies.Movie(local_dirs[ndx])

        for trx_ndx in range(n_trx):
            frames = get_labeled_frames(lbl, ndx, trx_ndx)
            cur_trx = trx[trx_ndx]

            for fnum in frames:
                if not check_fnum(fnum, cap, exp_name, ndx):
                    continue

                cur_env = get_cur_env(envs, split, conf, [ndx, fnum, trx_ndx], is_val,
                                      trx_split, predefined=predefined)
                frame_in, cur_loc = get_patch_trx(cap, cur_trx, fnum, conf, curpts[trx_ndx, fnum, :, sel_pts])

                rows, cols = frame_in.shape[0:2]
                depth = conf.imgDim

                image_raw = frame_in.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': int64_feature(rows),
                    'width': int64_feature(cols),
                    'depth': int64_feature(depth),
                    'trx_ndx': int64_feature(trx_ndx),
                    'locs': float_feature(cur_loc.flatten()),
                    'expndx': float_feature(ndx),
                    'ts': float_feature(fnum),
                    'image_raw': bytes_feature(image_raw)}))
                cur_env.write(example.SerializeToString())

                if cur_env is envs[1]:
                    val_count += 1
                    splits[1].append([ndx,fnum,trx_ndx])
                else:
                    count += 1
                    splits[0].append([ndx,fnum,trx_ndx])

        cap.close()  # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx + 1, len(local_dirs), count, val_count))
    envs[0].close()
    envs[1].close() if split else None
    print('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
    lbl.close()
    with open(os.path.join(conf.cachedir,'splitdata.json'),'w') as f:
        json.dump(splits, f)


def create_tf_record_time_from_lbl_with_trx(conf, split=True, split_file=None):
    create_val_data(conf)
    is_val, local_dirs, _ = load_val_data(conf)

    lbl = h5py.File(conf.labelfile, 'r')
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    trx_files = get_trx_files(lbl, local_dirs)

    env, val_env = create_envs(conf, split)
    view = conf.view
    count = 0
    val_count = 0
    sel_pts = int(view * npts_per_view) + conf.selpts
    tw = conf.time_window_size

    if conf.splitType is 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = json.load(split_file)
    else:
        predefined = None

    splits = [[],[]]
    for ndx, dir_name in enumerate(local_dirs):

        trx = sio.loadmat(trx_files[ndx])['trx'][0]
        n_trx = len(trx)

        cur_pts = trx_pts(lbl, ndx)
        trx_split = np.random.random(n_trx) < conf.valratio
        cap = movies.Movie(local_dirs[ndx])

        for trx_ndx in range(n_trx):
            frames = np.where(np.invert(np.all(np.isnan(cur_pts[trx_ndx, :, :, :]), axis=(1, 2))))[0]
            cur_trx = trx[trx_ndx]

            for fnum in frames:

                cur_env = get_cur_env(env, val_env, split, conf, ndx, fnum, trx_ndx, is_val, trx_split, predefined=predefined)

                frame_in, cur_loc = get_patch_trx(cap, cur_trx, fnum, conf.imsz[0], cur_pts[trx_ndx, fnum, :, sel_pts])

                if conf.imgDim == 1:
                    frame_in = frame_in[:, :, 0:1]
                frame_in = frame_in[np.newaxis, ...]

                # read prev and next frames
                next_array = []
                prev_array = []
                for cur_t in range(tw):
                    next_fr, cur_loc = get_patch_trx(cap, cur_trx, fnum, conf.imsz[0], cur_pts[trx_ndx, fnum, :, sel_pts], cur_t+1)

                    if conf.imgDim == 1:
                        next_fr = next_fr[:, :, 0:1]
                    next_fr = next_fr[np.newaxis, ...]
                    next_array.append(next_fr)

                    prev_fr, cur_loc = get_patch_trx(cap, cur_trx, fnum, conf.imsz[0], cur_pts[trx_ndx, fnum, :, sel_pts], -cur_t-1)

                    if conf.imgDim == 1:
                        prev_fr = prev_fr[:, :, 0:1]
                    prev_fr = prev_fr[np.newaxis, ...]
                    prev_array.append(prev_fr)

                prev_array = [i for i in reversed(prev_array)]
                all_f = np.concatenate(prev_array + [frame_in, ] + next_array)

                assert conf.imsz[0] == conf.imsz[1]

                rows, cols = all_f.shape[1:3]
                depth = all_f.shape[3]

                image_raw = all_f.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': int64_feature(rows),
                    'width': int64_feature(cols),
                    'depth': int64_feature(depth),
                    'trx_ndx': int64_feature(trx_ndx),
                    'locs': float_feature(cur_loc.flatten()),
                    'expndx': float_feature(ndx),
                    'ts': float_feature(fnum),
                    'image_raw': bytes_feature(image_raw)}))
                cur_env.write(example.SerializeToString())

                if cur_env is val_env:
                    val_count += 1
                    splits[1].append([ndx,fnum,trx_ndx])
                else:
                    count += 1
                    splits[0].append([ndx,fnum,trx_ndx])

        cap.close()  # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx, len(local_dirs), count, val_count))
    env.close()
    val_env.close() if split else None
    print('%d,%d number of pos examples added to the db and val-db' % (count, val_count))
    lbl.close()
    with open(os.path.join(conf.cachedir,'splitdata.json'),'w') as f:
        json.dump(splits, f)


def create_tf_record_rnn_from_lbl_with_trx(conf, split=True, split_file=None):
    # Uses rnn_before and rnn_after.
    create_val_data(conf)
    is_val, local_dirs, _ = load_val_data(conf)

    lbl = h5py.File(conf.labelfile, 'r')
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    trx_files = get_trx_files(lbl, local_dirs)

    env_rnn, val_env_rnn = create_envs(conf, split)
    env, val_env = create_envs(conf, split, db_type=None)
    view = conf.view
    count = 0
    val_count = 0
    sel_pts = int(view * npts_per_view) + conf.selpts

    if conf.splitType is 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = json.load(split_file)
    else:
        predefined = None

    splits = [[],[]]
    for ndx, dir_name in enumerate(local_dirs):

        trx = sio.loadmat(trx_files[ndx])['trx'][0]
        n_trx = len(trx)

        cur_pts = trx_pts(lbl, ndx)
        trx_split = np.random.random(n_trx) < conf.valratio
        cap = movies.Movie(local_dirs[ndx])

        for trx_ndx in range(n_trx):
            frames = np.where(np.invert(np.all(np.isnan(cur_pts[trx_ndx, :, :, :]), axis=(1, 2))))[0]
            cur_trx = trx[trx_ndx]

            for fnum in frames:

                cur_env_rnn = get_cur_env(env, val_env, split, conf, ndx, fnum, trx_ndx, is_val, trx_split, predefined=predefined)

                # current frame
                frame_in, cur_loc = get_patch_trx(cap, cur_trx, fnum, conf.imsz[0], cur_pts[trx_ndx, fnum, :, sel_pts])

                if conf.imgDim == 1:
                    frame_in = frame_in[:, :, 0:1]
                frame_in = frame_in[np.newaxis, ...]

                # read prev and next frames
                next_array = []
                prev_array = []
                for cur_t in range(conf.rnn_after):
                    next_fr, _ = get_patch_trx(cap, cur_trx, fnum, conf.imsz[0], cur_pts[trx_ndx, fnum, :, sel_pts], cur_t+1)
                    if conf.imgDim == 1:
                        next_fr = next_fr[:, :, 0:1]
                    next_fr = next_fr[np.newaxis, ...]
                    next_array.append(next_fr)

                for cur_t in range(conf.rnn_before):
                    prev_fr, _ = get_patch_trx(cap, cur_trx, fnum, conf.imsz[0], cur_pts[trx_ndx, fnum, :, sel_pts], -cur_t-1)
                    if conf.imgDim == 1:
                        prev_fr = prev_fr[:, :, 0:1]
                    prev_fr = prev_fr[np.newaxis, ...]
                    prev_array.append(prev_fr)

                prev_array = [i for i in reversed(prev_array)]

                if not next_array:
                    all_f = np.concatenate(prev_array + [frame_in, ])
                else:
                    all_f = np.concatenate(prev_array + [frame_in, ] + next_array)

                assert conf.imsz[0] == conf.imsz[1]

                rows, cols = all_f.shape[1:3]
                depth = all_f.shape[3]

                image_raw = all_f.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': int64_feature(rows),
                    'width': int64_feature(cols),
                    'depth': int64_feature(depth),
                    'trx_ndx': int64_feature(trx_ndx),
                    'locs': float_feature(cur_loc.flatten()),
                    'expndx': float_feature(ndx),
                    'ts': float_feature(fnum),
                    'image_raw': bytes_feature(image_raw)}))
                cur_env_rnn.write(example.SerializeToString())

                if cur_env_rnn is val_env_rnn:
                    val_count += 1
                    splits[1].append([ndx,fnum,trx_ndx])
                else:
                    count += 1
                    splits[0].append([ndx,fnum,trx_ndx])

        cap.close()  # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx + 1, len(local_dirs), count, val_count))
    env_rnn.close()
    val_env_rnn.close() if split else None
    print('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
    lbl.close()
    with open(os.path.join(conf.cachedir,'splitdata.json'),'w') as f:
        json.dump(splits, f)


def read_and_decode(filename_queue, conf):
    if hasattr(conf,'has_trx_ndx'):
        has_trx_ndx = conf.has_trx_ndx
    else:
        has_trx_ndx = True

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if has_trx_ndx:
        features = tf.parse_single_example(
            serialized_example,
            features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                      'width': tf.FixedLenFeature([], dtype=tf.int64),
                      'depth': tf.FixedLenFeature([], dtype=tf.int64),
                      'trx_ndx': tf.FixedLenFeature([], dtype=tf.int64),
                      'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                      'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                      'ts': tf.FixedLenFeature([], dtype=tf.float32),
                      'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                      })
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                      'width': tf.FixedLenFeature([], dtype=tf.int64),
                      'depth': tf.FixedLenFeature([], dtype=tf.int64),
                      'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                      'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                      'ts': tf.FixedLenFeature([], dtype=tf.float32),
                      'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                      })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    if has_trx_ndx:
        trx_ndx = tf.cast(features['trx_ndx'], tf.int64)
    image = tf.reshape(image, conf.imsz + (conf.imgDim,))

    locs = tf.cast(features['locs'], tf.float64)
    exp_ndx = tf.cast(features['expndx'], tf.float64)
    ts = tf.cast(features['ts'], tf.float64)  # tf.constant([0]); #
    if has_trx_ndx:
        info = [exp_ndx, ts, trx_ndx]
    else:
        info = [exp_ndx, ts]
    return image, locs, info


def read_and_decode_multi(filename_queue, conf):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    n_max = conf.max_n_animals
    features = tf.parse_single_example(
        serialized_example,
        features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                  'width': tf.FixedLenFeature([], dtype=tf.int64),
                  'depth': tf.FixedLenFeature([], dtype=tf.int64),
                  'locs': tf.FixedLenFeature(shape=[n_max, conf.n_classes, 2], dtype=tf.float32),
                  'n_animals': tf.FixedLenFeature(1, dtype=tf.int64),
                  'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                  'ts': tf.FixedLenFeature([], dtype=tf.float32),
                  'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                  })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    n_animals = tf.cast(features['n_animals'], tf.int64)

    if conf.imgDim > 1:
        image = tf.reshape(image, conf.imsz + (conf.imgDim,))
    else:
        image = tf.reshape(image, conf.imsz)

    locs = tf.cast(features['locs'], tf.float64)
    exp_ndx = tf.cast(features['expndx'], tf.float64)
    ts = tf.cast(features['ts'], tf.float64)  # tf.constant([0]); #

    return image, locs, [exp_ndx, ts, n_animals]


def read_and_decode_time(filename_queue, conf):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # n_max = conf.max_n_animals
    features = tf.parse_single_example(
        serialized_example,
        features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                  'width': tf.FixedLenFeature([], dtype=tf.int64),
                  'depth': tf.FixedLenFeature([], dtype=tf.int64),
                  'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                  'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                  'ts': tf.FixedLenFeature([], dtype=tf.float32),
                  'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                  })
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    tw = 2 * conf.time_window_size + 1
    image = tf.reshape(image, (tw,) + conf.imsz + (conf.imgDim,))

    locs = tf.cast(features['locs'], tf.float64)
    expndx = tf.cast(features['expndx'], tf.float64)
    ts = tf.cast(features['ts'], tf.float64)  # tf.constant([0]); #

    return image, locs, [expndx, ts]


def read_and_decode_rnn(filename_queue, conf):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # n_max = conf.max_n_animals
    features = tf.parse_single_example(
        serialized_example,
        features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                  'width': tf.FixedLenFeature([], dtype=tf.int64),
                  'depth': tf.FixedLenFeature([], dtype=tf.int64),
                  'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                  'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                  'ts': tf.FixedLenFeature([], dtype=tf.float32),
                  'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                  })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    tw = conf.rnn_before + conf.rnn_after + 1
    image = tf.reshape(image, (tw,) + conf.imsz + (conf.imgDim,))

    locs = tf.cast(features['locs'], tf.float64)
    expndx = tf.cast(features['expndx'], tf.float64)
    ts = tf.cast(features['ts'], tf.float64)

    return image, locs, [expndx, ts]


def read_and_decode_without_session(filename, conf, indices=(0,)):
    # reads the tf record db. Returns entries at location indices
    # If indices is empty, then it reads the whole database.

    xx = tf.python_io.tf_record_iterator(filename)
    all_ims = []
    all_locs = []
    all_info = []
    for ndx, record in enumerate(xx):
        if (len(indices) > 0) and (indices.count(ndx) is 0):
            continue

        example = tf.train.Example()
        example.ParseFromString(record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        depth = int(example.features.feature['depth'].int64_list.value[0])
        expid = int(example.features.feature['expndx'].float_list.value[0])
        t = int(example.features.feature['ts'].float_list.value[0])
        img_string = example.features.feature['image_raw'].bytes_list.value[0]
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, depth))
        locs = np.array(example.features.feature['locs'].float_list.value)
        locs = locs.reshape([conf.n_classes, 2])
        if 'trx_ndx' in example.features.feature.keys():
            trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
        else:
            trx_ndx = 0

        all_ims.append(reconstructed_img)
        all_locs.append(locs)
        all_info.append([expid, t, trx_ndx])

    xx.close()
    return all_ims, all_locs, all_info


class tf_reader(object):

    def __init__(self, conf, filename, shuffle):
        self.conf = conf
        self.file = filename
        self.iterator  = None
        self.shuffle = shuffle
        self.batch_size = self.conf.batch_size
#        self.vec_num = len(conf.op_affinity_graph)
        self.heat_num = self.conf.n_classes
        self.N = PoseTools.count_records(filename)


    def reset(self):
        if self.iterator:
            self.iterator.close()
        self.iterator = tf.python_io.tf_record_iterator(self.file)


    def read_next(self):
        if not self.iterator:
            self.iterator = tf.python_io.tf_record_iterator(self.file)
        try:
            record = self.iterator.next()
        except StopIteration:
            self.reset()
            record = self.iterator.next()
        return  record

    def next(self):

        all_ims = []
        all_locs = []
        all_info = []
        for b_ndx in range(self.batch_size):
            n_skip = np.random.randint(30) if self.shuffle else 0
            for _ in range(n_skip+1):
                record = self.read_next()

            example = tf.train.Example()
            example.ParseFromString(record)
            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            depth = int(example.features.feature['depth'].int64_list.value[0])
            expid = int(example.features.feature['expndx'].float_list.value[0])
            t = int(example.features.feature['ts'].float_list.value[0])
            img_string = example.features.feature['image_raw'].bytes_list.value[0]
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((height, width, depth))
            locs = np.array(example.features.feature['locs'].float_list.value)
            locs = locs.reshape([self.conf.n_classes, 2])
            if 'trx_ndx' in example.features.feature.keys():
                trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
            else:
                trx_ndx = 0
            all_ims.append(reconstructed_img)
            all_locs.append(locs)
            all_info.append(np.array([expid, t, trx_ndx]))

        ims = np.stack(all_ims)
        locs = np.stack(all_locs)
        info = np.stack(all_info)

#        return {'orig_images':ims, 'orig_locs':locs, 'info':info, 'extra_info':np.zeros([self.batch_size,1])}
        return ims, locs, info, np.zeros([self.batch_size,1])

