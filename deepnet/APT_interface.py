from __future__ import division
from __future__ import print_function

import argparse
import collections
import datetime
import json
import logging
from os.path import expanduser
from random import sample

# import PoseUNet
import PoseUNet_dataset as PoseUNet
import PoseUNet_resnet as PoseURes
import hdf5storage
import imageio
import multiResData
from multiResData import *
import leap.training
from leap.training import train_apt as leap_train
import open_pose
from deepcut.train import train as deepcut_train
import deepcut.train
import ast
import tempfile


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


def read_entry(x):
    if type(x) is h5py._hl.dataset.Dataset:
        return x[0, 0]
    else:
        return x


def read_string(x):
    if type(x) is h5py._hl.dataset.Dataset:
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


def convert(in_data,to_python):
    if type(in_data) in [list,tuple]:
        out_data = []
        for i in in_data:
            out_data.append(convert(i,to_python))
    elif type(in_data) is dict:
        out_data = {}
        for i in in_data.keys():
            out_data[i] = convert(in_data[i],to_python)
    elif in_data is None:
        out_data = None
    else:
        offset = -1 if to_python else 1
        out_data = in_data+offset
    return out_data

def to_py(in_data):
    return convert(in_data,to_python=True)

def to_mat(in_data):
    return convert(in_data,to_python=False)

def tf_serialize(data):
    # serialize data for writing to tf records file.
    frame_in, cur_loc, info = data
    rows, cols, depth = frame_in.shape
    expid, fnum, trxid = info
    image_raw = frame_in.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': int64_feature(rows),
        'width': int64_feature(cols),
        'depth': int64_feature(depth),
        'trx_ndx': int64_feature(trxid),
        'locs': float_feature(cur_loc.flatten()),
        'expndx': float_feature(expid),
        'ts': float_feature(fnum),
        'image_raw': bytes_feature(image_raw)}))

    return example.SerializeToString()


def create_tfrecord(conf, split=True, split_file=None, use_cache=False, on_gt=False, db_files=()):
    # function that creates tfrecords using db_from_lbl
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)

    if on_gt:
        train_filename = db_files[0]
        env = tf.python_io.TFRecordWriter(train_filename)
        val_env = None
        envs = [env, val_env]
    elif len(db_files) > 1:
        train_filename = db_files[0]
        env = tf.python_io.TFRecordWriter(train_filename)
        val_filename = db_files[1]
        val_env = tf.python_io.TFRecordWriter(val_filename)
        envs = [env, val_env]
    else:
        try:
            envs = multiResData.create_envs(conf, split)
        except IOError:
            logging.exception('DB_WRITE: Could not write to tfrecord database')
            exit(1)

    out_fns = [lambda data: envs[0].write(tf_serialize(data)),
               lambda data: envs[1].write(tf_serialize(data))]
    if use_cache:
        splits = db_from_cached_lbl(conf, out_fns, split, split_file, on_gt)
    else:
        splits = db_from_lbl(conf, out_fns, split, split_file, on_gt)

    envs[0].close()
    envs[1].close() if split else None
    try:
        with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
            json.dump(splits, f)
    except IOError:
        logging.warning('SPLIT_WRITE: Could not output the split data information')


def to_orig(conf, locs, x, y, theta):
    hsz_p = conf.imsz[0] // 2  # half size for pred
    tt = -theta - math.pi / 2
    r_mat = [[np.cos(tt), -np.sin(tt)], [np.sin(tt), np.cos(tt)]]
    curlocs = np.dot(locs - [hsz_p, hsz_p], r_mat) + [x, y]
    return curlocs


def convert_to_orig(base_locs, conf, fnum, cur_trx, crop_loc):
    '''converts locs in cropped image back to locations in original image. base_locs need to be in 1-indexed mat system.
    base_locs should be 2 dim.
    fnum should be 0-indexed'''
    if conf.has_trx_file:
        trx_fnum = fnum - int(cur_trx['firstframe'][0, 0] -1 )
        x = to_py(int(round(cur_trx['x'][0, trx_fnum])))
        y = to_py(int(round(cur_trx['y'][0, trx_fnum])))
        theta = cur_trx['theta'][0, trx_fnum]
        assert conf.imsz[0] == conf.imsz[1]
        base_locs_orig = to_orig(conf, base_locs, x, y, theta)
    elif crop_loc is not None:
        xlo, xhi, ylo, yhi = crop_loc
        base_locs_orig = base_locs.copy()
        base_locs_orig[ :, 0] += xlo
        base_locs_orig[ :, 1] += ylo
    else:
        base_locs_orig = base_locs.copy()

    return base_locs_orig


def get_matlab_ts(filename):
    # matlab's time is different from python
    k = datetime.datetime.fromtimestamp(os.path.getmtime(filename))
    return datetime2matlabdn(k)


def convert_unicode(data):
    if isinstance(data, basestring):
        return unicode(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert_unicode, data.iteritems()))
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


def get_net_type(lbl_file):
    lbl = h5py.File(lbl_file, 'r')
    dt_params_ndx = None
    for ndx in range(lbl['trackerClass'].shape[0]):
        cur_tracker = ''.join([chr(c) for c in lbl[lbl['trackerClass'][ndx][0]]])
        if cur_tracker == 'DeepTracker':
            dt_params_ndx = ndx
    dt_params = lbl[lbl['trackerData'][dt_params_ndx][0]]['sPrm']

    if 'netType' in dt_params.keys():
        return read_string(dt_params['netType'])
    else:
        return None

def create_conf(lbl_file, view, name, cache_dir=None, net_type='unet',conf_params=None):
    try:
        try:
            lbl = loadmat(lbl_file)
        except NotImplementedError:
            logging.info('Label file is in v7.3 format. Loading using h5py')
            lbl = h5py.File(lbl_file, 'r')
    except TypeError as e:
        logging.exception('LBL_READ: Could not read the lbl file {}'.format(lbl_file))

    from poseConfig import config
    conf = config()
    conf.n_classes = int(read_entry(lbl['cfg']['NumLabelPoints']))
    if lbl['projname'][0] == 0:
        proj_name = 'default'
    else:
        proj_name = read_string(lbl['projname'])
    conf.view = view
    conf.set_exp_name(proj_name)
    # conf.cacheDir = read_string(lbl['cachedir'])
    conf.has_trx_file = has_trx_file(lbl[lbl['trxFilesAll'][0, 0]])
    conf.selpts = np.arange(conf.n_classes)

    dt_params_ndx = None
    for ndx in range(lbl['trackerClass'].shape[0]):
        cur_tracker = ''.join([chr(c) for c in lbl[lbl['trackerClass'][ndx][0]]])
        if cur_tracker == 'DeepTracker':
            dt_params_ndx = ndx
    dt_params = lbl[lbl['trackerData'][dt_params_ndx][0]]['sPrm']


    cache_dir = read_string(dt_params['CacheDir']) if cache_dir is None else cache_dir
    conf.cachedir = os.path.join(cache_dir, proj_name, net_type, 'view_{}'.format(view), name)

    if not os.path.exists(conf.cachedir):
        os.makedirs(conf.cachedir)

    # If the project has trx file then we use the crop locs
    # specified by the user. If the project doesnt have trx files
    # then we use the crop size specified by user else use the whole frame.
    if conf.has_trx_file:
        width = int(read_entry(dt_params['sizex']))
        height = int(read_entry(dt_params['sizey']))
        conf.imsz = (height, width)
    else:
        if lbl['cropProjHasCrops'][0, 0] == 1:
            xlo, xhi, ylo, yhi = PoseTools.get_crop_loc(lbl, 0, view)
            conf.imsz = (int(yhi - ylo + 1), int(xhi - xlo + 1))
        else:
            vid_nr = int(read_entry(lbl[lbl['movieInfoAll'][0, 0]]['info']['nr']))
            vid_nc = int(read_entry(lbl[lbl['movieInfoAll'][0, 0]]['info']['nc']))
            conf.imsz = (vid_nr, vid_nc)
    # crop_locX = int(read_entry(dt_params['CropX_view{}'.format(view + 1)]))
    # crop_locY = int(read_entry(dt_params['CropY_view{}'.format(view + 1)]))
    # conf.cropLoc = {(vid_nr, vid_nc): [crop_locY, crop_locX]}
    conf.labelfile = lbl_file
    conf.sel_sz = min(conf.imsz)
    conf.unet_rescale = float(read_entry(dt_params['scale']))
    conf.op_rescale = float(read_entry(dt_params['scale']))
    conf.dlc_rescale = float(read_entry(dt_params['scale']))
    conf.leap_rescale = float(read_entry(dt_params['scale']))
    conf.rescale = conf.unet_rescale
    conf.adjust_contrast = int(read_entry(dt_params['adjustContrast'])) > 0.5
    conf.normalize_img_mean = int(read_entry(dt_params['normalize'])) > 0.5
    # conf.img_dim = int(read_entry(dt_params['NChannels']))
    ex_mov = multiResData.find_local_dirs(conf)[0][0]

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
        conf.flipud = int(read_entry(dt_params['flipud'])) > 0.5
    except KeyError:
        pass
    try:
        conf.unet_steps = int(read_entry(dt_params['dl_steps']))
        conf.dl_steps = int(read_entry(dt_params['dl_steps']))
        # conf.dlc_steps = int(read_entry(dt_params['dl_steps']))
    except KeyError:
        pass
    try:
        conf.save_td_step = read_entry(dt_params['save_td_step'])
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['brange'])
        conf.brange = [-bb, bb]
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['crange'])
        conf.crange = [1 - bb, 1 + bb]
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['trange'])
        conf.trange = bb
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['rrange'])
        conf.rrange = bb
    except KeyError:
        pass
    try:
        bb = ''.join([chr(c) for c in dt_params['op_affinity_graph']]).split(',')
        graph = []
        for b in bb:
            mm = re.search('(\d+)\s+(\d+)', b)
            n1 = int(mm.groups()[0]) - 1
            n2 = int(mm.groups()[1]) - 1
            graph.append([n1, n2])
        conf.op_affinity_graph = graph
    except KeyError:
        pass

    conf.mdn_groups = [(i,) for i in range(conf.n_classes)]

    done_keys = ['CacheDir','scale','brange','crange','trange','rrange','op_affinity_graph','flipud','dl_steps','scale','adjustContrast','normalize','sizex','sizey']

    for k in dt_params.keys():
        if k in done_keys:
            continue

        if hasattr(conf,k):
            if type(getattr(conf,k)) == str:
                setattr(conf,k,read_string(dt_params[k]))
            else:
                attr_type = type(getattr(conf,k))
                setattr(conf, k, attr_type(read_entry(dt_params[k])))
        else:
            setattr(conf,k,read_entry(dt_params[k]))


    if conf_params is not None:
        cc = conf_params
        assert len(cc)%2 == 0, 'Config params should be in pairs of name value'
        for n,v in zip(cc[0::2],cc[1::2]):
            setattr(conf,n,ast.literal_eval(v))

    # overrides for each network
    if net_type == 'openpose':
        # openpose uses its own normalization
        conf.normalize_img_mean = False
    elif net_type == 'deeplabcut':
        conf.batch_size = 1
    elif net_type == 'unet':
        conf.use_pretrained_weights = False

    return conf


def get_cur_trx(trx_file, trx_ndx):
    try:
        trx = sio.loadmat(trx_file)['trx'][0]
        cur_trx = trx[trx_ndx]
        n_trx = len(trx)
    except NotImplementedError:
        # trx file in v7.3 format
        # print('Trx file is in v7.3 format. Loading using h5py')
        trx = h5py.File(trx_file,'r')['trx']
        cur_trx = {}
        for k in trx.keys():
            cur_trx[k] = np.array(trx[trx[k][trx_ndx,0]]).T
        n_trx = trx['x'].shape[0]
    return cur_trx, n_trx


def db_from_lbl(conf, out_fns, split=True, split_file=None, on_gt=False):
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

    assert not (on_gt and split), 'Cannot split gt data'

    local_dirs, _ = multiResData.find_local_dirs(conf, on_gt)
    lbl = h5py.File(conf.labelfile, 'r')
    view = conf.view
    flipud = conf.flipud
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    sel_pts = int(view * npts_per_view) + conf.selpts

    splits = [[], []]
    count = 0
    val_count = 0

    mov_split = None
    predefined = None
    if conf.splitType is 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = PoseTools.json_load(split_file)
    elif conf.splitType is 'movie':
        nexps = len(local_dirs)
        mov_split = sample(list(range(nexps)), int(nexps * conf.valratio))
        predefined = None
    elif conf.splitType is 'trx':
        assert conf.has_trx_file, 'Train/Validation was selected to be trx but the project has no trx files'

    for ndx, dir_name in enumerate(local_dirs):

        exp_name = conf.getexpname(dir_name)
        cur_pts = trx_pts(lbl, ndx, on_gt)
        crop_loc = PoseTools.get_crop_loc(lbl, ndx, view, on_gt)

        try:
            cap = movies.Movie(dir_name)
        except ValueError:
            logging.exception('MOVIE_READ: ' + local_dirs[ndx] + ' is missing')
            sys.exit(1)

        if conf.has_trx_file:
            trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)
            _, n_trx = get_cur_trx(trx_files[ndx],0)
            trx_split = np.random.random(n_trx) < conf.valratio
        else:
            trx = [None]
            n_trx = 1
            trx_split = None
            cur_pts = cur_pts[np.newaxis, ...]

        for trx_ndx in range(n_trx):

            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx, on_gt)
            cur_trx, _ = get_cur_trx(trx_files[ndx], trx_ndx)
            for fnum in frames:
                if not check_fnum(fnum, cap, exp_name, ndx):
                    continue

                info = [ndx, fnum, trx_ndx]
                cur_out = multiResData.get_cur_env(out_fns, split, conf, info, mov_split, trx_split=trx_split, predefined=predefined)

                frame_in, cur_loc = multiResData.get_patch( cap, fnum, conf, cur_pts[trx_ndx, fnum, :, sel_pts], cur_trx=cur_trx, flipud=flipud, crop_loc=crop_loc)
                cur_out([frame_in, cur_loc, info])

                if cur_out is out_fns[1] and split:
                    val_count += 1
                    splits[1].append(info)
                else:
                    count += 1
                    splits[0].append(info)

        cap.close()  # close the movie handles
        logging.info('Done %d of %d movies, train count:%d val count:%d' % (ndx + 1, len(local_dirs), count, val_count))

    logging.info('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
    lbl.close()
    return splits


def db_from_cached_lbl(conf, out_fns, split=True, split_file=None, on_gt=False):
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

    assert not (on_gt and split), 'Cannot split gt data'

    local_dirs, _ = multiResData.find_local_dirs(conf, on_gt)
    lbl = h5py.File(conf.labelfile, 'r')
    view = conf.view
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    sel_pts = int(view * npts_per_view) + conf.selpts

    splits = [[], []]
    count = 0
    val_count = 0

    mov_split = None
    predefined = None
    if conf.splitType is 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = PoseTools.json_load(split_file)
    elif conf.splitType is 'movie':
        nexps = len(local_dirs)
        mov_split = sample(list(range(nexps)), int(nexps * conf.valratio))
        predefined = None
    elif conf.splitType is 'trx':
        assert conf.has_trx_file, 'Train/Validation was selected to be trx but the project has no trx files'

    m_ndx = lbl['preProcData_MD_mov'].value[0, :].astype('int')
    t_ndx = lbl['preProcData_MD_iTgt'].value[0, :].astype('int') - 1
    f_ndx = lbl['preProcData_MD_frm'].value[0, :].astype('int') - 1

    if on_gt:
        m_ndx = -m_ndx

    if conf.has_trx_file:
        trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)

    prev_trx_mov = -1
    psz = max(conf.imsz)

    for ndx in range(lbl['preProcData_I'].shape[1]):
        if m_ndx[ndx] < 0:
            continue
        mndx = m_ndx[ndx] - 1
        if mndx != prev_trx_mov:
            cur_pts = trx_pts(lbl, mndx, on_gt)
            if cur_pts.ndim == 3:
                cur_pts = cur_pts[np.newaxis, ...]
        crop_loc = PoseTools.get_crop_loc(lbl, mndx, view, on_gt)
        cur_locs = cur_pts[t_ndx[ndx], f_ndx[ndx], :, sel_pts].copy()
        cur_frame = lbl[lbl['preProcData_I'][conf.view, ndx]].value.copy()
        cur_frame = cur_frame.T

        assert cur_frame.shape[0] == conf.imsz[0], 'height of cached images does not match the height specified in the params'
        assert cur_frame.shape[1] == conf.imsz[1], 'width of cached images does not match the width specified in the params'

        if cur_frame.ndim == 2:
            cur_frame = cur_frame[..., np.newaxis]

        if conf.has_trx_file:

            # dont load trx file if the current movie is same as previous.
            # and trx split wont work well if the frames for the same animal are not contiguous
            if ndx is 0 or t_ndx[ndx - 1] != t_ndx[ndx] or prev_trx_mov != mndx:
                cur_trx, n_trx = get_cur_trx(trx_files[mndx], t_ndx[ndx])

            if prev_trx_mov is not mndx:
                trx_split = np.random.random(n_trx) < conf.valratio

            prev_trx_mov = mndx

            x, y, theta = read_trx(cur_trx, f_ndx[ndx])

            cur_frame, cur_locs = multiResData.crop_patch_trx(conf, cur_frame, psz//2, psz//2, theta, cur_locs-[x,y] + [psz//2,psz//2])

            # theta = theta + math.pi / 2
            # patch = cur_frame
            # rot_mat = cv2.getRotationMatrix2D((psz / 2, psz / 2), theta * 180 / math.pi, 1)
            # rpatch = cv2.warpAffine(patch, rot_mat, (psz, psz))
            # if rpatch.ndim == 2:
            #     rpatch = rpatch[:, :, np.newaxis]
            # cur_frame = rpatch
            #
            # ll = cur_locs.copy()
            # ll = ll - [x, y]
            # rot = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            # lr = np.dot(ll, rot) + [psz / 2, psz / 2]
            # if conf.imsz[0] < conf.imsz[1]:
            #     extra = (psz - conf.imsz[0]) / 2
            #     lr[:, 1] -= extra
            # elif conf.imsz[1] < conf.imsz[0]:
            #     extra = (psz - conf.imsz[1]) / 2
            #     lr[:, 0] -= extra
            # cur_locs = lr

        else:
            trx_split = None
            if crop_loc is not None:
                xlo, xhi, ylo, yhi = crop_loc
            else:
                xlo = 0
                ylo = 0

            cur_locs[:, 0] = cur_locs[:, 0] - xlo  # ugh, the nasty x-y business.
            cur_locs[:, 1] = cur_locs[:, 1] - ylo
            # -1 because matlab is 1-indexed

        info = [mndx, f_ndx[ndx], t_ndx[ndx]]

        cur_out = multiResData.get_cur_env(out_fns, split, conf, info,
                                           mov_split, trx_split=trx_split, predefined=predefined)

        cur_out([cur_frame, cur_locs, info])

        if cur_out is out_fns[1] and split:
            val_count += 1
            splits[1].append(info)
        else:
            count += 1
            splits[0].append(info)

        if ndx % 100 == 99 and ndx > 0:
            logging.info('%d,%d number of pos examples added to the db and valdb' % (count, val_count))

    logging.info('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
    lbl.close()
    return splits


def create_leap_db(conf, split=False, split_file=None, use_cache=False):
    # function showing how to use db_from_lbl for tfrecords
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)

    train_data = []
    val_data = []

    # collect the images and labels in arrays
    out_fns = [lambda data: train_data.append(data), lambda data: val_data.append(data)]
    if use_cache:
        splits = db_from_cached_lbl(conf, out_fns, split, split_file)
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

        ims = np.array([i[0] for i in cur_data])
        locs = np.array([i[1] for i in cur_data])
        info = np.array([i[2] for i in cur_data])
        hmaps = PoseTools.create_label_images(locs, conf.imsz[:2], 1, conf.label_blur_rad)
        hmaps = (hmaps + 1) / 2  # brings it back to [0,1]

        hf = h5py.File(out_file, 'w')
        hf.create_dataset('box', data=ims)
        hf.create_dataset('confmaps', data=hmaps)
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
            im = data[0][:, :, 0]
        else:
            im = data[0]
        img_name = os.path.join(outdir, 'img_{:06d}.png'.format(count[0]))
        imageio.imwrite(img_name, im)
        locs = data[1]
        bp = conf.n_classes
        for b in range(bp):
            fis[b].write('{}\t{}\t{}\n'.format(count[0], locs[b, 0], locs[b, 1]))
        mod_locs = np.insert(np.array(locs), 0, range(bp), axis=1)
        save_data.append([img_name, im.shape, mod_locs])
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
        splits = db_from_cached_lbl(conf, out_fns, split, split_file)
    else:
        splits = db_from_lbl(conf, out_fns, split, split_file)
    [f.close() for f in train_fis]
    [f.close() for f in val_fis]
    with open(os.path.join(conf.cachedir, 'train_data.p'), 'w') as f:
        pickle.dump(train_data, f, protocol=2)
    if split:
        with open(os.path.join(conf.cachedir, 'val_data.p'), 'w') as f:
            pickle.dump(val_data, f, protocol=2)

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

        elif conf.splitType == 'frames':
            for ndx in range(len(local_dirs)):
                for mndx in range(len(mov_info[ndx])):
                    valid_folds = np.where(per_fold < lbls_per_fold)[0]
                    cur_fold = np.random.choice(valid_folds)
                    splits[cur_fold].extend(mov_info[ndx][mndx:mndx + 1])
                    per_fold[cur_fold] += 1
        else:
            raise ValueError('splitType has to be either movie trx or frames')

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
        with open(cur_split_file, 'w'):
            json.dump([cur_train, splits[ndx]])

    return all_train, splits, split_files


def create_batch_ims(to_do_list, conf, cap, flipud, trx, crop_loc):
    bsize = conf.batch_size
    all_f = np.zeros((bsize,) + conf.imsz + (conf.img_dim,))
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
        first_frames = np.array([x['firstframe'][0, 0] for x in trx]) - 1  # for converting from 1 indexing to 0 indexing
    else:
        trx = [None, ]
        n_trx = 1
        end_frames = np.array([n_frames])
        first_frames = np.array([0])
    return trx, first_frames, end_frames, n_trx


def get_trx_ids(trx_ids_in, n_trx, has_trx_file):
    if has_trx_file:
        if len(trx_ids_in) == 0:
            trx_ids = np.arange(n_trx)
        else:
            trx_ids = np.array(trx_ids_in)
    else:
        trx_ids = np.array([0])
    return trx_ids


def get_augmented_images(conf, out_file, distort=True, on_gt = False, use_cache=True,nsamples=None):

        data_in = []
        out_fns = [lambda data: data_in.append(data),
                   lambda data: None]
        if use_cache:
            splits = db_from_cached_lbl(conf, out_fns, False, None, on_gt)
        else:
            splits = db_from_lbl(conf, out_fns, False, None, on_gt)

        ims = np.array([d[0] for d in data_in])
        locs = np.array([d[1] for d in data_in])
        if nsamples is not None:
            sel = np.random.choice(ims.shape[0],nsamples)
            ims = ims[sel,...]
            locs = locs[sel,...]

        ims, locs = PoseTools.preprocess_ims(ims,locs,conf,distort,conf.rescale)

        hdf5storage.savemat(out_file,{'ims':ims,'locs':locs})


def classify_list(conf, pred_fn, cap, to_do_list, trx_file, crop_loc):
    '''Classify a list of images
    all inputs and outputs are 0-indexed
    '''

    flipud = conf.flipud
    bsize = conf.batch_size
    n_list = len(to_do_list)
    n_batches = int(math.ceil(float(n_list) / bsize))
    pred_locs = np.zeros([n_list, conf.n_classes, 2])
    pred_locs[:] = np.nan
    trx, first_frames, _, _ = get_trx_info(trx_file, conf, 0)
    sz = (cap.get_height(), cap.get_width())

    for cur_b in range(n_batches):

        cur_start = cur_b * bsize
        ppe = min(n_list - cur_start, bsize)
        all_f = create_batch_ims(to_do_list[cur_start:(cur_start + ppe)], conf, cap, flipud, trx, crop_loc)

        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']
        hmaps = ret_dict['hmaps']

        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            trx_ndx = cur_entry[1]
            cur_trx = trx[trx_ndx]
            cur_f = cur_entry[0]
            base_locs_orig = convert_to_orig(base_locs[cur_t, ...], conf, cur_f, cur_trx, crop_loc)
            # pred_locs[cur_start + cur_t, :, :] = base_locs_orig[0, ...]
            # KB 20190123: this was just copying the first landmark for all landmarks
            pred_locs[cur_start + cur_t, :, :] = base_locs_orig

    return pred_locs


def get_pred_fn(model_type, conf, model_file=None,name='deepnet'):
    ''' Returns prediction functions and close functions for different network types

    '''
    if model_type == 'openpose':
        pred_fn, close_fn, model_file = open_pose.get_pred_fn(conf, model_file,name=name)
    elif model_type == 'unet':
        pred_fn, close_fn, model_file = get_unet_pred_fn(conf, model_file,name=name)
    elif model_type == 'mdn':
        pred_fn, close_fn, model_file = get_mdn_pred_fn(conf, model_file,name=name)
    elif model_type == 'leap':
        pred_fn, close_fn, model_file = leap.training.get_pred_fn(conf, model_file,name=name)
    elif model_type == 'deeplabcut':
        pred_fn, close_fn, model_file = deepcut.train.get_pred_fn(conf, model_file,name=name)
    else:
        raise ValueError('Undefined type of model')

    return pred_fn, close_fn, model_file


def classify_list_all(model_type, conf, in_list, on_gt, model_file, movie_files=None, trx_files=None, crop_locs=None):
    '''
    Classifies a list of examples.
    in_list should be of list of type [mov_file, frame_num, trx_ndx]
    everything is 0-indexed
    '''
    pred_fn, close_fn, model_file = get_pred_fn(model_type, conf, model_file)

    if on_gt:
        local_dirs, _ = multiResData.find_gt_dirs(conf)
    else:
        local_dirs, _ = multiResData.find_local_dirs(conf)
    is_external_movies = False
    if movie_files is not None:
        local_dirs = movie_files
        is_external_movies = True
        is_crop = (crop_locs is not None) and (len(crop_locs) > 0)

    lbl = h5py.File(conf.labelfile, 'r')
    view = conf.view
    if conf.has_trx_file:
        if not is_external_movies:
            trx_files = multiResData.get_trx_files(lbl, local_dirs)
    else:
        trx_files = [None, ] * len(local_dirs)

    pred_locs = np.zeros([len(in_list), conf.n_classes, 2])
    pred_locs[:] = np.nan

    for ndx, dir_name in enumerate(local_dirs):

        cur_list = [[l[1] , l[2] ] for l in in_list if l[0] == ndx]
        cur_idx = [i for i, l in enumerate(in_list) if l[0] == ndx]
        if is_external_movies:
            if is_crop:
                crop_loc = crop_locs[ndx]
            else:
                crop_loc = None
        else:
            crop_loc = PoseTools.get_crop_loc(lbl, ndx, view, on_gt)

        try:
            cap = movies.Movie(dir_name)
        except ValueError:
            logging.exception('MOVIE_READ: ' + local_dirs[ndx] + ' is missing')
            exit(1)

        cur_pred_locs = classify_list(conf, pred_fn, cap, cur_list, trx_files[ndx], crop_loc)
        pred_locs[cur_idx, ...] = cur_pred_locs

        cap.close()  # close the movie handles

    lbl.close()
    close_fn()
    return pred_locs


def classify_db(conf, read_fn, pred_fn, n, return_ims=False):
    '''Classifies n examples generated by read_fn'''
    bsize = conf.batch_size
    all_f = np.zeros((bsize,) + conf.imsz + (conf.img_dim,))
    pred_locs = np.zeros([n, conf.n_classes, 2])
    n_batches = int(math.ceil(float(n) / bsize))
    labeled_locs = np.zeros([n, conf.n_classes, 2])
    info = []
    if return_ims:
        all_ims = np.zeros([n, conf.imsz[0], conf.imsz[1], conf.img_dim])
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            all_f[ndx, ...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])
        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']
        hmaps = ret_dict['hmaps']

        for ndx in range(ppe):
            pred_locs[cur_start + ndx, ...] = base_locs[ndx, ...]
            if return_ims:
                all_ims[cur_start + ndx, ...] = all_f[ndx, ...]

    if return_ims:
        return pred_locs, labeled_locs, info, all_ims
    else:
        return pred_locs, labeled_locs, info


def classify_db_all(model_type, conf, db_file, model_file=None):
    ''' Classifies examples in DB'''
    if model_type == 'openpose':
        tf_iterator = multiResData.tf_reader(conf, db_file, False)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        pred_fn, close_fn, model_file = open_pose.get_pred_fn(conf, model_file)
        pred_locs, label_locs, info = classify_db(conf, read_fn, pred_fn, tf_iterator.N)
        close_fn()
    elif model_type == 'unet':
        tf_iterator = multiResData.tf_reader(conf, db_file, False)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        pred_fn, close_fn, model_file = get_unet_pred_fn(conf, model_file)
        pred_locs, label_locs, info = classify_db(conf, read_fn, pred_fn, tf_iterator.N)
        close_fn()
    elif model_type == 'mdn':
        tf_iterator = multiResData.tf_reader(conf, db_file, False)
        tf_iterator.batch_size = 1
        read_fn = tf_iterator.next
        pred_fn, close_fn, model_file = get_mdn_pred_fn(conf, model_file)
        pred_locs, label_locs, info = classify_db(conf, read_fn, pred_fn, tf_iterator.N)
        close_fn()
    elif model_type == 'leap':
        leap_gen, n = leap.training.get_read_fn(conf, db_file)
        pred_fn, close_fn, latest_model_file = leap.training.get_pred_fn(conf, model_file)
        pred_locs, label_locs, info = classify_db(conf, leap_gen, pred_fn, n)
    elif model_type == 'deeplabcut':
        read_fn, n = deepcut.train.get_read_fn(conf, db_file)
        pred_fn, close_fn, latest_model_file = deepcut.train.get_pred_fn(conf, model_file)
        pred_locs, label_locs, info = classify_db(conf, read_fn, pred_fn, n)
    else:
        raise ValueError('Undefined model type')

    return pred_locs, label_locs, info


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
        print('Checking db from {}'.format(db_file))
        read_fn, n = deepcut.train.get_read_fn(conf, db_file)
    else:
        raise ValueError('Undefined model type')

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
def classify_list_file(conf, model_type, list_file, model_file, out_file):

    success = False
    pred_locs = None
    list_fp = open(list_file,'r')

    if not os.path.isfile(list_file):
        print('File %s does not exist'%list_file)
        return success, pred_locs

    toTrack = json.load(list_fp)

    # minimal checks
    if 'movieFiles' not in toTrack:
        print('movieFiles not defined in json file %s'%list_file)
        return success, pred_locs
    nMovies = len(toTrack['movieFiles'])
    if 'toTrack' not in toTrack:
        print('toTrack list not defined in json file %s'%list_file)
        return success, pred_locs
        
    hasTrx = 'trxFiles' in toTrack
    trxFiles = []
    if hasTrx:
        nTrx = len(toTrack['trxFiles'])
        if nTrx != nMovies:
            print('Numbers of movies and trx files do not match')
            return success, pred_locs
        trxFiles = toTrack['trxFiles']
    hasCrops = 'cropLocs' in toTrack
    cropLocs = None
    if hasCrops:
        nCrops = len(toTrack['cropLocs'])
        if nCrops != nMovies:
            print('Number of movie files and cropLocs do not match')
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
            print('toTrack[%d] has out of range movie index %d'%(i,mov))
            return success, pred_locs
        if tgt <= 0:
            print('toTrack[%d] has out of range target index %d'%(i,tgt))
            return success, pred_locs
        if frm <= 0:
            print('toTrack[%d] has out of range frame index %d'%(i,frm))
            return success, pred_locs

        cur_list.append([mov-1,frm-1,tgt-1])

    pred_locs = classify_list_all(model_type, conf, cur_list, on_gt=False, model_file=model_file, movie_files=toTrack['movieFiles'], trx_files=trxFiles, crop_locs=cropLocs)    
    mat_pred_locs = to_mat(pred_locs)

    sio.savemat(out_file, {'pred_locs': mat_pred_locs,
                           'list_file': list_file} )

    success = True

    return success, pred_locs

def classify_gt_data(conf, model_type, out_file, model_file):
    ''' Classify GT data in the label file.
    Returned values are 0-indexed.
    Saved values are 1-indexed.
    '''
    local_dirs, _ = multiResData.find_gt_dirs(conf)
    lbl = h5py.File(conf.labelfile, 'r')
    view = conf.view
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    sel_pts = int(view * npts_per_view) + conf.selpts

    cur_list = []
    labeled_locs = []
    for ndx, dir_name in enumerate(local_dirs):
        cur_pts = trx_pts(lbl, ndx, on_gt=True)

        if not conf.has_trx_file:
            cur_pts = cur_pts[np.newaxis, ...]

        if conf.has_trx_file:
            trx_files = multiResData.get_trx_files(lbl, local_dirs)
            trx = sio.loadmat(trx_files[ndx])['trx'][0]
            n_trx = len(trx)
        else:
            n_trx = 1

        for trx_ndx in range(n_trx):
            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx, on_gt=True)
            for f in frames:
                cur_list.append([ndx , f , trx_ndx ])
                labeled_locs.append(cur_pts[trx_ndx, f, :, sel_pts])

    pred_locs = classify_list_all(model_type, conf, cur_list, on_gt=True, model_file=model_file)
    mat_pred_locs = to_mat(pred_locs)
    mat_labeled_locs = to_mat(np.array(labeled_locs))
    mat_list = cur_list

    sio.savemat(out_file, {'pred_locs': mat_pred_locs,
                           'labeled_locs': mat_labeled_locs,
                           'list': mat_list})
    lbl.close()
    return pred_locs, labeled_locs, cur_list


def convert_to_mat_trk(in_pred, conf, start, end, trx_ids):
    ''' Converts predictions to compatible trk format'''
    pred_locs = in_pred.copy()
    pred_locs = pred_locs[:, trx_ids, ...]
    pred_locs = pred_locs[:(end-start), ...]
    if pred_locs.ndim == 4:
        pred_locs = pred_locs.transpose([2, 3, 0, 1])
    else:
        pred_locs = pred_locs.transpose([2, 0, 1])
    if not conf.has_trx_file:
        pred_locs = pred_locs[..., 0]
    return pred_locs


def write_trk(out_file, pred_locs_in, extra_dict, start, end, trx_ids, conf, info, mov_file):
    '''
    pred_locs is the predicted locations of size
    n_frames in the movie x n_Trx x n_body_parts x 2
    n_done is the number of frames that have been tracked.
    everything should be 0-indexed
    '''
    pred_locs = convert_to_mat_trk(pred_locs_in, conf, start, end, trx_ids)
    pred_locs = to_mat(pred_locs)

    tgt = to_mat(np.array(trx_ids))  # target animals that have been tracked.
    # For projects without trx file this is always 1.
    ts_shape = pred_locs.shape[0:1] + pred_locs.shape[2:]
    ts = np.ones(ts_shape) * datetime2matlabdn()  # time stamp
    tag = np.zeros(ts.shape).astype('bool')  # tag which is always false for now.
    tracked_shape = pred_locs.shape[2]
    tracked = np.zeros([1,
                        tracked_shape])  # which of the predlocs have been tracked. Mostly to help APT know how much tracking has been done.
    tracked[0, :] = to_mat(np.arange(start,end))

    out_dict = {'pTrk': pred_locs,
                'pTrkTS': ts,
                'expname': mov_file,
                'pTrkiTgt': tgt,
                'pTrkTag': tag,
                'pTrkFrm': tracked,
                'trkInfo': info}
    for k in extra_dict.keys():
        out_dict['pTrk' + k] = convert_to_mat_trk(extra_dict[k], conf, start, end, trx_ids)

    hdf5storage.savemat(out_file, out_dict, appendmat=False, truncate_existing=True)


def classify_movie(conf, pred_fn,
                   mov_file='',
                   out_file='',
                   trx_file=None,
                   start_frame=0,
                   end_frame=-1,
                   skip_rate=1,
                   trx_ids=(),
                   model_file='',
                   name='',
                   save_hmaps=False,
                   crop_loc=None):
    ''' Classifies frames in a movie. All animals in a frame are classified before moving to the next frame.'''

    cap = movies.Movie(mov_file)
    sz = (cap.get_height(), cap.get_width())
    n_frames = int(cap.get_n_frames())
    T, first_frames, end_frames, n_trx = get_trx_info(trx_file, conf, n_frames)
    trx_ids = get_trx_ids(trx_ids, n_trx, conf.has_trx_file)
    bsize = conf.batch_size
    flipud = conf.flipud

    info = {}  # tracking info. Can be empty.
    info[u'model_file'] = model_file
    info[u'trnTS'] = get_matlab_ts(model_file + '.meta')
    info[u'name'] = name
    param_dict = convert_unicode(conf.__dict__.copy())
    param_dict.pop('cropLoc', None)
    info[u'params'] = param_dict

    if end_frame < 0: end_frame = end_frames.max()
    if end_frame > end_frames.max(): end_frame = end_frames.max()
    if start_frame > end_frame: return None

    max_n_frames = end_frame - start_frame
    min_first_frame = start_frame
    pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
    pred_locs[:] = np.nan

    extra_dict = {}

    hmap_out_dir = os.path.splitext(out_file)[0] + '_hmap'
    if not os.path.exists(hmap_out_dir):
        os.mkdir(hmap_out_dir)

    to_do_list = []
    for cur_f in range(start_frame, end_frame):
        for t in range(n_trx):
            if not np.any(trx_ids == t):
                continue
            if (end_frames[t] > cur_f) and (first_frames[t] <= cur_f):
                to_do_list.append([cur_f, t])

    n_list = len(to_do_list)
    n_batches = int(math.ceil(float(n_list) / bsize))
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n_list - cur_start, bsize)
        all_f = create_batch_ims(to_do_list[cur_start:(cur_start + ppe)], conf, cap, flipud, T, crop_loc)


        ret_dict = pred_fn(all_f)
        base_locs = ret_dict.pop('locs')
        hmaps = ret_dict.pop('hmaps')
        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            trx_ndx = cur_entry[1]
            cur_trx = T[trx_ndx]
            cur_f = cur_entry[0]
            base_locs_orig = convert_to_orig(base_locs[cur_t, ...], conf, cur_f, cur_trx, crop_loc)
            pred_locs[cur_f - min_first_frame, trx_ndx, :, :] = base_locs_orig[ ...]

            if save_hmaps:
                write_hmaps(hmaps[cur_t, ...], hmap_out_dir, trx_ndx, cur_f)

            # for everything else that is returned..
            for k in ret_dict.keys():

                if ret_dict[k].ndim == 4:  # hmaps
                    if save_hmaps:
                        cur_hmap = ret_dict[k]
                        write_hmaps(cur_hmap[cur_t, ...], hmap_out_dir, trx_ndx, cur_f, k[5:])

                else:
                    cur_v = ret_dict[k]
                    if not extra_dict.has_key(k):
                        sz = cur_v.shape[1:]
                        extra_dict[k] = np.zeros((max_n_frames, n_trx) + sz)

                    if k.startswith('locs'):  # transform locs
                        cur_orig = convert_to_orig(cur_v[cur_t, ...], conf, cur_f, cur_trx, crop_loc)
                    else:
                        cur_orig = cur_v[cur_t, ...]

                    extra_dict[k][cur_f - min_first_frame, trx_ndx, ...] = cur_orig

        if cur_b % 20 == 19:
            sys.stdout.write('.')
        if cur_b % 400 == 399:
            sys.stdout.write('\n')
            write_trk(out_file + '.part', pred_locs, extra_dict, start_frame, to_do_list[cur_start][0], trx_ids, conf, info, mov_file)

    write_trk(out_file, pred_locs, extra_dict, start_frame, end_frame, trx_ids, conf, info, mov_file)
    if os.path.exists(out_file + '.part'):
        os.remove(out_file + '.part')
    cap.close()
    tf.reset_default_graph()
    return pred_locs


def get_unet_pred_fn(conf, model_file=None,name='deepnet'):
    ''' Prediction function for UNet network'''
    tf.reset_default_graph()
    self = PoseUNet.PoseUNet(conf, name=name)
    if name == 'deepnet':
        self.train_data_name = 'traindata'
    return self.get_pred_fn(model_file)


def get_mdn_pred_fn(conf, model_file=None,name='deepnet'):
    tf.reset_default_graph()
    self = PoseURes.PoseUMDN_resnet(conf, name=name)
    if name == 'deepnet':
        self.train_data_name = 'traindata'
    return self.get_pred_fn(model_file)


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
        files = leap.training.latest_model(conf, name)
    elif net_type == 'openpose':
        files = open_pose.latest_model(conf, name)
    elif net_type == 'deeplabcut':
        files = deepcut.train.model_files(conf, name)
    else:
        assert False, 'Undefined Net Type'

    for f in files:
        assert os.path.exists(f), 'Model file {} does not exist'.f

    return files

def classify_movie_all(model_type, **kwargs):
    ''' Classify movie wrapper'''
    conf = kwargs['conf']
    model_file = kwargs['model_file']
    train_name = kwargs['train_name']
    del kwargs['model_file'], kwargs['conf'], kwargs['train_name']
    pred_fn, close_fn, model_file = get_pred_fn(model_type, conf, model_file,name=train_name)
    logging.info('Saving hmaps') if kwargs['save_hmaps'] else logging.info('NOT saving hmaps')
    try:
        classify_movie(conf, pred_fn, model_file=model_file, **kwargs)
    except (IOError, ValueError) as e:
        close_fn()
        logging.exception('Could not track movie')


def train_unet(conf, args, restore,split, split_file=None):
    if not args.skip_db:
        create_tfrecord(conf, split=split, use_cache=args.use_cache,split_file=split_file)
    tf.reset_default_graph()
    self = PoseUNet.PoseUNet(conf, name='deepnet')
    self.train_data_name = 'traindata'
    self.train_unet(restore=restore)


def train_mdn(conf, args, restore,split, split_file=None):
    if not args.skip_db:
        create_tfrecord(conf, split=split, use_cache=args.use_cache,split_file=split_file)
    tf.reset_default_graph()
    self = PoseURes.PoseUMDN_resnet(conf, name=args.train_name)
    if args.train_name == 'deepnet':
        self.train_data_name = 'traindata'
    else:
        self.train_data_name = None
    self.train_umdn(restore=restore)


def train_leap(conf, args, split, split_file=None):
    if not args.skip_db:
        create_leap_db(conf, split=split, use_cache=args.use_cache,split_file=split_file)
    leap_train(conf,name=args.train_name)


def train_openpose(conf, args, split, split_file=None):
    if not args.skip_db:
        create_tfrecord(conf, split=split, use_cache=args.use_cache,split_file=split_file)
    open_pose.training(conf,name=args.train_name)


def train_deepcut(conf, args, split_file=None):
    if not args.skip_db:
        create_deepcut_db(conf, False, use_cache=args.use_cache,split_file=split_file)
    deepcut_train(conf,name=args.train_name)


def train(lblfile, nviews, name, args):
    ''' Creates training db and calls the appropriate network's training function '''

    view = args.view
    net_type = args.type
    restore = args.restore
    if view is None:
        views = range(nviews)
    else:
        views = [view]

    for cur_view in views:
        conf = create_conf(lblfile, cur_view, name, net_type=net_type, cache_dir=args.cache,conf_params=args.conf_params)

        conf.view = cur_view
        if args.split_file is not None:
            assert(os.path.exists(args.split_file))
            in_data = PoseTools.json_load(args.split_file)
            out_data = []
            for d in in_data:
                out_data.append((np.array(d)-1).tolist())
            t_file = os.path.join(tempfile.tempdir,next(tempfile._get_candidate_names()))
            with open(t_file,'w') as f:
                json.dump(out_data,f)
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
                train_mdn(conf, args, restore, split, split_file=split_file)
            elif net_type == 'openpose':
                if args.use_defaults:
                    open_pose.set_openpose_defaults(conf)
                train_openpose(conf, args, split, split_file=split_file)
            elif net_type == 'leap':
                if args.use_defaults:
                    leap.training.set_leap_defaults(conf)
                train_leap(conf, args, split, split_file=split_file)
            elif net_type == 'deeplabcut':
                if args.use_defaults:
                    deepcut.train.set_deepcut_defaults(conf)
                train_deepcut(conf,args, split_file=split_file)
        except tf.errors.InternalError as e:
            logging.exception(
                'Could not create a tf session. Probably because the CUDA_VISIBLE_DEVICES is not set properly')
            exit(1)
        except tf.errors.ResourceExhaustedError as e:
            logging.exception('Out of GPU Memory. Either reduce the batch size or scale down the image')
            exit(1)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("lbl_file",
                        help="path to lbl file")
    parser.add_argument('-name', dest='name', help='Name for the run. Default - pose_unet', default='pose_unet')
    parser.add_argument('-view', dest='view', help='Run only for this view. If not specified, run for all views',
                        default=None, type=int)
    parser.add_argument('-model_file', dest='model_file',
                        help='Use this model file. For tracking this overrides the latest model file. For training this will be used for initialization',
                        default=None)
    parser.add_argument('-cache', dest='cache', help='Override cachedir in lbl file', default=None)
    parser.add_argument('-debug', dest='debug', help='Print debug messages', action='store_true')
    parser.add_argument('-train_name', dest='train_name', help='Training name', default='deepnet')
    parser.add_argument('-err_file', dest='err_file', help='Err file', default=None)
    parser.add_argument('-log_file', dest='log_file', help='Log file', default=None)
    parser.add_argument('-conf_params', dest='conf_params', help='conf params. These will override params from lbl file', default=None, nargs='*')
    parser.add_argument('-type', dest='type', help='Network type, default is unet', default='unet',
                        choices=['unet', 'openpose', 'deeplabcut', 'leap', 'mdn'])
    subparsers = parser.add_subparsers(help='train or track or gt_classify', dest='sub_name')

    parser_train = subparsers.add_parser('train', help='Train the detector')
    parser_train.add_argument('-skip_db', dest='skip_db', help='Skip creating the data base', action='store_true')
    parser_train.add_argument('-use_defaults', dest='use_defaults', action='store_true',
                              help='Use default settings of openpose, deeplabcut or leap')
    parser_train.add_argument('-use_cache', dest='use_cache', action='store_true',
                              help='Use cached images in the label file to generate the training data.')
    parser_train.add_argument('-continue', dest='restore', action='store_true',
                              help='Continue from previously unfinished traning. Only for unet')
    parser_train.add_argument('-split_file', dest='split_file', help='Split file to split data for train and validation', default=None)
    # parser_train.add_argument('-cache',dest='cache_dir',
    #                           help='cache dir for training')

    parser_classify = subparsers.add_parser('track', help='Track a movie')
    parser_classify.add_argument("-mov", dest="mov",
                                 help="movie(s) to track", nargs='+') # KB 20190123 removed required because list_file does not require mov
    parser_classify.add_argument("-trx", dest="trx",
                                 help='trx file for above movie', default=None, nargs='*')
    parser_classify.add_argument('-start_frame', dest='start_frame', help='start tracking from this frame', type=int,
                                 default=1)
    parser_classify.add_argument('-end_frame', dest='end_frame', help='end frame for tracking', type=int, default=-1)
    parser_classify.add_argument('-skip_rate', dest='skip', help='frames to skip while tracking', default=1, type=int)
    parser_classify.add_argument('-out', dest='out_files', help='file to save tracking results to', required=True,
                                 nargs='+')
    parser_classify.add_argument('-trx_ids', dest='trx_ids', help='only track these animals', nargs='*', type=int,
                                 default=[])
    parser_classify.add_argument('-hmaps', dest='hmaps', help='generate heatmpas', action='store_true')
    parser_classify.add_argument('-crop_loc', dest='crop_loc', help='crop location given xlo xhi ylo yhi', nargs='*', type=int,
                                 default=None)
    parser_classify.add_argument('-list_file',dest='list_file', help='JSON file with list of movies, targets and frames to track',default=None)

    parser_gt = subparsers.add_parser('gt_classify', help='Classify GT labeled frames')
    parser_gt.add_argument('-out', dest='out_file_gt',
                           help='Mat file to save output to. _[view_num].mat will be appended', required=True)

    parser_aug = subparsers.add_parser('data_aug', help='get the augmented images')
    parser_aug.add_argument('-no_aug',dest='no_aug',help='dont augment the images. Return the original images',default=False)
    parser_aug.add_argument('-out_file',dest='out_file',help='Destination to save the images',required=True)
    parser_aug.add_argument('-use_cache', dest='use_cache', action='store_true', help='Use cached images in the label file to generate the augmented images')
    parser_aug.add_argument('-nsamples', dest='nsamples', default=None, help='Number of examples to be generated',type=int)

    parser_db = subparsers.add_parser('classify', help='Classify validation data')
    parser_db.add_argument('-out_file',dest='out_file',help='Destination to save the output',required=True)
    parser_db.add_argument('-db_file',dest='db_file',help='Validation data set to classify',default=None)

    parser_model = subparsers.add_parser('model_files', help='prints the list of model files')

    print(argv)
    args = parser.parse_args(argv)
    if args.view is not None:
        args.view = convert(args.view,to_python=True)
    if args.sub_name == 'track':
        if len(args.trx_ids) > 0:
            args.trx_ids = to_py(args.trx_ids)
        args.start_frame = to_py(args.start_frame)
        args.crop_loc = to_py(args.crop_loc)

    net_type =  get_net_type(args.lbl_file)
    if net_type is not None:
        args.type = net_type
    return args

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

    nviews = int(read_entry(H['cfg']['NumViews']))

    if args.sub_name == 'train':
        train(lbl_file, nviews, name, args)

    elif args.sub_name == 'track' and args.list_file is not None:

        # KB 20190123: added list_file input option
        assert args.mov is None, 'Input list_file should specify movie files'
        assert nviews == 1 or args.view is not None, 'View must be specified for multiview projects'
        assert args.trx is None, 'Input list_file should specify trx files'
        assert args.crop_loc is None, 'Input list_file should specify crop locations'
        
        if args.view is None:
            ivw = 0
        else:
            ivw = args.view-1
        if type(args.model_file) is not list:
            args.model_file = [args.model_file]

        conf = create_conf(lbl_file, ivw, name, net_type=args.type, 
                           cache_dir=args.cache,conf_params=args.conf_params)
        success,pred_locs = classify_list_file(conf, args.type, args.list_file, args.model_file[0], args.out_files[0])
        assert success, 'Error classifying list_file ' + args.list_file

    elif args.sub_name == 'track':

        if args.view is None:
            assert len(args.mov) == nviews, 'Number of movie files should be same number of views'
            assert len(args.out_files) == nviews, 'Number of out files should be same as number of views'
            if args.trx is None:
                args.trx = [None] * nviews
            else:
                assert len(args.mov) == len(args.trx), 'Number of movie files should be same as the number of trx files'
            if args.crop_loc is not None:
                assert len(
                    args.crop_loc) == 4 * nviews, 'cropping location should be specified as xlo xhi ylo yhi for all the views'
            views = range(nviews)
        else:
            if args.trx is None:
                args.trx = [None]
            assert len(args.mov) == 1, 'Number of movie files should be one when view is specified'
            assert len(args.trx) == 1, 'Number of trx files should be one when view is specified'
            assert len(args.out_files) == 1, 'Number of out files should be one when view is specified'
            if args.crop_loc is not None:
                assert len(args.crop_loc) == 4, 'cropping location should be specified as xlo xhi ylo yhi'
            views = [args.view]

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache,conf_params=args.conf_params)
            if args.crop_loc is not None:
                crop_loc = [int(x) for x in args.crop_loc]
                # crop_loc = np.array(crop_loc).reshape([len(views), 4])[view_ndx, :] - 1
                # KB 20190123: crop_loc was being decremented twice, removed one
                crop_loc = np.array(crop_loc).reshape([len(views), 4])[view_ndx, :]
            else:
                crop_loc = None

            classify_movie_all(args.type,
                               conf=conf,
                               mov_file=args.mov[view_ndx],
                               trx_file=args.trx[view_ndx],
                               out_file=args.out_files[view_ndx],
                               start_frame=args.start_frame,
                               end_frame=args.end_frame,
                               skip_rate=args.skip,
                               trx_ids=args.trx_ids,
                               name=name,
                               save_hmaps=args.hmaps,
                               crop_loc=crop_loc,
                               model_file=args.model_file,
                               train_name=args.train_name
                               )

    elif args.sub_name == 'gt_classify':
        if args.view is None:
            views = range(nviews)
        else:
            views = [args.view]

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache,conf_params=args.conf_params)
            out_file = args.out_file + '_{}.mat'.format(view)
            classify_gt_data(args.type, conf, out_file, model_file=args.model_file)

    elif args.sub_name == 'data_aug':
        if args.view is None:
            views = range(nviews)
        else:
            views = [args.view]

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache,
                               conf_params=args.conf_params)
            out_file = args.out_file + '_{}.mat'.format(view)
            distort = not args.no_aug
            get_augmented_images(conf,out_file,distort,args.use_cache,nsamples=args.nsamples)

    elif args.sub_name == 'classify':
        if args.view is None:
            views = range(nviews)
        else:
            views = [args.view]

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache, conf_params=args.conf_params)
            out_file = args.out_file + '_{}.mat'.format(view)
            if args.db_file is not None:
                db_file = args.db_file
            else:
                if args.type in ['mdn','unet','openpose']:
                    val_filename = conf.valfilename + '.tfrecords'
                elif args.type == 'leap':
                    val_filename = 'leap_val.h5'
                elif args.type == 'deeplabcut':
                    val_filename = 'val_data.p'
                else:
                    raise ValueError('Unrecognized net type')
                db_file = os.path.join(conf.cachedir, val_filename)
            preds, locs, info = classify_db_all(args.type, conf, db_file, model_file=args.model_file)
            # A = convert_to_orig_list(conf,preds,locs, info)
            # info = to_mat(info)
            # preds, locs = to_mat(A)
            hdf5storage.savemat(out_file, {'pred_locs': preds, 'labeled_locs': locs, 'list':info},appendmat=False,truncate_existing=True)

    elif args.sub_name == 'model_files':
        if args.view is None:
            views = range(nviews)
        else:
            views = [args.view]

        m_files = []
        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, net_type=args.type, cache_dir=args.cache, conf_params=args.conf_params)
            m_files.append(get_latest_model_files(conf,net_type=args.type,name=args.train_name))
        print(m_files)

def main(argv):
    args = parse_args(argv)

    log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(message)s')

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
