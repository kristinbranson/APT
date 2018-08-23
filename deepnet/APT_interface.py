from __future__ import division
from __future__ import print_function

import argparse
import collections
import datetime
import json
import logging
from os.path import expanduser
from random import sample

import PoseUNet
import hdf5storage
import imageio
import multiResData
from multiResData import *
import leap.training
from leap.training import train_apt as leap_train
import open_pose
from deepcut.train import train as deepcut_train
import deepcut.train

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    From: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True, appendmat=False)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

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
    elif len(db_files)>1:
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

    out_fns = []
    out_fns.append(lambda data: envs[0].write(tf_serialize(data)))
    out_fns.append(lambda data: envs[1].write(tf_serialize(data)))
    if use_cache:
        splits = db_from_cached_lbl(conf,out_fns, split, split_file, on_gt)
    else:
        splits = db_from_lbl(conf, out_fns, split, split_file, on_gt)

    envs[0].close()
    envs[1].close() if split else None
    try:
        with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
            json.dump(splits, f)
    except IOError:
        logging.warning('SPLIT_WRITE: Could not output the split data information')



def convert_to_orig(base_locs, conf, cur_trx, trx_fnum_start, all_f, sz, nvalid, crop_loc):
    # converts locs in cropped image back to locations in original image.
    if conf.has_trx_file:
        hsz_p = conf.imsz[0] / 2  # half size for pred
        base_locs_orig = np.zeros(base_locs.shape)
        for ii in range(nvalid):
            trx_fnum = trx_fnum_start + ii
            x = int(round(cur_trx['x'][0, trx_fnum])) - 1
            y = int(round(cur_trx['y'][0, trx_fnum])) - 1
            # -1 for 1-indexing in matlab and 0-indexing in python
            theta = cur_trx['theta'][0, trx_fnum]
            assert conf.imsz[0] == conf.imsz[1]
            tt = -theta - math.pi / 2
            R = [[np.cos(tt), -np.sin(tt)], [np.sin(tt), np.cos(tt)]]
            curlocs = np.dot(base_locs[ii, :, :] - [hsz_p, hsz_p], R) + [x, y]
            base_locs_orig[ii, ...] = curlocs
    else:
        xlo, xhi, ylo, yhi = crop_loc
        base_locs_orig = base_locs.copy()
        base_locs_orig[:, :, 0] += xlo
        base_locs_orig[:, :, 1] += ylo

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

def write_hmaps(hmaps, hmaps_dir, trx_ndx, frame_num):
    for bpart in range(hmaps.shape[-1]):
        cur_out = os.path.join(hmaps_dir, 'hmap_trx_{}_t_{}_part_{}.jpg'.format(trx_ndx + 1, frame_num + 1, bpart + 1))
        cur_im = hmaps[:, :, bpart]
        cur_im = ((np.clip(cur_im, -1, 1) * 128) + 128).astype('uint8')
        imageio.imwrite(cur_out, cur_im, 'jpg', quality=75)
        # cur_out_png = os.path.join(hmaps_dir,'hmap_trx_{}_t_{}_part_{}.png'.format(trx_ndx+1,frame_num+1,bpart+1))
        # imageio.imwrite(cur_out_png,cur_im)

def create_conf(lbl_file, view, name, net_type='unet', cache_dir=None):
    try:
        try:
            H = loadmat(lbl_file)
        except NotImplementedError:
            print('Label file is in v7.3 format. Loading using h5py')
            H = h5py.File(lbl_file, 'r')
    except TypeError as e:
        logging.exception('LBL_READ: Could not read the lbl file {}'.format(lbl_file))

    from poseConfig import config
    conf = config()
    conf.n_classes = int(read_entry(H['cfg']['NumLabelPoints']))
    proj_name = read_string(H['projname']) + '_view{}'.format(view)
    conf.view = view
    conf.set_exp_name(proj_name)
    # conf.cacheDir = read_string(H['cachedir'])
    dt_params_ndx = None
    for ndx in range(H['trackerClass'].shape[0]):
        cur_tracker = ''.join([chr(c) for c in H[H['trackerClass'][ndx][0]]])
        if cur_tracker == 'DeepTracker':
            dt_params_ndx = ndx

    dt_params = H[H['trackerData'][dt_params_ndx][0]]['sPrm']
    if cache_dir is None:
        conf.cachedir = os.path.join(read_string(dt_params['CacheDir']), proj_name, name)
    else:
        conf.cachedir = cache_dir

    if not os.path.exists(os.path.split(conf.cachedir)[0]):
        os.mkdir(os.path.split(conf.cachedir)[0])
    # conf.cachedir = os.path.join(localSetup.bdir, 'cache', proj_name)
    conf.has_trx_file = has_trx_file(H[H['trxFilesAll'][0, 0]])
    conf.selpts = np.arange(conf.n_classes)

    # If the project has trx file then we use the crop locs specified by the user. If the project doesnt have trx files then we use the crop size specified by user else use the whole frame.
    if conf.has_trx_file:
        width = int(read_entry(dt_params['sizex']))
        height = int(read_entry(dt_params['sizey']))
        conf.imsz = (height, width)
    else:
        if H['cropProjHasCrops'][0,0] == 1:
            xlo, xhi, ylo, yhi = PoseTools.get_crop_loc(H,0,view)
            conf.imsz = (int(yhi-ylo+1), int(xhi-xlo+1))
        else:
            vid_nr = int(read_entry(H[H['movieInfoAll'][0, 0]]['info']['nr']))
            vid_nc = int(read_entry(H[H['movieInfoAll'][0, 0]]['info']['nc']))
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
    conf.adjustContrast = int(read_entry(dt_params['adjustContrast'])) > 0.5
    conf.normalize_img_mean = int(read_entry(dt_params['normalize'])) > 0.5
    # conf.imgDim = int(read_entry(dt_params['NChannels']))
    ex_mov = multiResData.find_local_dirs(conf)[0][0]

    if 'NumChans' in H['cfg'].keys():
        conf.imgDim = read_entry(H['cfg']['NumChans'])
    else:
        cap = movies.Movie(ex_mov, interactive=False)
        ex_frame = cap.get_frame(0)
        if np.ndim(ex_frame) > 2:
            conf.imgDim = ex_frame[0].shape[2]
        else:
            conf.imgDim = 1
        cap.close()
    try:
        conf.flipud = int(read_entry(dt_params['flipud'])) > 0.5
    except KeyError:
        pass
    try:
        conf.unet_steps = int(read_entry(dt_params['dl_steps']))
        conf.dl_steps = int(read_entry(dt_params['dl_steps']))
        #conf.dlc_steps = int(read_entry(dt_params['dl_steps']))
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
            mm = re.search('(\d+)\s+(\d+)',b)
            n1 = int(mm.groups()[0]) - 1
            n2 = int(mm.groups()[1]) - 1
            graph.append([n1,n2])
        conf.op_affinity_graph = graph
    except KeyError:
        pass

    if net_type == 'openpose':
        # openpose uses its own normalization
        conf.normalize_img_mean = False

    return conf


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
    lbl = h5py.File(conf.labelfile,'r')
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
            trx = sio.loadmat(trx_files[ndx])['trx'][0]
            n_trx = len(trx)
            trx_split = np.random.random(n_trx) < conf.valratio
        else:
            trx = [None]
            n_trx = 1
            trx_split = None
            cur_pts = cur_pts[np.newaxis, ...]

        for trx_ndx in range(n_trx):

            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx, on_gt)
            cur_trx = trx[trx_ndx]
            for fnum in frames:
                if not check_fnum(fnum, cap, exp_name, ndx):
                    continue

                info = [ndx, fnum, trx_ndx]
                cur_out = multiResData.get_cur_env(out_fns, split, conf, info,
                                                   mov_split, trx_split=trx_split, predefined=predefined)

                frame_in, cur_loc = multiResData.get_patch(
                    cap, fnum, conf, cur_pts[trx_ndx, fnum, :, sel_pts], cur_trx=cur_trx, flipud=flipud, crop_loc=crop_loc)
                cur_out([frame_in, cur_loc, info])

                if cur_out is out_fns[1] and split:
                    val_count += 1
                    splits[1].append(info)
                else:
                    count += 1
                    splits[0].append(info)

        cap.close()  # close the movie handles
        print('Done %d of %d movies, train count:%d val count:%d' % (ndx + 1, len(local_dirs), count, val_count))

    print('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
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
    lbl = h5py.File(conf.labelfile,'r')
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

    m_ndx = lbl['preProcData_MD_mov'].value[0,:].astype('int')
    t_ndx = lbl['preProcData_MD_iTgt'].value[0,:].astype('int') - 1
    f_ndx = lbl['preProcData_MD_frm'].value[0,:].astype('int') - 1

    if on_gt:
        m_ndx = -m_ndx

    if conf.has_trx_file:
        trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)

    prev_trx_mov = -1
    psz = max(conf.imsz)

    for ndx in range(lbl['preProcData_I'].shape[1]):
        if m_ndx[ndx] < 0: continue
        mndx = m_ndx[ndx] - 1
        cur_pts = trx_pts(lbl, mndx, on_gt)
        if cur_pts.ndim == 3:
            cur_pts = cur_pts[np.newaxis,...]
        crop_loc = PoseTools.get_crop_loc(lbl, mndx, view, on_gt)
        cur_locs = cur_pts[t_ndx[ndx], f_ndx[ndx], :, sel_pts].copy()
        cur_frame = lbl[lbl['preProcData_I'][conf.view,ndx]].value.copy()
        cur_frame = cur_frame.T

        if cur_frame.ndim == 2:
            cur_frame = cur_frame[...,np.newaxis]

        if conf.has_trx_file:

            # dont load trx file if the current movie is same as previous.
            # and trx split wont work well if the frames for the same animal are not contiguous
            if prev_trx_mov == mndx:
                cur_trx = trx[t_ndx[ndx]]
            else:
                trx = sio.loadmat(trx_files[mndx])['trx'][0]
                cur_trx = trx[t_ndx[ndx]]
                prev_trx_mov = mndx
                n_trx = len(trx)
                trx_split = np.random.random(n_trx) < conf.valratio

            x, y, theta = read_trx(cur_trx, f_ndx[ndx])
            ll = cur_locs.copy()
            ll = ll - [x, y]
            rot = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            lr = np.dot(ll, rot) + [psz / 2, psz / 2]
            if conf.imsz[0] < conf.imsz[1]:
                extra = (psz-conf.imsz[0])/2
                lr[:,1] -= extra
            elif conf.imsz[1] < conf.imsz[0]:
                extra = (psz-conf.imsz[1])/2
                lr[:,0] -= extra
            cur_locs = lr

        else:
            trx_split = None
            n_trx = 1
            if crop_loc is not None:
                xlo, xhi, ylo, yhi = crop_loc
            else:
                xlo = 0; ylo = 0

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

    print('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
    lbl.close()
    return splits


def create_leap_db(conf, split=False, split_file=None, use_cache=False):
    # function showing how to use db_from_lbl for tfrecords
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)

    train_data = []
    val_data = []

    # collect the images and labels in arrays
    out_fns = []
    out_fns.append(lambda data: train_data.append(data))
    out_fns.append(lambda data: val_data.append(data))
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
        if conf.imgDim == 1:
            im = data[0][:,:,0]
        else:
            im = data[0]
        img_name = os.path.join(outdir,'img_{:06d}.png'.format(count[0]))
        imageio.imwrite(img_name, im)
        locs = data[1]
        bparts = conf.n_classes
        for b in range(bparts):
            fis[b].write('{}\t{}\t{}\n'.format(count[0],locs[b,0],locs[b,1]))
        mod_locs = np.insert(np.array(locs),0,range(bparts),axis=1)
        save_data.append([img_name, im.shape, mod_locs])
        count[0] += 1

    bparts = ['part_{}'.format(i) for i in range(conf.n_classes)]
    train_count = [0]
    train_dir = os.path.join(conf.cachedir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_fis = [open(os.path.join(train_dir,b+'.csv'),'w') for b in bparts]
    train_data = []
    val_count = [0]
    val_dir = os.path.join(conf.cachedir, 'val')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    val_fis= [open(os.path.join(val_dir,b+'.csv'),'w') for b in bparts]
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
    with open(os.path.join(conf.cachedir,'train_data.p'),'w') as f:
        pickle.dump(train_data,f,protocol=2)
    if split:
        with open(os.path.join(conf.cachedir,'val_data.p'),'w') as f:
            pickle.dump(val_data,f,protocol=2)

    # save the split data
    try:
        with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
            json.dump(splits, f)
    except IOError:
        logging.warning('SPLIT_WRITE: Could not output the split data information')


def create_cv_split_files(conf, n_splits=3):
    # creates json files for the xv splits
    local_dirs, _ = multiResData.find_local_dirs(conf)
    lbl = h5py.File(conf.labelfile,'r')

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
            cur_trx_info = list(zip(mm, tt, frames.tolist()))
            trx_info.append(cur_trx_info)
            cur_mov_info.extend(cur_trx_info)
            n_labeled_frames += frames.size
        mov_info.append(cur_mov_info)
    lbl.close()

    lbls_per_fold = n_labeled_frames / n_splits

    imbalance = True
    for retry in range(10):
        per_fold = np.zeros([n_splits])
        splits = [[] for i in range(n_splits)]

        if conf.splitType is 'movie':
            for ndx in range(len(local_dirs)):
                valid_folds = np.where(per_fold < lbls_per_fold)[0]
                cur_fold = np.random.choice(valid_folds)
                splits[cur_fold].extend(mov_info[ndx])
                per_fold[cur_fold] += len(mov_info[ndx])

        elif conf.splitType is 'trx':
            for tndx in range(len(trx_info)):
                valid_folds = np.where(per_fold < lbls_per_fold)[0]
                cur_fold = np.random.choice(valid_folds)
                splits[cur_fold].extend(trx_info[tndx])
                per_fold[cur_fold] += len(trx_info[tndx])

        elif conf.splitType is 'frames':
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
    all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))
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
    if conf.has_trx_file:
        T = sio.loadmat(trx_file)['trx'][0]
        n_trx = len(T)
        end_frames = np.array([x['endframe'][0, 0] for x in T])
        first_frames = np.array([x['firstframe'][0, 0] for x in T]) - 1  # for converting from 1 indexing to 0 indexing
    else:
        T = [None, ]
        n_trx = 1
        end_frames = np.array([n_frames])
        first_frames = np.array([0])
    return T, first_frames, end_frames, n_trx

def get_trx_ids(trx_ids_in, n_trx, has_trx_file):
    if has_trx_file:
        if len(trx_ids_in) == 0:
            trx_ids = np.arange(n_trx)
        else:
            trx_ids = np.array(trx_ids_in)
    else:
        trx_ids = [0]
    return trx_ids


def classify_list(conf, pred_fn, cap, to_do_list, trx_file, crop_loc):
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
        all_f = create_batch_ims(to_do_list[cur_start:(cur_start+ppe)], conf,
                                 cap, flipud, trx, crop_loc)

        base_locs, hmaps = pred_fn(all_f)
        base_locs = base_locs + 1  # for matlabs 1 - indexing

        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            trx_ndx = cur_entry[1]
            cur_trx = trx[trx_ndx]
            cur_f = cur_entry[0]
            trx_fnum_start = cur_f - first_frames[trx_ndx]
            base_locs_orig = convert_to_orig(base_locs[cur_t:cur_t + 1, ...], conf, cur_trx, trx_fnum_start, all_f, sz, 1, crop_loc)
            pred_locs[cur_start + cur_t, :, :] = base_locs_orig[0, ...]

    return pred_locs


def get_pred_fn(model_type, conf, model_file=None):
    if model_type == 'openpose':
        pred_fn, close_fn, model_file = open_pose.get_pred_fn(conf, model_file)
    elif model_type == 'unet':
        pred_fn, close_fn, model_file = get_unet_pred_fn(conf, model_file)
    elif model_type == 'leap':
        pred_fn, close_fn, model_file = leap.training.get_pred_fn(conf, model_file)
    elif model_type == 'deeplabcut':
        pred_fn, close_fn, model_file = deepcut.train.get_pred_fn(conf, model_file)
    else:
        raise ValueError('Undefined type of model')

    return pred_fn, close_fn, model_file

def classify_list_all(model_type, conf, in_list, on_gt, model_file):
    # in_list should be of list of type [mov_file, frame_num, trx_ndx]
    # all of them should be 1-indexed.

    pred_fn, close_fn, model_file = get_pred_fn(model_type, conf, model_file)

    if on_gt:
        local_dirs, _ = multiResData.find_gt_dirs(conf)
    else:
        local_dirs, _ = multiResData.find_local_dirs(conf)

    lbl = h5py.File(conf.labelfile,'r')
    view = conf.view
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    if conf.has_trx_file:
        trx_files = multiResData.get_trx_files(lbl, local_dirs)
    else:
        trx_files = [None,]*len(local_dirs)

    pred_locs = np.zeros([len(in_list), conf.n_classes, 2])
    pred_locs[:] = np.nan

    for ndx, dir_name in enumerate(local_dirs):

        cur_list = [ [l[1]-1, l[2]-1] for l in in_list if l[0] == (ndx+1)]
        cur_idx = [i for i,l in enumerate(in_list) if l[0]==(ndx+1)]
        crop_loc = PoseTools.get_crop_loc(lbl, ndx, view, on_gt)

        try:
            cap = movies.Movie(dir_name)
        except ValueError:
            logging.exception('MOVIE_READ: ' + local_dirs[ndx] + ' is missing')
            exit(1)

        cur_pred_locs = classify_list(conf, pred_fn, cap, cur_list, trx_files[ndx], crop_loc)
        pred_locs[cur_idx,...]= cur_pred_locs

        cap.close()  # close the movie handles

    lbl.close()
    close_fn()
    return pred_locs


def classify_db(conf, read_fn, pred_fn, n):
    bsize = conf.batch_size
    all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))
    pred_locs = np.zeros([n, conf.n_classes, 2])
    n_batches = int(math.ceil(float(n) / bsize))
    labeled_locs = np.zeros([n, conf.n_classes, 2])
    info = []
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            all_f[ndx,...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])
        base_locs, hmaps = pred_fn(all_f)
        for ndx in range(ppe):
            pred_locs[cur_start + ndx, ...] = base_locs[ndx,...]

    return pred_locs, labeled_locs, info



def classify_db_all(model_type, conf, db_file, model_file=None):
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
    elif model_type == 'leap':
        leap_gen, n = leap.training.get_read_fn(conf, db_file)
        pred_fn, latest_model_file = leap.training.get_pred_fn(conf, model_file)
        pred_locs, label_locs, info = classify_db(conf, leap_gen, pred_fn, n)
    elif model_type == 'deeplabcut':
        read_fn, n = deepcut.train.get_read_fn(conf, db_file)
        pred_fn, latest_model_file = deepcut.train.get_pred_fn(conf, model_file)
        pred_locs, label_locs, info = classify_db(conf, read_fn, pred_fn, n)
    else:
        raise ValueError('Undefined model type')

    return pred_locs, label_locs, info


def check_train_db(model_type, conf, out_file):
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
    samples = np.linspace(0,n-1,n_out).astype('int')
    all_f = np.zeros((n_out,) + conf.imsz + (conf.imgDim,))
    labeled_locs = np.zeros([n_out, conf.n_classes, 2])
    count = 0
    info = []
    for cur_b in range(n):
        next_db = read_fn()
        if cur_b in samples:
            all_f[count,...] = next_db[0]
            labeled_locs[count, ...] = next_db[1]
            info.append(next_db[2])
            count += 1

    with open(out_file,'w') as f:
        pickle.dump({'ims':all_f, 'locs': labeled_locs, 'info':np.array(info)},f,protocol=2)


def classify_gt_data(conf, model_type, out_file, model_file):
    local_dirs, _ = multiResData.find_gt_dirs(conf)
    lbl = h5py.File(conf.labelfile,'r')
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
            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx,on_gt=True)
            for f in frames:
                cur_list.append([ndx+1, f+1, trx_ndx+1])
                labeled_locs.append(cur_pts[trx_ndx, f, :, sel_pts])

    pred_locs = classify_list_all(model_type, conf, cur_list, on_gt=True, model_file=model_file)
    mat_pred_locs = pred_locs + 1
    mat_labeled_locs = np.array(labeled_locs) +1
    mat_list = cur_list
#    mat_list = [[m+1, f+1, t+1 ] for m,f,t in cur_list]

    sio.savemat(out_file,{'pred_locs':mat_pred_locs,
                          'labeled_locs': mat_labeled_locs,
                          'list': mat_list})
    lbl.close()
    return pred_locs, labeled_locs, cur_list


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
    # classifies movies frame by frame instead of trx by trx.

    def write_trk(pred_locs_in, n_done):
        # pred_locs is the predicted locations of size
        # n_frames in the movie x n_Trx x n_body_parts x 2
        # n_done is the number of frames that have been tracked.
        pred_locs = pred_locs_in.copy()
        pred_locs = pred_locs[:, trx_ids, ...]
        pred_locs = pred_locs.transpose([2, 3, 0, 1])
        pred_locs = pred_locs[:, :, n_done, :]
        tgt = trx_ids + 1  # target animals that have been tracked.
        # For projects without trx file this is always 1.
        if not conf.has_trx_file:
            pred_locs = pred_locs[..., 0]
        ts_shape = pred_locs.shape[0:1] + pred_locs.shape[2:]
        ts = np.ones(ts_shape) * datetime2matlabdn() # time stamp
        tag = np.zeros(ts.shape).astype('bool') # tag which is always false for now.
        tracked_shape = pred_locs.shape[2]
        tracked = np.zeros([1, tracked_shape]) # which of the predlocs have been tracked. Mostly to help APT know how much tracking has been done.
        tracked[0, :] = np.array(n_done) + 1
        info = {} # tracking info. Can be empty.
        info[u'model_file'] = model_file
        info[u'trnTS'] = get_matlab_ts(model_file + '.meta')
        info[u'name'] = name
        param_dict = convert_unicode(conf.__dict__.copy())
        param_dict.pop('cropLoc', None)
        info[u'params'] = param_dict
        hdf5storage.savemat(out_file,
                            {'pTrk': pred_locs, 'pTrkTS': ts, 'expname': mov_file, 'pTrkiTgt': tgt,
                             'pTrkTag': tag, 'pTrkFrm': tracked, 'trkInfo': info},
                            appendmat=False, truncate_existing=True)

    cap = movies.Movie(mov_file)
    sz = (cap.get_height(), cap.get_width())
    n_frames = int(cap.get_n_frames())
    T, first_frames, end_frames, n_trx = get_trx_info(trx_file, conf, n_frames)
    trx_ids = get_trx_ids(trx_ids, n_trx, conf.has_trx_file)
    bsize = conf.batch_size
    flipud = conf.flipud

    if end_frame < 0: end_frame = end_frames.max()
    if end_frame > end_frames.max(): end_frame = end_frames.max()
    if start_frame > end_frame: return None

    max_n_frames = end_frames.max() - first_frames.min()
    min_first_frame = first_frames.min()
    pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
    pred_locs[:] = np.nan

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
        all_f = create_batch_ims(to_do_list[cur_start:(cur_start+ppe)], conf,
                                 cap, flipud, T, crop_loc)

        base_locs, hmaps = pred_fn(all_f)
        base_locs = base_locs + 1  # for matlabs 1 - indexing
        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            trx_ndx = cur_entry[1]
            cur_trx = T[trx_ndx]
            cur_f = cur_entry[0]
            trx_fnum_start = cur_f - first_frames[trx_ndx]
            base_locs_orig = convert_to_orig(base_locs[cur_t:cur_t + 1, ...], conf, cur_trx, trx_fnum_start, all_f, sz, 1, crop_loc)
            pred_locs[cur_f - min_first_frame, trx_ndx, :, :] = base_locs_orig[0, ...]

            if save_hmaps:
                write_hmaps(hmaps[cur_t, ...], hmap_out_dir, trx_ndx, cur_f)

        if cur_b % 20 == 19:
            sys.stdout.write('.')
        if cur_b % 400 == 399:
            sys.stdout.write('\n')
            write_trk(pred_locs, range(start_frame, to_do_list[cur_start][0]))

    write_trk(pred_locs, range(start_frame, end_frame))
    cap.close()
    tf.reset_default_graph()
    return pred_locs


def get_unet_pred_fn(conf, model_file=None):

    tf.reset_default_graph()
    self = PoseUNet.PoseUNet(conf)
    try:
        sess, latest_model_file = self.init_net_meta(1, model_file)
    except tf.errors.InternalError:
        logging.exception(
            'Could not create a tf session. Probably because the CUDA_VISIBLE_DEVICES is not set properly')
        sys.exit(1)

    def pred_fn(all_f):
        # this is the function that is used for classification.
        # this should take in an array B x H x W x C of images, and
        # output an array of predicted locations.
        # predicted locations should be B x N x 2
        # PoseTools.get_pred_locs can be used to convert heatmaps into locations.

        bsize = conf.batch_size
        xs, _ = PoseTools.preprocess_ims(
            all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf,
            distort=False, scale=self.conf.unet_rescale)

        self.fd[self.ph['x']] = xs
        self.fd[self.ph['phase_train']] = False
        # self.fd[self.ph['keep_prob']] = 1.
        try:
            pred = sess.run(self.pred, self.fd)
        except tf.errors.ResourceExhaustedError:
            logging.exception('Out of GPU Memory. Either reduce the batch size or increase unet_rescale')
            exit(1)
        base_locs = PoseTools.get_pred_locs(pred)
        base_locs = base_locs * conf.unet_rescale
        return base_locs, pred

    def close_fn():
        sess.close()
        self.close_cursors()

    return pred_fn, close_fn, latest_model_file


def classify_movie_all(model_type, **kwargs):
    conf = kwargs['conf']
    model_file = kwargs['model_file']
    pred_fn, close_fn, model_file = get_pred_fn(model_type, conf, model_file)
    try:
        classify_movie(conf, pred_fn, model_file=model_file, **kwargs)
    except (IOError, ValueError) as e:
        close_fn()
        logging.exception('Could not track movie')


def train_unet(conf, args):
    if not args.skip_db:
        create_tfrecord(conf, False, use_cache=args.use_cache)
    tf.reset_default_graph()
    self = PoseUNet.PoseUNet(conf)
    self.train_data_name = 'traindata'
    self.train_unet(False, 1)


def train_leap(conf, args):
    if not args.skip_db:
        create_leap_db(conf, False, use_cache=args.use_cache)
    leap_train(conf)


def train_openpose(conf,args):
    if not args.skip_db:
        create_tfrecord(conf, False, use_cache=args.use_cache)
    open_pose.training(conf)


def train_deepcut(conf, args):
    if not args.skip_db:
        create_deepcut_db(conf, False, use_cache=args.use_cache)
    deepcut_train(conf)


def train(lblfile, nviews, name, args):
    view = args.view
    type = args.type
    if view is None:
        views = range(nviews)
    else:
        views = [view]

    for cur_view in views:
        conf = create_conf(lblfile, cur_view, name, cache_dir=args.cache)
        if args.cache is not None:
            conf.cachedir = args.cache

        conf.view = cur_view

        try:
            if type == 'unet':
                train_unet(conf, args)
            elif type == 'openpose':
                if args.use_defaults:
                    open_pose.set_openpose_defaults(conf)
                train_openpose(conf,args)
            elif type == 'leap':
                if args.use_defaults:
                    leap.training.set_leap_defaults(conf)
                train_leap(conf, args)
            elif type == 'deeplabcut':
                if args.use_defaults:
                    deepcut.train.set_deepcut_defaults(conf)
                deepcut_train(conf)
        except tf.errors.InternalError as e:
            logging.exception(
                'Could not create a tf session. Probably because the CUDA_VISIBLE_DEVICES is not set properly')
            exit(1)
        except tf.errors.ResourceExhaustedError as e:
            logging.exception('Out of GPU Memory. Either reduce the batch size or increase unet_rescale')
            exit(1)


# def classify(lbl_file, n_views, name, mov_file, trx_file, out_file, start_frame, end_frame, skip_rate):
#     # print(mov_file)
#
#     for view in range(n_views):
#         conf = create_conf(lbl_file, view, name)
#         tf.reset_default_graph()
#         self = PoseUNet.PoseUNet(conf)
#         sess = self.init_net_meta(0, True)
#
#         cap = movies.Movie(mov_file[view], interactive=False)
#         n_frames = int(cap.get_n_frames())
#         height = int(cap.get_height())
#         width = int(cap.get_width())
#         cur_end_frame = end_frame
#         if end_frame < 0:
#             cur_end_frame = n_frames
#         cap.close()
#
#         if trx_file is not None:
#             pred_list = self.classify_movie_trx(mov_file[view], trx_file[view], sess, end_frame, start_frame)
#             temp_predScores = pred_list[1]
#             predScores = -1 * np.ones((n_frames,) + temp_predScores.shape[1:])
#             # predScores[start_frame:cur_end_frame, ...] = temp_predScores
#             # dummy predscores for now.
#
#             temp_pred_locs = pred_list
#             pred_locs = np.nan * np.ones((n_frames,) + temp_pred_locs.shape[1:])
#             pred_locs[start_frame:cur_end_frame, ...] = temp_pred_locs
#             pred_locs = pred_locs.transpose([2, 3, 0, 1])
#             tgt = np.arange(pred_locs.shape[-1]) + 1
#         else:
#             pred_list = self.classify_movie(mov_file[view], sess, end_frame, start_frame)
#
#             rescale = conf.unet_rescale
#             orig_crop_loc = conf.cropLoc[(height, width)]
#             crop_loc = [int(x / rescale) for x in orig_crop_loc]
#             end_pad = [int((height - conf.imsz[0]) / rescale) - crop_loc[0],
#                        int((width - conf.imsz[1]) / rescale) - crop_loc[1]]
#
#             pp = [(0, 0), (crop_loc[0], end_pad[0]), (crop_loc[1], end_pad[1]), (0, 0)]
#             temp_predScores = np.pad(pred_list[1], pp, mode='constant', constant_values=-1.)
#             predScores = np.zeros((n_frames,) + temp_predScores.shape[1:])
#             predScores[start_frame:cur_end_frame, ...] = temp_predScores
#
#             temp_pred_locs = pred_list[0]
#             temp_pred_locs[:, :, 0] += orig_crop_loc[1]
#             temp_pred_locs[:, :, 1] += orig_crop_loc[0]
#             pred_locs = np.nan * np.ones((n_frames,) + temp_pred_locs.shape[1:])
#             pred_locs[start_frame:cur_end_frame, ...] = temp_pred_locs
#             pred_locs = pred_locs.transpose([1, 2, 0])
#             tgt = np.arange(1) + 1
#
#         ts_shape = pred_locs.shape[0:1] + pred_locs.shape[2:]
#         ts = np.ones(ts_shape) * datetime2matlabdn()
#         tag = np.zeros(ts.shape).astype('bool')
#         hdf5storage.savemat(out_file + '_view_{}'.format(view) + '.trk',
#                             {'pTrk': pred_locs, 'pTrkTS': ts, 'expname': mov_file[view], 'pTrkiTgt': tgt,
#                              'pTrkTag': tag}, appendmat=False, truncate_existing=True)
#

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("lbl_file",
                        help="path to lbl file")
    parser.add_argument('-name', dest='name', help='Name for the run. Default - pose_unet', default='pose_unet')
    parser.add_argument('-view', dest='view', help='Run only for this view. If not specified, run for all views', default=None, type=int)
    parser.add_argument('-model_file', dest='model_file', help='Use this model file for prediction instead of the latest model file', default=None)
    parser.add_argument('-cache', dest='cache', help='Override cachedir in lbl file', default=None)
    parser.add_argument('-out_dir', dest='out_dir', help='Directory to output log files', default=None)
    parser.add_argument('-type', dest='type', help='Network type, default is unet', default='unet',
                        choices=['unet', 'openpose','deeplabcut','leap'])
    subparsers = parser.add_subparsers(help='train or track or gt_classify', dest='sub_name')

    parser_train = subparsers.add_parser('train', help='Train the detector')
    parser_train.add_argument('-skip_db', dest='skip_db', help='Skip creating the data base', action='store_true')
    parser_train.add_argument('-use_defaults',dest='use_defaults',action='store_true', help='Use default settings of openpose, deeplabcut or leap')
    parser_train.add_argument('-use_cache',dest='use_cache',action='store_true', help='Use cached images in the label file to generate the training data.')
    # parser_train.add_argument('-cache',dest='cache_dir',
    #                           help='cache dir for training')

    parser_classify = subparsers.add_parser('track', help='Track a movie')
    parser_classify.add_argument("-mov", dest="mov",
                                 help="movie(s) to track", required=True, nargs='+')
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
    parser_classify.add_argument('-crop_loc', dest='crop_loc', help='crop location given xlo xhi ylo yhi', nargs='*', default=None)

    parser_gt = subparsers.add_parser('gt_classify', help='Classify GT labeled frames')
    parser_gt.add_argument('-out',dest='out_file_gt',help='Mat file to save output to. _[view_num].mat will be appended',required=True)

    print(argv)
    args = parser.parse_args(argv)
    if args.view is not None:
        args.view = args.view - 1
    if args.sub_name == 'track':
        if len(args.trx_ids) > 0:
            args.trx_ids = [t - 1 for t in args.trx_ids]
        args.start_frame = args.start_frame - 1
    return args


def run(args):
    name = args.name

    lbl_file = args.lbl_file
    try:
        try:
            H = loadmat(lbl_file)
        except NotImplementedError:
            print('Label file is in v7.3 format. Loading using h5py')
            H = h5py.File(lbl_file,'r')
    except TypeError as e:
        logging.exception('LBL_READ: Could not read the lbl file {}'.format(lbl_file))
        exit(1)

    nviews = int(read_entry(H['cfg']['NumViews']))

    if args.sub_name == 'train':
        train(lbl_file, nviews, name, args)

    elif args.sub_name == 'track':

        if args.view is None:
            assert len(args.mov) == nviews, 'Number of movie files should be same number of views'
            assert len(args.out_files) == nviews, 'Number of out files should be same as number of views'
            if args.trx is None:
                args.trx = [None] * nviews
            else:
                assert len(args.mov) == len(args.trx), 'Number of movie files should be same as the number of trx files'
            if args.crop_loc is not None:
                assert len(args.crop_loc)==4*nviews, 'cropping location should be specified as xlo xhi ylo yhi for all the views'
            views = range(nviews)
        else:
            if args.trx is None:
                args.trx = [None]
            assert len(args.mov) == 1, 'Number of movie files should be one when view is specified'
            assert len(args.trx) == 1, 'Number of trx files should be one when view is specified'
            assert len(args.out_files) == 1, 'Number of out files should be one when view is specified'
            if args.crop_loc is not None:
                assert len(args.crop_loc)==4*nviews, 'cropping location should be specified as xlo xhi ylo yhi'
            views = [args.view]

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, cache_dir=args.cache)
            if args.cache is not None:
                conf.cachedir = args.cache
            if args.crop_loc is not None:
                crop_loc = np.array(args.crop_loc).reshape([len(views), 4])[view_ndx,:]
            else:
                crop_loc = None

            classify_movie_all(args.type,
                               conf= conf,
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
                               model_file=args.model_file
                               )

    elif args.sub_name == 'gt_classify':
        if args.view is None:
            views = range(nviews)
        else:
            views = [args.view]

        for view_ndx, view in enumerate(views):
            conf = create_conf(lbl_file, view, name, cache_dir=args.cache)
            out_file = args.out_file + '_{}.mat'.format(view)
            classify_gt_data(args.type, conf, out_file, model_file=args.model_file)

def main(argv):
    args = parse_args(argv)

    if args.out_dir is not None:
        assert os.path.exists(args.out_dir), 'Output directory doesnt exist'
        logfile = os.path.join(args.out_dir, '{}.err'.format(args.name))
    else:
        logfile = os.path.join(expanduser("~"), '{}.err'.format(args.name))
    fileh = logging.FileHandler(logfile, 'w')
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(levelname)s:%(message)s -- %(asctime)s')
    log.handlers[0].setFormatter(formatter)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    log.addHandler(consoleHandler)

    try:
        run(args)
    except Exception as e:
        logging.exception('UNKNOWN: APT_interface errored because of some error')


if __name__ == "__main__":
    main(sys.argv[1:])
