import sys
import os
import datetime
import argparse
import logging
import pickle
import glob
import subprocess
import logging
import inspect
import getpass
import random

import tensorflow as tf
import imgaug as ia
import numpy as np
import h5py
import matplotlib.pyplot as plt

import APT_interface as apt
import deepposekit as dpk
import apt_dpk
import PoseTools as pt
import TrainingGeneratorTFRecord as TGTFR
import kerascallbacks
import run_apt_expts_2 as rae
import util
import multiResData as mrd

logr = logging.getLogger('APT')

user = getpass.getuser()
if user == 'leea30':
    dbs = {
        'dpkfly': {'h5dset': '/groups/branson/home/leea30/git/dpkd/datasets/fly/annotation_data_release_AL.h5',
                   'slbl': '/groups/branson/bransonlab/apt/experiments/data/leap_dataset_gt_stripped_numchans1.lbl',
                   }
    }
    alcache = '/groups/branson/bransonlab/apt/dl.al.2020/cache'
    aldeepnet = '/groups/branson/home/leea30/git/apt.aldl/deepnet'
elif user == 'al':
    dbs = {
        'dpkfly': {'h5dset': '/home/al/git/dpkd/datasets/fly/annotation_data_release_AL.h5',
                   'slbl': '/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/leap_dataset_gt_stripped_numchans1.lbl',
                   }
    }
    alcache = '/dat0/apt/cache'
    aldeepnet = '/home/al/git/APT_aldl/deepnet'

'''
def get_rae_normal_conf():
    
    Get 'normal'/base conf from run_apt_exps.

    Also massages/replaces a few "string props" with their numeric/literal versions.
    :return:
    

    importlib.reload(rae)
    rae.setup('alice')  # actual dset shouldn't matter, just a single-view proj
    out = rae.run_normal_training(run_type='dry')
    conf_dict = out['dpk_vw0'][0]

    # special-case massage gah
    conf_dict['brange'] = ast.literal_eval(conf_dict['brange'].replace("\\", ""))
    conf_dict['crange'] = ast.literal_eval(conf_dict['crange'].replace("\\", ""))

    conf = poseConfig.config()
    for k in conf_dict.keys():
        v = conf_dict[k]
        print('Overriding param {} <= {}'.format(k, v))
        setattr(conf, k, conf_dict[k])

    return conf
'''


def split_tfr_proper(tfrsrc, tfrdst0, tfrdst1, n0, npts):
    '''
    Split an existing tfr into two tfrs
    :param tfrsrc: src tfrecords file
    :param tfrdst0: first dst tfrecords file (to be created)
    :param tfrdst1: second dst tfrecords file (to be created)
    :param n0: number of rows to write to first dst (remaining rows are written to second dst)
    :param npts: num kpts or n_classes (for decoding tfr)
    :return:
    '''


    # read the src
    ims, locs, info, occ = mrd.read_and_decode_without_session(tfrsrc,
                                                               npts,
                                                               indices=())

    # gen a split
    n = len(ims)
    # n0 = int(np.round(frac0*n))
    n1 = n - n0
    print("{} orig els. split into {}/{} for 0/1".format(n, n0, n1))
    s0 = set(random.sample(range(n), n0))
    s1 = set(range(n)) - s0

    envs = (tf.python_io.TFRecordWriter(tfrdst0),
            tf.python_io.TFRecordWriter(tfrdst1))

    for i in range(n):
        towrite = apt.tf_serialize([ims[i], locs[i], info[i], occ[i]])
        ienv = int(i in s1)  # 1=set1, 0=set0
        envwrite = envs[ienv]
        envwrite.write(towrite)

        if i % 100 == 99:
            print('Wrote {} rows'.format(i + 1))

    print('Wrote {} rows'.format(i + 1))

def split_tfr_proper_normal_exp(expdir, frac, npts):
    '''
    For 'regular' apt exps

    1. backup train_TF.tfrecords
    2. create a new train_TF.tfrecords with frac*ntrn rows randomly selected
    3. put other rows into val_TF.tfrecords

    :param expdir:
    :param frac: fraction of orig train_TF.tfrecords to put in new train_TF
    :param npts:
    :return:
    '''

    trntfr = os.path.join(expdir, 'train_TF.tfrecords')
    valtfr = os.path.join(expdir, 'val_TF.tfrecords')
    trntfrbak = os.path.join(expdir, 'train_bak_TF.tfrecords')
    assert os.path.exists(trntfr)
    assert not os.path.exists(valtfr)
    assert not os.path.exists(trntfrbak)

    os.rename(trntfr, trntfrbak)
    logr.info("Renamed {}->{}".format(trntfr, trntfrbak))

    ntrn0 = pt.count_records(trntfrbak)
    ntrn = int(np.round(frac * ntrn0))
    logr.info('Orig train had {} rows. New train/val will have {}/{}.'.format(
        ntrn0, ntrn, ntrn0-ntrn))

    split_tfr_proper(trntfrbak, trntfr, valtfr, ntrn, npts)
    logr.info("Wrote new tfrs {} and {}".format(trntfr, valtfr))


def split_tfr_with_rename(trntfr0, valtfr0, ntrnnew, npts):
    '''
    for exp3, holdout-a-proper-dev-set
    :param trntfr0: original train_TF
    :param valtfr0: original val_TF
    :param ntrnnew: how many records to hold in new train_TF

    1. back up train_TF
    2. rename val->test
    3. create new train/val from orig train_TF
    :return:
    '''

    expdir = os.path.dirname(trntfr0)
    tsttfr1 = os.path.join(expdir, 'test_TF.tfrecords')
    trntfrbak = os.path.join(expdir, 'train_bak_TF.tfrecords')
    # valtfrbak = os.path.join(expdir,'val_bak_TF.tfrecords')

    assert not os.path.exists(tsttfr1)
    assert not os.path.exists(trntfrbak)
    # assert not os.path.exists(valtfrbak)

    # orig val becomes new tst
    os.rename(valtfr0, tsttfr1)
    print("Renamed {}->{}".format(valtfr0, tsttfr1))

    # orig trn becomes trnbak`
    os.rename(trntfr0, trntfrbak)
    print("Renamed {}->{}".format(trntfr0, trntfrbak))

    assert not os.path.exists(trntfr0)
    assert not os.path.exists(valtfr0)

    # split orig trn into new trn/val
    split_tfr_proper(trntfrbak, trntfr0, valtfr0, ntrnnew, npts)
    print("Wrote new tfrs {} and {}".format(trntfr0, valtfr0))


def verify_split_tfr(trntfr0, valtfr0, trntfr1, valtfr1, tsttfr1, npts):
    '''
    Verify results of split_tfr_with_rename
    trntfr0: original train.tfrecords
    valtfr0: original val.tfrecords
    trntfr1: new etc
    '''

    def check_ims_locs_ifo_occ(ims0, locs0, ifo0, occ0,
                               ims1, locs1, ifo1, occ1):
        assert np.array_equal(np.concatenate(ims0), np.concatenate(ims1))
        assert np.array_equal(np.concatenate(locs0), np.concatenate(locs1))
        assert np.array_equal(np.concatenate(occ0), np.concatenate(occ1))
        assert ifo0 == ifo1
        print("Checking n={}".format(len(ims0)))

    imst0, locst0, ifot0, occt0 = mrd.read_and_decode_without_session(trntfr0, npts, indices=())
    imsv0, locsv0, ifov0, occv0 = mrd.read_and_decode_without_session(valtfr0, npts, indices=())
    imst1, locst1, ifot1, occt1 = mrd.read_and_decode_without_session(trntfr1, npts, indices=())
    imsv1, locsv1, ifov1, occv1 = mrd.read_and_decode_without_session(valtfr1, npts, indices=())
    imstst1, locstst1, ifotst1, occtst1 = mrd.read_and_decode_without_session(tsttfr1, npts, indices=())

    check_ims_locs_ifo_occ(imsv0, locsv0, ifov0, occv0,
                           imstst1, locstst1, ifotst1, occtst1)
    print("Verified that old val is new tst")

    # combine trn1 and val1
    ims1tot = imst1 + imsv1
    locs1tot = locst1 + locsv1
    ifo1tot = ifot1 + ifov1
    occ1tot = occt1 + occv1
    idxsorted = sorted(range(len(ifo1tot)), key=lambda k: ifo1tot[k])
    ims1totS = [ims1tot[x] for x in idxsorted]
    locs1totS = [locs1tot[x] for x in idxsorted]
    ifo1totS = [ifo1tot[x] for x in idxsorted]
    occ1totS = [occ1tot[x] for x in idxsorted]

    # old trn isnow trn/val
    check_ims_locs_ifo_occ(imst0, locst0, ifot0, occt0,
                           ims1totS, locs1totS, ifo1totS, occ1totS)
    print("Verified that old trn is now new trn/val combined")


def create_callbacks_exp1orig_train(conf):
    logr.info("configing callbacks")

    # `Logger` evaluates the validation set( or training set if `validation_split = 0` in the `TrainingGenerator`) at the end of each epoch and saves the evaluation data to a HDF5 log file( if `filepath` is set).
    nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    # logfile = 'log{}.h5'.format(nowstr)
    # logger = deepposekit.callbacks.Logger(
    #                 filepath=os.path.join(conf.cachedir, logfile),
    #                 validation_batch_size=10)

    '''
    ppr: patience=10, min_delta=.001
    step3_train_model.ipynb: patience=20, min_delta=1e-4 (K dflt)
    Guess prefer the ipynb for now, am thinking it is 'ahead'
    '''
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",  # monitor="val_loss"
        factor=0.2,
        verbose=1,
        patience=20,
    )

    # `ModelCheckpoint` automatically saves the model when the validation loss improves at the end of each epoch. This allows you to automatically save the best performing model during training, without having to evaluate the performance manually.
    ckptfile = 'ckpt{}.h5'.format(nowstr)
    ckpt = os.path.join(conf.cachedir, ckptfile)
    model_checkpoint = dpk.callbacks.ModelCheckpoint(
        ckpt,
        monitor="val_loss",  # monitor="val_loss"
        verbose=1,
        save_best_only=True,
    )

    # Ppr: patience=50, min_delta doesn't really say, but maybe suggests 0 (K dflt)
    # step3_train_model.ipynb: patience=100, min_delta=.001
    # Use min_delta=0.0 here it is more conservative
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # monitor="val_loss"
        min_delta=0.0,
        patience=100,
        verbose=1
    )

    callbacks = [reduce_lr, model_checkpoint, early_stop]
    return callbacks


def create_callbacks_exp2orig_train(conf,
                                    sdn,
                                    valbsize,
                                    nvalbatch,
                                    runname='deepnet',
                                    ):
    '''
    This is a "standard" DPK-style train
    :param conf:
    :param sdn:
    :param valbsize:
    :param nvalbatch:
    :param runname:
    :return:
    '''

    if conf.dpk_reduce_lr_style == 'ppr':
        lr_patience = 10
        lr_min_delta = .001
    elif conf.dpk_reduce_lr_style == 'ipynb':
        lr_patience = 20
        lr_min_delta = 1e-4
    else:
        assert False
    logr.info('dpk_lr_style: {}'.format(conf.dpk_reduce_lr_style))
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        verbose=1,
        patience=lr_patience,
        min_delta=lr_min_delta,
    )

    nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    ckptfile = 'ckpt{}.h5'.format(nowstr)
    ckpt = os.path.join(conf.cachedir, ckptfile)
    model_checkpoint = dpk.callbacks.ModelCheckpoint(
        ckpt,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )

    ckpt_reg = 'cpkt{}'.format(nowstr)
    ckpt_reg += '-{epoch: 05d}-{val_loss: .2f}.h5'
    model_checkpoint_reg = tf.keras.callbacks.ModelCheckpoint(
        ckpt_reg,
        save_freq=conf.save_step,  # save every this many batches
    )

    if conf.dpk_early_stop_style == 'ppr':
        es_patience = 50
        es_min_delta = 0.0
    elif conf.dpk_early_stop_style == 'ipynb':
        es_patience = 100
        es_min_delta = .001
    else:
        # "original" exp2, pre 20200616
        #es_patience = 100
        #es_min_delta = 0.0
        assert False
    logr.info('dpk_early_stop_style: {}'.format(conf.dpk_early_stop_style))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # monitor="val_loss"
        min_delta=es_min_delta,
        patience=es_patience,
        verbose=1
    )

    logfile = 'trn{}.log'.format(nowstr)
    logfile = os.path.join(conf.cachedir, logfile)
    loggercbk = tf.keras.callbacks.CSVLogger(logfile)

    tgtfr = sdn.train_generator
    dsval_kps = tgtfr(n_outputs=1,
                      batch_size=valbsize,
                      validation=True,
                      confidence=False,
                      infinite=False)
    logfilevdist = 'trn{}.vdist.log'.format(nowstr)
    logfilevdist = os.path.join(conf.cachedir, logfilevdist)
    logfilevdistlong = 'trn{}.vdist.pickle'.format(nowstr)
    logfilevdistlong = os.path.join(conf.cachedir, logfilevdistlong)
    vdistcbk = kerascallbacks.ValDistLogger(dsval_kps,
                                            logfilevdist,
                                            logfilevdistlong,
                                            nvalbatch)

    cbks = [reduce_lr, model_checkpoint, model_checkpoint_reg,
            loggercbk, early_stop, vdistcbk]
    return cbks


def update_conf_rae(conf):
    '''
    set/update RAE-related steps for apt-style train
    :param conf:
    :return:
    '''

    raeconf = rae.common_conf
    for k in raeconf:
        setattr(conf, k, raeconf[k])


def checkattr_with_warnoverride(conf, prop, val):
    val0 = getattr(conf, prop)
    if not val0 == val:
        logr.warning("Overriding conf.{}, using value={}".format(prop, val))
    return val


def exp1orig_create_base_conf(expname, cacheroot, dset):
    slbl = dbs[dset]['slbl']
    expname = expname if expname else 'dpkorig'
    NET = 'dpksdn'
    conf = apt.create_conf(slbl, 0, expname,
                           cacheroot, NET, quiet=False)
    return conf

def exp1orig_train(expname,
                   dset,
                   cacheroot,
                   runname='deepnet',
                   expname_trnvalsplit=None, # exp should exist 'alongside' expname; conf.pickle in this exp used
                                             #  to set val_index and train_index
                   valbsize=10,
                   shortdebugrun=False,
                   ):

    iaver = ia.__version__
    dpkver = dpk.__version__
    assert iaver == '0.2.9', "Your imgaug version is {}".format(iaver)
    assert dpkver == '0.3.4.dev', "Your dpk version is {}".format(dpkver)

    h5dset = dbs[dset]['h5dset']

    dg = dpk.io.DataGenerator(h5dset)

    # create an APT conf. this is used for
    # - file/cache locs
    # - params (mostly if not all dpk_* params)
    #
    # For the leapfly, the projname in the slbl is 'leap_dset'
    # Note though that the DPK version of the leap dset has
    # locs off-by-one

    conf = exp1orig_create_base_conf(expname, cacheroot, dset)
    # this conf-updating is actually prob unnec but prob doesnt hurt
    conf = apt_dpk.update_conf_dpk(conf,
                                   dg.graph,
                                   dg.swap_index,
                                   n_keypoints=dg.n_keypoints,
                                   imshape=dg.compute_image_shape(),
                                   useimgaug=True,
                                   imgaugtype=dset)
    apt_dpk.print_dpk_conf(conf)

    iaaug = apt_dpk.make_imgaug_augmenter('dpkfly', dg)

    VALSPLIT = 0.1
    assert conf.dpk_downsample_factor == 2
    assert conf.dpk_input_sigma == 5.0
    assert conf.dpk_graph_scale == 1.0
    tg = dpk.TrainingGenerator(generator=dg,
                               downsample_factor=conf.dpk_downsample_factor,
                               augmenter=iaaug,
                               use_graph=True,
                               shuffle=True,
                               sigma=conf.dpk_input_sigma,
                               validation_split=VALSPLIT,
                               graph_scale=conf.dpk_graph_scale,
                               random_seed=0)
    if expname_trnvalsplit is not None and len(expname_trnvalsplit) > 0:
        expparent = os.path.dirname(conf.cachedir)
        edir_trnvalsplit = os.path.join(expparent, expname_trnvalsplit)
        picf = os.path.join(edir_trnvalsplit, 'conf.pickle')
        pic0 = pt.pickle_load(picf)

        val_index_new = pic0['tg']['val_index']
        trn_index_new = pic0['tg']['train_index']
        assert len(val_index_new) == len(tg.val_index)
        assert len(trn_index_new) == len(tg.train_index)
        tg.val_index = val_index_new
        tg.train_index = trn_index_new
        tg.val_index.sort()
        tg.train_index.sort()
        logr.info("read train/val idxs ({}/{}) from {}".format(len(trn_index_new),
                                                               len(val_index_new),
                                                               picf))

    assert conf.dpk_n_stacks == 2
    assert conf.dpk_growth_rate == 48
    dpk_use_pretrained = checkattr_with_warnoverride(conf, 'dpk_use_pretrained', True)
    sdn = dpk.models.StackedDenseNet(tg,
                                     n_stacks=conf.dpk_n_stacks,
                                     growth_rate=conf.dpk_growth_rate,
                                     pretrained=dpk_use_pretrained,
                                     )

    callbacks = create_callbacks_exp1orig_train(conf)

    # compile
    '''
    We trained our models (Figure 2) using mean squared error loss optimized using the 
    ADAM optimizer (Kingma and Ba, 2014) with a learning rate of 1 × 10-3 and a batch size of 16.
    '''
    DECAY = 0.0  # LR modulated via callback
    assert conf.dpk_base_lr_factory == .001
    optimizer = tf.keras.optimizers.Adam(
        lr=conf.dpk_base_lr_factory, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=DECAY, amsgrad=False)
    sdn.compile(optimizer=optimizer, loss='mse')

    # fit
    tgconf = tg.get_config()
    sdnconf = sdn.get_config()
    conf_file = os.path.join(conf.cachedir, 'conf.pickle')
    with open(conf_file, 'wb') as fh:
        pickle.dump({'conf': conf, 'tg': tgconf, 'sdn': sdnconf}, fh)
    logr.info("Saved confs to {}".format(conf_file))

    bsize = checkattr_with_warnoverride(conf, 'batch_size', 16)
    #VALBSIZE = bsize # 10  # step3 ipynb
    logr.info("your valbsize is {}".format(valbsize))
    if shortdebugrun:
        logr.warning('SHORT DEBUG RUN!!')
        EPOCHS = 100
    else:
        EPOCHS = 1000
    sdn.fit(
        batch_size=bsize,
        validation_batch_size=valbsize,
        callbacks=callbacks,
        epochs=EPOCHS,
        steps_per_epoch=None,  # validation_steps=VALSTEPS,
        verbose=2
    )


def simple_dpk_generator(dg, indices, bsize, ):
    '''
    bare-bones data generator from DataGenerator
    :param dg: dpk DataGenerator
    :param indices: list of indices to produce
    :param bsize:
    :return: generator fn, yields ims, locs, idx
    '''

    ngen = len(indices)
    print("simple gen, n_dg={}, n={}".format(len(dg), ngen))

    igen0 = 0
    while igen0 < ngen:
        igen1 = min(igen0 + bsize, ngen)
        nshort = igen0 + bsize - igen1
        idx = indices[igen0:igen1]
        X, y = dg[idx]
        yield X, y, idx
        igen0 += bsize

    return


def simple_tgtfr_val_kpt_generator(conf, bsize):
    '''
    Adaptor
    :param conf:
    :param bsize:
    :return:
    '''

    conf.batch_size = bsize
    tgtfr = TGTFR.TrainingGeneratorTFRecord(conf)
    g = tgtfr(batch_size=bsize, validation=True, shuffle=False,
              confidence=False, infinite=False, debug=True)
    while True:
        ims, tgts, locs, info = next(g)
        assert len(ims) == 1
        ims = ims[0]
        assert np.array_equal(tgts, locs)
        info = info[:, 0].copy()
        yield ims, locs, info


def exp1orig_assess_set(expnamebase,
                        runrange=range(5),
                        dset='dpkfly',
                        cacheroot=alcache,
                        runpat='{}_run{}',
                        **kwargs):
    eresall = []
    for run in runrange:
        expname = runpat.format(expnamebase, run)
        eres = exp1orig_assess(expname, dset, cacheroot, **kwargs)
        eresall.append(eres)

    euc_coll_ptiles50s = np.vstack([x['euc_coll_ptiles5090'][:, 0] for x in eresall]).T
    euc_coll_ptiles90s = np.vstack([x['euc_coll_ptiles5090'][:, 1] for x in eresall]).T

    return eresall, euc_coll_ptiles50s, euc_coll_ptiles90s


def get_latest_ckpt_h5(expdir):
    cpth5 = glob.glob(os.path.join(expdir, 'ckpt*h5'))
    cpth5.sort()
    if len(cpth5) == 0:
        print("No ckpts found in {}".format(expdir))
    elif len(cpth5) > 1:
        print("Warning: more than one ckpt found. Using last one, {}".format(cpth5[-1]))

    cpth5 = cpth5[-1]
    return cpth5


def exp1orig_assess(expname,
                    dset='dpkfly',
                    cacheroot=alcache,
                    validxs=None,  # normally read from conf.pickle
                    bsize=16,
                    doplot=True,
                    gentype='tgtfr',
                    useaptcpt=False,  # if true, use most recent deepnet-xxxxx cpt
                    returnsdn=False,  # if true, early-return with just loaded sdn ready to predict
                    net='dpksdn',
                    ):
    h5dset = dbs[dset]['h5dset']
    slbl = dbs[dset]['slbl']

    dg = dpk.io.DataGenerator(h5dset)

    # make a conf just to get the path to the expdir
    expname = expname if expname else 'dpkorig'
    conf = apt.create_conf(slbl, 0, expname,
                           cacheroot, net, quiet=False)
    # this conf-updating is actually prob unnec but prob doesnt hurt
    conf = apt_dpk.update_conf_dpk(conf,
                                   dg.graph,
                                   dg.swap_index,
                                   n_keypoints=dg.n_keypoints,
                                   imshape=dg.compute_image_shape(),
                                   useimgaug=True,
                                   imgaugtype=dset)

    expdir = conf.cachedir

    if useaptcpt:
        cpt = pt.get_latest_model_file_keras(conf, 'deepnet')
        sdn, conf_saved, _ = apt_dpk.load_apt_cpkt(expdir, cpt)

        print("conf vs conf_saved:")
        util.dictdiff(conf, conf_saved, logr.info)
    else:
        cpth5 = get_latest_ckpt_h5(expdir)
        if gentype == 'dg':
            loadmodelgen = None  # dg
        elif gentype == 'tgtfr':
            loadmodelgen = None
        else:
            assert False
        try:
            sdn = dpk.models.load_model(cpth5, generator=loadmodelgen)
        except KeyError:
            if loadmodelgen is not None:
                print("Warning: load_model failed with non-None gentype. trying with loadmodelgen=none")
                sdn = dpk.models.load_model(cpth5, generator=None)
            else:
                raise
        conf_saved_f = os.path.join(expdir, 'deepnet.conf.pickle')
        conf_saved = pt.pickle_load(conf_saved_f)
        print("conf vs conf_saved:")
        util.dictdiff(conf, conf_saved['conf'], logr.info)

    '''
    The TG is randomly initted at creation/load_model time I think so the various
    val_index train_index etc will not be preserved.
    
    assert sdn.train_generator.generator.datapath == pic['tg']['datapath']
    for f in ['val_index', 'index', 'train_index',]:
        assert np.array_equal( getattr(sdn.train_generator, f), pic['tg'][f] ), \
            "mismatch in field {}".format(f)
    '''

    if returnsdn:
        return sdn, conf_saved, conf

    validxs_specified = validxs is not None

    if gentype == 'tgtfr':
        if validxs_specified:
            logr.info("Ignoring validxs spec; reading val_TF.tfrecords")
        g = simple_tgtfr_val_kpt_generator(conf, bsize)
    elif gentype == 'dg':
        if not validxs_specified:
            logr.info("Reading val idxs from conf.pickle")
            pic = os.path.join(expdir, '*conf.pickle')
            pic = glob.glob(pic)
            assert len(pic) == 1
            logr.info("Found conf.pickle: {}".format(pic[0]))
            pic = pt.pickle_load(pic[0])
            validxs = pic['tg']['val_index']
        g = simple_dpk_generator(dg, validxs, bsize)
    else:
        assert False

    eres = evaluate(sdn.predict_model, g)
    euc_coll, euc_coll_cols, euc_coll_colcnt = \
        collapse_swaps(eres['euclidean'], dg.swap_index)
    eres['euc_coll'] = euc_coll
    eres['euc_coll_cols'] = euc_coll_cols
    eres['euc_coll_colcnt'] = euc_coll_colcnt
    eres['euc_coll_ptiles5090'] = np.percentile(euc_coll, [50, 90], axis=0).T

    nval = eres['euclidean'].shape[0]

    if doplot:
        plt.rcParams.update({'font.size': 26})
        plt.figure()
        plt.boxplot(eres['euc_coll'], labels=eres['euc_coll_cols'])
        plt.title('{} {}. ntst={}'.format(dset, expname, nval))
        plt.xlabel('kpt/pair')
        plt.ylabel('L2err')
        plt.grid(axis='y')

    return eres


def evaluate(predmodel, gen):
    '''

    :param gen: generator object as simple_dpk_generator
    :param batch_size:
    :return:
    '''

    # see dpk BaseModel/evaluate

    y_true_list = []
    y_pred_list = []
    confidence_list = []
    y_error_list = []
    euclidean_list = []
    idx_list = []
    for X, y_true, idxs in gen:
        y_true_list.append(y_true)
        idx_list.append(idxs)

        y_pred = predmodel.predict_on_batch(X)
        confidence_list.append(y_pred[..., -1])
        y_pred_coords = y_pred[..., :2]
        y_pred_list.append(y_pred_coords)

        errors = dpk.utils.keypoints.keypoint_errors(y_true, y_pred_coords)
        y_error, euclidean, mae, mse, rmse = errors
        y_error_list.append(y_error)
        euclidean_list.append(euclidean)

        # note, final batch may be "wrong-sized" but that works fine;
        # in fact generator need not produce constant-sized bches at all.
        logr.info(".")

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    confidence = np.concatenate(confidence_list)
    y_error = np.concatenate(y_error_list)
    euclidean = np.concatenate(euclidean_list)

    evaluation_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_error": y_error,
        "euclidean": euclidean,
        "confidence": confidence,
        "idxs": idx_list
    }

    return evaluation_dict


def collapse_swaps(x0, swap_index):
    # x: [n x nkpt] data arr

    assert x0.ndim == 2

    x = np.copy(x0)

    colkeep = []
    for i, j in enumerate(swap_index):
        if j == -1:
            # this col has no swap partner
            colkeep.append((i, 1))
        elif i < j:
            assert swap_index[j] == i
            x[:, i] += x[:, j]
            colkeep.append((i, 2))

    colkeep, cnt = zip(*colkeep)
    xcollapsed = x[:, colkeep] / np.array(cnt)
    return xcollapsed, colkeep, cnt


def dpkfly_fix_h5(dset, skel):
    h5dset0 = dbs[dset]['h5dset']
    h5dset = os.path.splitext(h5dset0)
    h5dset = h5dset[0] + '_AL' + h5dset[1]

    print("orig h5: {}".format(h5dset0))
    print("new h5: {}".format(h5dset))

    h50 = h5py.File(h5dset0, 'r')
    h5 = h5py.File(h5dset, 'w')

    for k in ['annotated', 'annotations', 'images']:
        x = np.array(h50[k])
        h5.create_dataset(k, data=x)

    h5.create_dataset('skeleton', data=skel)

    h50.close()
    h5.close()


def exp_train_bsub_codegen(expname, dset, cacheroot,  # mandatory apt_dpk_exps args
                           exptype,  # exp1orig_train or exp2orig_train
                           expnote,
                           submit,  # True to actually launch
                           **kwargs  # addnl args to apt_dpk_exps
                           ):
    conf = exp1orig_create_base_conf(expname, cacheroot, dset)
    edir = conf.cachedir

    jobname = expname
    logfile = os.path.join(edir, '{}.log'.format(expname))
    errfile = os.path.join(edir, '{}.err'.format(expname))
    nslots = 2
    queue = 'gpu_any'

    argstr = '--expname {} --dset {} --cacheroot {} {}'.format(
        expname, dset, cacheroot, exptype)
    for k, v in kwargs.items():
        argstr += ' --{} {}'.format(k, v)
    scriptcmd = os.path.join(aldeepnet, 'run_apt_dpk_exps_orig2.sh {}'.format(argstr))

    bsubscript = os.path.join(edir, '{}.bsub.sh'.format(expname))
    expnotefile = os.path.join(edir, 'EXPNOTE')

    ssscript = os.path.join(aldeepnet, '..', 'matlab', 'repo_snapshot.sh')
    ssfile = os.path.join(edir, '{}.aptss'.format(expname))
    sscmd = "{} > {}".format(ssscript, ssfile)

    code = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -J {} -oo {} -eo {} -n{} -W 2160 -gpu "num=1" -q {} "singularity exec --nv -B /groups/branson -B /nrs/branson /misc/local/singularity/branson_allen.simg {}"' '''.format(
        jobname, logfile, errfile, nslots, queue, scriptcmd)

    if submit:
        if not os.path.exists(edir):
            print("making {}".format(edir))
            os.makedirs(edir)
        with open(bsubscript, 'w') as f:
            f.write(code)
            f.write('\n')
        print("Wrote {}".format(bsubscript))
        if expnote:
            with open(expnotefile, 'w') as f:
                f.write(expnote)
                f.write('\n')
            print("Wrote {}".format(expnotefile))
        subprocess.call(sscmd, shell=True)
        print('Wrote apt snapshot: {}'.format(sscmd))
        subprocess.call(code, shell=True)
        print('submitted {}'.format(expname))
    else:
        print(code)


def exp12orig_train_bsub_codegen(
        exptype,  # exp1orig_train or exp2orig_train
        run_dstr,  # eg 'dpkorig_20200512'
        run_range,  # eg range(4)
        run_pat='dpkorig_{}_run{}',
        cacheroot=alcache,
        dset='dpkfly',
        expnote=None,
        submit=False,
        **kwargs  # addnl args apt_dpk_exps. if eg useimgaug, use 0 and 1 NOT False/True
):
    for irun in run_range:
        expname = run_pat.format(run_dstr, irun)
        exp_train_bsub_codegen(expname,
                               dset,
                               cacheroot,
                               exptype,
                               expnote,
                               submit,
                               **kwargs)


def exp2orig_create_tfrs(expname_from, cacheroot, dset, expname=None):
    '''
    Create TFrecord train/val dbs from h5dset, val/trainidxs in conf.pickle from existing exp1orig experximent
    :param expname: destination/new exp
    :param expname_from: existing exp (run with h5 trainingGenerator)
    :param cacheroot:
    :param dset:
    :return:
    '''

    if expname is None:
        expname = expname_from + "_tfr"

    conf_from = exp1orig_create_base_conf(expname_from, cacheroot, dset)
    picf = os.path.join(conf_from.cachedir, 'conf.pickle')
    pic = pt.pickle_load(picf)
    validx0b = pic['tg']['val_index']

    h5dset = dbs[dset]['h5dset']
    dg = dpk.io.DataGenerator(h5dset)

    conf_new = exp1orig_create_base_conf(expname, cacheroot, dset)
    train_tf = os.path.join(conf_new.cachedir, 'train_TF.tfrecords')
    val_tf = os.path.join(conf_new.cachedir, 'val_TF.tfrecords')

    print("using dg reading {}".format(h5dset))
    print("val_idxs from {}".format(picf))
    print("writing to {}, {}".format(train_tf, val_tf))
    apt_dpk.apt_db_from_datagen(dg, train_tf, val_idx=validx0b, val_tf=val_tf)


def exp2orig_train(expname,
                   dset='dpkfly',
                   cacheroot=alcache,
                   runname='deepnet',
                   shortdebugrun=False,
                   returnsdn=False,  # return model right before calling fit()
                   bsize=16,
                   usetfdata=True,
                   useimgaug=True,  # if false, use PoseTools
                   valbsize=10,
                   reduce_lr_style='ipynb',
                   early_stop_style='ipynb',
                   **kwargs
                   ):
    iaver = ia.__version__
    dpkver = dpk.__version__
    assert iaver == '0.2.9', "Your imgaug version is {}".format(iaver)
    assert dpkver == '0.3.4.dev', "Your dpk version is {}".format(dpkver)

    ### create conf

    h5dset = dbs[dset]['h5dset']
    dg = dpk.io.DataGenerator(h5dset)
    conf = exp1orig_create_base_conf(expname, cacheroot, dset)
    # conf.img_dim=1? think unnec now in slbl
    conf = apt_dpk.update_conf_dpk(conf,
                                   dg.graph,
                                   dg.swap_index,
                                   n_keypoints=dg.n_keypoints,
                                   imshape=dg.compute_image_shape(),
                                   useimgaug=useimgaug,
                                   imgaugtype=dset)
    update_conf_rae(conf)

    conf.dpk_use_tfdata = usetfdata
    if not useimgaug:
        assert dset == 'dpkfly'
        exp2_set_posetools_aug_config_leapfly(conf)
    conf.dpk_val_batch_size = valbsize  # not actually used anywhere
    conf.dpk_reduce_lr_style = reduce_lr_style
    conf.dpk_early_stop_style = early_stop_style

    # try to match exp1orig
    conf.batch_size = bsize
    conf.display_step = 84  # ntrn=1350, bsize=16 => batches/epoch ~ 84
    conf.save_step = conf.save_step // 84 * 84
    ### see apt_dpk.train

    tgtfr, sdn = apt_dpk.compile(conf)
    assert tgtfr is sdn.train_generator
    # validation bsize needs to be spec'd for K fit_generator since the val
    # data comes as a generator (with no len() call) vs a K Sequence
    nvalbatch = int(np.ceil(tgtfr.n_validation / valbsize))
    cbks = create_callbacks_exp2orig_train(conf,
                                           sdn,
                                           valbsize,
                                           nvalbatch,
                                           runname=runname)

    apt_dpk.print_dpk_conf(conf)
    if not useimgaug:
        conf.print_dataaug_flds(logr.info)
    conf_tgtfr = tgtfr.conf
    util.dictdiff(conf, conf_tgtfr, logr.info)

    tgconf = tgtfr.get_config()
    sdnconf = sdn.get_config()
    conf_file = os.path.join(conf.cachedir, '{}.conf.pickle'.format(runname))
    with open(conf_file, 'wb') as fh:
        pickle.dump({'conf': conf, 'tg': tgconf, 'sdn': sdnconf}, fh)
    logr.info("Saved confs to {}".format(conf_file))

    logr.info("nval={}, nvalbatch={}, valbsize={}".format(
        tgtfr.n_validation, nvalbatch, valbsize))

    if shortdebugrun:
        logr.warning('SHORT DEBUG RUN!!')
        epochs = 8
        steps_per_epoch = 3
    else:
        epochs = conf.dl_steps // conf.display_step
        steps_per_epoch = conf.display_step

    if returnsdn:
        return sdn

    if conf.dpk_use_tfdata:
        # separate 'manual' fit for tfdata. this mirror sdn.fit()

        dstrn = sdn.train_generator(sdn.n_outputs,
                                    bsize,
                                    validation=False,
                                    confidence=True,
                                    shuffle=True,
                                    infinite=True)
        dsval = sdn.train_generator(sdn.n_outputs,
                                    valbsize,
                                    validation=True,
                                    confidence=True,
                                    shuffle=False,
                                    infinite=False)
        sdn.activate_callbacks(cbks)

        train_model = sdn.train_model

        train_model.fit(dstrn,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=2,
                        callbacks=cbks,
                        validation_data=dsval,
                        validation_steps=nvalbatch,
                        )

    else:
        assert False, "prob non-op due to datagen"
        sdn.fit(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=conf.batch_size,
            verbose=2,
            callbacks=cbks,
            validation_steps=nvalbatch,  # max_queue_size=1,
            validation_batch_size=conf.batch_size,
        )

    '''
    AL20200514. K.Model.fit_generator() issues: 
        validation_data, validation_steps, max_queue_size, use_multiproc, workers
    The val generator does not get called as many times as expected, given the various other args.
    Very confusing, multiple dimensions of oddness. Appears like a decent chance of a bug, or at 
    minimum fit_generator with regular Py generators has the lowest level of support.
    
    Focusing on the call as given above (see also dpk/engine/fit). 
    * With max_queue_size=1, the val generator seems to get called 2 extra steps per epoch (ie 
    two extra batches) than expected.
    * With max_queue_size unspecified (default to 10),  the valgen seems to get called 11 extra
     steps per epoch 
    * => the total calls to valgen is (validation_steps+max_queue_size+1) per epoch.
    * confirmed for various N
    
    It's unclear whether these extra calls to val are used in any computation, or just somehow
    the queue is getting filled up etc. However, in any case this makes moot the notion of making
    the val generator very precise about covering the valset precisely once per nvalbatch next()
    calls.
    
    If we have time later we can investigate, file a bug report, move to tfdata etc. It sounds
    like using K.Sequence (for tfrecords this would mean tfdata) is much preferred and better
    supported. 
    
    For now:
    1. Set max_queue_size=1
    2. Set up the val generator to be infinite. Run it for nvalbatch steps, knowing this 
    may be off by ~2batches per epoch. So the valset may be slightly "overlapped" per epoch
    but presumably this is not a huge deal.
    3. Don't worry about the "last batch" of the val gen etc. 
    
    does max_queue_size hurt us in terms of perf?
    
    New obs. Above was running in Pycharm Console; in raw cmdline run, max_queue_size
    does not seem to affect the num of calls to valgen, but it is still off in that the valgen
    is called once more than expected.
    '''


def exp2_set_posetools_aug_config_leapfly(conf):
    c = conf
    LEAPFLY_IMSZ = 192

    c.adjust_contrast = False
    c.rescale = 1.0

    # flip
    c.horz_flip = True
    c.vert_flip = True
    c.flipLandmarkMatches = apt_dpk.swap_index_to_flip_landmark_matches(c.dpk_swap_index)

    # affine
    c.use_scale_factor_range = True
    c.scale_factor_range = 1.1
    c.rrange = 180
    c.trange = np.round(.05 * LEAPFLY_IMSZ)
    c.check_bounds_distort = True
    # (no shear)

    # adjust ##
    c.brange = [-.001, .001]  # set me?
    c.crange = [-.001, .001]  # set me?
    c.imax = 255.0

    # normalize
    c.normalize_img_mean = False
    c.img_dim = 1
    c.perturb_color = False
    # imax: 255.0
    c.normalize_batch_mean = False


def test():
    logr.debug('debug')
    logr.info('info')
    logr.warning('warn')

def parse_sig(sig):
    for k in sig.parameters:
        p = sig.parameters[k]
        dv = p.default
        print("{}: dv={}".format(k, dv))


def parse_sig_and_add_args(sig, parser, skipargs):
    for k in sig.parameters:
        if k in skipargs:
            continue
        p = sig.parameters[k]
        dv = p.default
        try:
            if isinstance(dv, bool):
                parser.add_argument('--{}'.format(k),
                                    default=dv,
                                    type=int,
                                    help="default={}".format(dv),
                                    )
                print("Added bool arg {}".format(k))
            elif isinstance(dv, int):
                parser.add_argument('--{}'.format(k),
                                    default=dv,
                                    type=int,
                                    help="default={}".format(dv),
                                    )
                print("Added int arg {}".format(k))
            else:
                parser.add_argument('--{}'.format(k),
                                    default=dv,
                                    help="default={}".format(dv),
                                    )
                print("Added arg {}".format(k))
        except argparse.ArgumentError:
            print("Skipping arg {}".format(k))


def parseargs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname',
                        required=True,
                        help='Experiment name/ID')
    parser.add_argument('--dset',
                        choices=['dpkfly'],  # bub obsolete, move to rae
                        default='dpkfly',
                        help='(DPK) dataset name; doubles as projname')
    parser.add_argument('--cacheroot',
                        default=alcache)
    SKIP_ARGS = ['expname', 'dset', 'cacheroot', 'kwargs']
    subparsers = parser.add_subparsers(help='exptype', dest='exptype')
    parser1 = subparsers.add_parser('exp1orig_train')
    sig1 = inspect.signature(exp1orig_train)
    parse_sig_and_add_args(sig1, parser1, SKIP_ARGS)
    parser2 = subparsers.add_parser('exp2orig_train')
    sig2 = inspect.signature(exp2orig_train)
    parse_sig_and_add_args(sig2, parser2, SKIP_ARGS)

    '''
    parser.add_argument('--runname', default='deepnet')
    parser.add_argument('--expname_trnvalsplit', default='')
    parser.add_argument('--valbsize', default=10, type=int)
    parser.add_argument('--debugrun',
                        default=False,
                        action='store_true')
    '''
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parseargs(sys.argv[1:])
    logr.info("args are:")
    logr.info(args)
    argsdict = vars(args)
    if args.exptype == 'exp1orig_train' or args.exptype == 'exp2orig_train':
        trainfcn = globals()[args.exptype]
        trainfcn(**argsdict)
    elif args.exptype == 'test':
        test()
    else:
        assert False
else:
    pass

'''
    h5file = dpk_fly_h5
    dg = deepposekit.io.DataGenerator(h5file)
    cdpk = apt_dpk_conf(dg, cacheroot, 'testproj', 'testexp')
    augmenter = make_augmenter(dg)
    sdn, cbks = train(cdpk, augmenter, compileonly=True)

    import cv2

    im = cv2.imread(isotri)
    loc = isotrilocs
    im = im[np.newaxis, ...]
    loc = loc[np.newaxis, ...]
    (im_lr, locs_lr), (im_ud, locs_ud), (im_lria, locs_lria), (im_udia, locs_udia) = check_flips(im,loc,isotriswapidx)

    PoseTools.show_result(im, range(1), loc, fignum=10, mrkrsz=200)
    PoseTools.show_result(im_udia, range(1), locs_udia, fignum=11, mrkrsz=200)
    PoseTools.show_result(im_lria, range(1), locs_lria, fignum=12, mrkrsz=200)
    '''
