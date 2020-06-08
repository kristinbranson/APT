
import sys
import os
import datetime
import argparse
import logging
import pickle
import glob
import subprocess
import logging

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

logr = logging.getLogger()
logr.setLevel(logging.DEBUG)

dbs = {
    'dpkfly': {'h5dset': '/groups/branson/home/leea30/git/dpkd/datasets/fly/annotation_data_release_AL.h5',
               'slbl': '/groups/branson/bransonlab/apt/experiments/data/leap_dataset_gt_stripped_numchans1.lbl',
                }
}

alcache = '/groups/branson/bransonlab/apt/dl.al.2020/cache'
aldeepnet = '/groups/branson/home/leea30/git/apt.aldl/deepnet'
#alcache = '/dat0/apt/cache'

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
        monitor="val_loss", # monitor="val_loss"
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

def create_callbacks_exp2orig_train(conf, sdn, runname='deepnet'):

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        verbose=1,
        patience=20,
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

    '''
    tg = sdn.train_generator(
        n_outputs=sdn.n_outputs,
        batch_size=conf.batch_size,
        validation=False,
        confidence=True,
        instrumentedname='KCbkTrn'
    )
    vg = sdn.train_generator(
        n_outputs=1,
        batch_size=conf.batch_size,
        validation=True,
        confidence=False,
        infinite=True,  # infinite, only for logging val_dist
        instrumentedname='KCbkVal'
    )
    aptcbk = kerascallbacks.APTKerasCbk(conf, (tg, vg), runname=runname)
    '''

    # Ppr: patience=50, min_delta doesn't really say, but maybe suggests 0 (K dflt)
    # step3_train_model.ipynb: patience=100, min_delta=.001
    # Use min_delta=0.0 here it is more conservative
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # monitor="val_loss"
        min_delta=0.0,
        patience=100,
        verbose=1
    )

    logfile = 'trn{}.log'.format(nowstr)
    logfile = os.path.join(conf.cachedir, logfile)
    loggercbk = tf.keras.callbacks.CSVLogger(logfile)
    cbks = [reduce_lr, model_checkpoint, loggercbk, early_stop]
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

def exp1orig_train(expname, dset, cacheroot,
                   runname='deepnet',
                   expname_trnvalsplit=None, # exp should exist 'alongside' expname; conf.pickle in this exp used
                                             #  to set val_index and train_index
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
    VALBSIZE = bsize # 10  # step3 ipynb

    if shortdebugrun:
        logr.warning('SHORT DEBUG RUN!!')
        EPOCHS = 100
    else:
        EPOCHS = 1000
    sdn.fit(
        batch_size=bsize,
        validation_batch_size=VALBSIZE,
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
        igen1 = min(igen0+bsize, ngen)
        nshort = igen0+bsize-igen1
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
    if len(cpth5) > 1:
        print("Warning: more than one ckpt found. Using last one, {}".format(cpth5[-1]))
    cpth5 = cpth5[-1]
    return cpth5

def exp1orig_assess(expname,
                    dset='dpkfly',
                    cacheroot=alcache,
                    validxs = None,  # normally read from conf.pickle
                    bsize=16,
                    doplot=True,
                    gentype='tgtfr',
                    useaptcpt=False, # if true, use most recent deepnet-xxxxx cpt
                    ):

    h5dset = dbs[dset]['h5dset']
    slbl = dbs[dset]['slbl']

    dg = dpk.io.DataGenerator(h5dset)

    # make a conf just to get the path to the expdir
    expname = expname if expname else 'dpkorig'
    NET = 'dpksdn'
    conf = apt.create_conf(slbl, 0, expname,
                           cacheroot, NET, quiet=False)
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
        util.dictdiff(conf, conf_saved)
    else:
        cpth5 = get_latest_ckpt_h5(expdir)
        if gentype == 'dg':
            loadmodelgen = None # dg
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

    '''
    The TG is randomly initted at creation/load_model time I think so the various
    val_index train_index etc will not be preserved.
    
    assert sdn.train_generator.generator.datapath == pic['tg']['datapath']
    for f in ['val_index', 'index', 'train_index',]:
        assert np.array_equal( getattr(sdn.train_generator, f), pic['tg'][f] ), \
            "mismatch in field {}".format(f)
    '''

    validxs_specified = validxs is not None

    if gentype == 'tgtfr':
        if validxs_specified:
            print("Ignoring validxs spec; reading val_TF.tfrecords")
        g = simple_tgtfr_val_kpt_generator(conf, bsize)
    elif gentype == 'dg':
        if not validxs_specified:
            print("Reading val idxs from conf.pickle")
            pic = os.path.join(expdir, '*conf.pickle')
            pic = glob.glob(pic)
            assert len(pic) == 1
            print("Found conf.pickle: {}".format(pic[0]))
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
    eres['euc_coll_ptiles5090'] = np.percentile(euc_coll, [50,90], axis=0).T

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
    xcollapsed = x[:, colkeep]/np.array(cnt)
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

def exp_train_bsub_codegen(expname, exptype, cacheroot, dset, expnote, submit,
                           expname_trnvalsplit=None):

    conf = exp1orig_create_base_conf(expname, cacheroot, dset)
    edir = conf.cachedir

    jobname = expname
    logfile = os.path.join(edir, '{}.log'.format(expname))
    errfile = os.path.join(edir, '{}.err'.format(expname))
    nslots = 2
    queue = 'gpu_any'

    argstr = '--expname {} --exptype {}'.format(expname, exptype)
    if expname_trnvalsplit is not None:
        argstr += ' --expname_trnvalsplit {}'.format(expname_trnvalsplit)
    scriptcmd = os.path.join(aldeepnet, 'run_apt_dpk_exps_orig2.sh {}'.format(argstr))

    bsubscript = os.path.join(edir, '{}.bsub.sh'.format(expname))
    expnotefile = os.path.join(edir, 'EXPNOTE')

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
        subprocess.call(code, shell=True)
        print('submitted {}'.format(expname))
    else:
        print(code)


def exp1orig_train_bsub_codegen(
        nruns=5,
        cacheroot=alcache,
        dset='dpkfly',
        expnote=None,
        submit=False
):
    nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    for irun in range(nruns):
        expname = nowstr + "_run{}".format(irun)
        exp_train_bsub_codegen(expname, cacheroot, dset, expnote, submit)

def exp2orig_train_bsub_codegen(
        run_dstr,
        run_range,
        run_pat='dpkorig_{}_run{}',
        cacheroot=alcache,
        dset='dpkfly',
        expnote=None,
        submit=False,
    ):
    for irun in run_range:
        expname = run_pat.format(run_dstr,irun)
        exp_train_bsub_codegen(expname, 'exp2orig_train', cacheroot, dset, expnote, submit)



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
                   dset,
                   cacheroot,
                   runname='deepnet',
                   shortdebugrun=False,
                   returnsdn=False,  # return model right before calling fit()
                   bsize=16,
                   usetfdata=True,
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
    # conf.img_dim=1?
    conf = apt_dpk.update_conf_dpk(conf,
                                   dg.graph,
                                   dg.swap_index,
                                   n_keypoints=dg.n_keypoints,
                                   imshape=dg.compute_image_shape(),
                                   useimgaug=True,
                                   imgaugtype=dset)
    update_conf_rae(conf)

    conf.dpk_use_tfdata = usetfdata

    # try to match exp1orig
    conf.batch_size = bsize
    conf.display_step = 84  # ntrn=1350, bsize=16 => batches/epoch ~ 84
    conf.save_step = conf.save_step // 84 * 84
    ### see apt_dpk.train

    tgtfr, sdn = apt_dpk.compile(conf)
    cbks = create_callbacks_exp2orig_train(conf, sdn, runname=runname)

    apt_dpk.print_dpk_conf(conf)

    tgconf = tgtfr.get_config()
    sdnconf = sdn.get_config()
    conf_file = os.path.join(conf.cachedir, '{}.conf.pickle'.format(runname))
    with open(conf_file, 'wb') as fh:
        pickle.dump({'conf': conf, 'tg': tgconf, 'sdn': sdnconf}, fh)
    logr.info("Saved confs to {}".format(conf_file))

    # validation bsize needs to be spec'd for K fit_generator since the val
    # data comes as a generator (with no len() call) vs a K Sequence
    nvalbatch = int(np.ceil(tgtfr.n_validation / conf.batch_size))
    logr.info("nval={}, nvalbatch={}".format(tgtfr.n_validation, nvalbatch))

    if shortdebugrun:
        logr.warning('SHORT DEBUG RUN!!')
        epochs = 8
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
                                    bsize,
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
        sdn.fit(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=conf.batch_size,
            verbose=2,
            callbacks=cbks,
            validation_steps=nvalbatch, # max_queue_size=1,
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





def parseargs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname',
                        required=True,
                        help='Experiment name/ID')
    parser.add_argument('--dset',
                        choices=['dpkfly'],  # bub obsolete, move to rae
                        default='dpkfly',
                        help='(DPK) dataset name; doubles as projname')
    #parser.add_argument('--datasrc',
    #                    choices=['h5', 'tgtfr'],
    #                    default='tgtfr',
    #                    help='Data source/input pipeline. If tgtfr, train/val dbs must be present under expname in cache')
    parser.add_argument('--exptype',
                        choices=['exp1orig_train', 'exp2orig_train'],
                        default='exp1orig_train',
                        )
    parser.add_argument('--runname', default='deepnet')
    parser.add_argument('--expname_trnvalsplit', default='')
    parser.add_argument('--debugrun',
                        default=False,
                        action='store_true')
    # parser.add_argument('--augtype',
    #                     choices=['imgaug', 'posetools'],
    #                     default='posetools',
    #                     help='Imgaug choice, applicable only when traintype==apt')
    parser.add_argument('--cacheroot',
                        default=alcache)
    #parser.add_argument('--compileonly',
    #                    action='store_true',
    #                    help="Don't train just compile")
    #parser.add_argument('--dpkloc',
    #                    default='/groups/branson/home/leea30/git/dpk',
    #                    help='location of dpk repo/dependency')
    #parser.add_argument('--imgaugloc',
    #                    default='/groups/branson/home/leea30/git/imgaug',
    #                    help='location of imgaug repo/dependency')

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parseargs(sys.argv[1:])
    if args.exptype == 'exp1orig_train' or args.exptype == 'exp2orig_train':
        trainfcn = globals()[args.exptype]
        trainfcn(args.expname, args.dset, args.cacheroot,
                 runname=args.runname,
                 shortdebugrun=args.debugrun,
                 expname_trnvalsplit=args.expname_trnvalsplit)
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
