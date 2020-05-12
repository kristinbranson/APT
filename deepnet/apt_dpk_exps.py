
import sys
import os
import datetime
import argparse
import logging
import pickle
import glob

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


dbs = {
    'dpkfly': {'h5dset': '/groups/branson/home/leea30/git/dpkd/datasets/fly/annotation_data_release_AL.h5',
               'slbl': '/groups/branson/bransonlab/apt/experiments/data/leap_dataset_gt_stripped_numchans1.lbl',
                }
}

alcache = '/groups/branson/bransonlab/apt/dl.al.2020/cache'
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
    logging.info("configing callbacks")

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

def checkattr_with_warnoverride(conf, prop, val):
    val0 = getattr(conf, prop)
    if not val0 == val:
        logging.warning("Overriding conf.{}, using value={}".format(prop, val))
    return val

def exp1orig_train(expname, dset, cacheroot, shortdebugrun=False):

    iaver = ia.__version__
    dpkver = dpk.__version__
    assert iaver == '0.2.9', "Your imgaug version is {}".format(iaver)
    assert dpkver == '0.3.4.dev', "Your dpk version is {}".format(dpkver)

    h5dset = dbs[dset]['h5dset']
    slbl = dbs[dset]['slbl']

    dg = dpk.io.DataGenerator(h5dset)

    # create an APT conf. this is used for
    # - file/cache locs
    # - params (mostly if not all dpk_* params)
    #
    # For the leapfly, the projname in the slbl is 'leap_dset'
    # Note though that the DPK version of the leap dset has
    # locs off-by-one
    expname = 'dpkorig_' + expname if expname else 'dpkorig'
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
                                   imgaugtype=args.dset)
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
        lr=conf.dpk_base_lr_factory, beta_1=0.9, beta_2=0.999,
        decay=DECAY, amsgrad=False)
    sdn.compile(optimizer=optimizer, loss='mse')

    # fit
    tgconf = tg.get_config()
    sdnconf = sdn.get_config()
    conf_file = os.path.join(conf.cachedir, 'conf.pickle')
    with open(conf_file, 'wb') as fh:
        pickle.dump({'conf': conf, 'tg': tgconf, 'sdn': sdnconf}, fh)
    logging.info("Saved confs to {}".format(conf_file))

    bsize = checkattr_with_warnoverride(conf, 'batch_size', 16)
    VALBSIZE = 10  # step3 ipynb

    if shortdebugrun:
        logging.warning('SHORT DEBUG RUN!!')
        EPOCHS = 100
    else:
        EPOCHS = 1000
    sdn.fit(
        batch_size=bsize,
        validation_batch_size=VALBSIZE,
        callbacks=callbacks,
        epochs=EPOCHS,
        steps_per_epoch=None,  # validation_steps=VALSTEPS,
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

def simple_tgtfr_generator(conf, bsize):
    '''
    Adaptor
    :param conf:
    :param bsize:
    :return:
    '''

    conf.batch_size = bsize
    tgtfr = TGTFR.TrainingGeneratorTFRecord(conf)
    g = tgtfr(batch_size=bsize, validation=True,
              confidence=False, infinite=False, debug=True)
    while True:
        ims, tgts, locs, info = next(g)
        assert len(ims) == 1
        ims = ims[0]
        assert np.array_equal(tgts, locs)
        info = info[:, 0].copy()
        yield ims, locs, info

def exp1orig_assess(dset, cacheroot, expname,
                    validxs = None,  # normally read from conf.pickle
                    bsize=16,
                    doplot=True,
                    gentype='tgtfr'):
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
    cpth5 = glob.glob(os.path.join(expdir,'ckpt*h5'))
    cpth5.sort()
    if len(cpth5) > 1:
        print("Warning: more than one ckpt found. Using last one, {}".format(cpth5[-1]))
    cpth5 = cpth5[-1]

    sdn = dpk.models.load_model(cpth5, generator=dg)

    '''
    assert sdn.train_generator.generator.datapath == pic['tg']['datapath']
    for f in ['val_index', 'index', 'train_index',]:
        assert np.array_equal( getattr(sdn.train_generator, f), pic['tg'][f] ), \
            "mismatch in field {}".format(f)
    '''

    validxs_specified = validxs is not None

    if gentype == 'tgtfr':
        if validxs_specified:
            print("Ignoring validxs spec; reading val_TF.tfrecords")
        g = simple_tgtfr_generator(conf, bsize)
    elif gentype == 'dg':
        if not validxs_specified:
            print("Reading val idxs from conf.pickle")
            pic = os.path.join(expdir, 'conf.pickle')
            pic = pt.pickle_load(pic)
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
        logging.info(".")

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

def collapse_swaps(x, swap_index):
    # x: [n x nkpt] data arr

    assert x.ndim == 2

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
                        choices=['exp1orig_train'],
                        default='exp1orig_train',
                        )
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
    if args.exptype == 'exp1orig_train':
        exp1orig_train(args.expname, args.dset, args.cacheroot,
                 shortdebugrun=args.debugrun)
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
