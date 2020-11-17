from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import math
import json
import time
import os
import pickle
import logging
import csv
import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras import backend as K

import deepposekit.utils.keypoints as dpkkpts
from deepposekit.callbacks import ModelCheckpoint

import tfdatagen
import PoseTools


'''
Callback defns and callback-set instantiations for training flavors.
'''

logr = logging.getLogger('APT')



class APTModelCheckpoint(ModelCheckpoint):
    """
    Keras-style ModelCheckpoint callback with APT-style save freq/naming convention.
        (Save every N batches, ckpts named with batch number)

    So dumb.
    https://github.com/tensorflow/tensorflow/issues/38668
    https://github.com/tensorflow/tensorflow/pull/38669
    """

    def __init__(self, ckpt_fullpath_finalmodel, **kwargs):
        '''

        :param ckpt_fullpath: full path for final model
        :param kwargs: include filepath for regular/intermediate models
        '''
        super(ModelCheckpoint, self).__init__(**kwargs)
        # in keras ModelCheckpoint
        # self._samples_seen_since_last_saving = 0
        self._batches_seen = 0
        # self.model = sdn # this will be 'activated'
        self.ckpt_fpn_final = ckpt_fullpath_finalmodel

    def on_batch_end(self, batch, logs=None):
        # Based on keras/callbacks.py

        #print("OBE, batch={}".format(batch))
        logs = logs or {}
        if isinstance(self.save_freq, int):
            dsamp = logs.get('size', 1)
            self._samples_seen_since_last_saving += dsamp
            self._batches_seen += dsamp
            if self._samples_seen_since_last_saving >= self.save_freq:
                self._save_model_apt(self._batches_seen)
                self._samples_seen_since_last_saving = 0

    def on_train_end(self, logs=None):
        self.model.save(self.ckpt_fpn_final, overwrite=True)
        logr.info("Saved final model: {}".format(self.ckpt_fpn_final))

    def _save_model_apt(self, batch):
        filepath = self.filepath.format(batch=batch)
        self.model.save(filepath, overwrite=True)
        logr.info("Saved model: {}".format(filepath))

def create_lr_sched_callback(steps_per_epoch, base_lr, gamma, decaysteps,
                             return_decay_fcn=False):
    logr.info("LR callback: APT fixed sched")

    def lr_decay(epoch0b):
        steps = (epoch0b + 1) * steps_per_epoch
        lrate = base_lr * math.pow(gamma, steps / decaysteps)
        return lrate

    if return_decay_fcn:
        return lr_decay

    lratecbk = LearningRateScheduler(lr_decay)
    return lratecbk


def create_lr_sched_callback_dropstep(num_epochs,
                                      base_lr,
                                      drop_step_fac,
                                      return_decay_fcn=False
                                      ):

    def step_decay(cur_epoch):
        lrate = base_lr if cur_epoch < ((1-drop_step_fac) * num_epochs) \
                        else base_lr/10.0
        return lrate

    if return_decay_fcn:
        return step_decay

    lratecbk = LearningRateScheduler(step_decay)
    return lratecbk



class TrainDataLogger(Callback):
    def __init__(self, conf, dataits, runname='deepnet'):
        # dataits: (trnDG, valDG)
        #
        # trnDG: training generator. should produce y=hmaps, n_outputs=<as training>
        # valDG: val generator. should produce y=locs, n_outputs=1

        self.train_di, self.val_di = dataits
        self.train_info = {}
        self.train_info['step'] = []
        self.train_info['train_dist'] = []
        self.train_info['train_loss'] = []  # scalar loss (dotted with weightvec)
        #self.train_info['train_loss_K'] = []  # scalar loss as reported by K
        #self.train_info['train_loss_full'] = []  # full loss, layer by layer
        #self.train_info['val_dist'] = []
        self.train_info['lr'] = []
        self.config = conf
        self.pred_model = None
        # self.model is training model
        self.save_start = time.time()
        self.runname = runname

    def pass_model(self, dpkmodel):
        #logging.info("Set pred_model on APTKerasCbk")
        self.pred_model = dpkmodel.predict_model

    def on_epoch_end(self, epoch, logs={}):
        iterations_per_epoch = self.config.display_step
        batch_size = self.config.batch_size
        train_model = self.model
        pred_model = self.pred_model
        step = (epoch + 1) * iterations_per_epoch

        '''
        train_x, train_y = next(self.train_di)
        #val_x, val_y = next(self.val_di)

        assert isinstance(train_x, list) and len(train_x) == 1
        #assert isinstance(val_x, list) and len(val_x) == 1
        assert isinstance(train_y, list) and len(train_y) >= 2, \
            "Expect training model to have n_output>=2"
        #assert not isinstance(val_y, list)
        '''
        # training loss
        #train_loss_full = train_model.evaluate(train_x, train_y,
        #                                       batch_size=batch_size,
        #                                       steps=1,
        #                                       verbose=0)
        #[train_loss_K, *train_loss_full] = train_loss_full
        #train_loss = dot(train_loss_full, loss_weights_vec)
        #train_loss = np.nan

        # train_dist: single-batch with argmax pred. whatevs
        '''
        train_out = train_model.predict(train_x, batch_size=batch_size)
        assert len(train_out) == len(train_y)
        npts = self.config.n_classes
        # final output/map, first npts maps are keypoint confs
        tt1 = PoseTools.get_pred_locs(train_out[-1][..., :npts]) - \
              PoseTools.get_pred_locs(train_y[-1][..., :npts])
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        train_dist = np.nanmean(tt1)
        
        TODO: need to run train_di in session for tfdata dataset; 
        otherwise see ade.assess. also, use predict model!
        '''
        # average over batches/pts, *in output space/coords*

        # val dist
        '''
        val_x = val_x[0]
        assert val_x.shape[0] == batch_size
        val_out = pred_model.predict_on_batch(val_x)
        val_out = val_out[:, :, :2]  # cols should be x, y, max/conf
        assert val_y.shape == val_out.shape, "val_y {} val_out {}".format(val_y.shape, val_out.shape)
        _, val_dist, _, _, _ = dpkkpts.keypoint_errors(val_y, val_out)
        # valdist should be in input-space
        assert val_dist.shape == (batch_size, self.config.n_classes)
        val_dist = np.mean(val_dist)  # all batches, all pts
        '''
        #lr = K.eval(train_model.optimizer.lr)

        #self.train_info['val_dist'].append(val_dist)
        self.train_info['train_dist'].append(logs['loss'])
        self.train_info['train_loss'].append(logs['loss'])
        #self.train_info['train_loss_K'].append(train_loss_K)
        #self.train_info['train_loss_full'].append(train_loss_full)
        self.train_info['step'].append(int(step))
        self.train_info['lr'].append(logs['lr'])

        '''
        p_str = ''
        for k in self.train_info.keys():
            lastval = self.train_info[k][-1]
            if k == 'lr':
                p_str += '{:s}:{:.4g} '.format(k, lastval)
            elif isinstance(lastval, list):
                p_str += '{:s}:<list {} els>'.format(k, len(lastval))
            else:
                p_str += '{:s}:{:.2f} '.format(k, lastval)
        logr.info(p_str)
        '''

        conf = self.config
        train_data_file = os.path.join(self.config.cachedir, 'traindata')

        json_data = {}
        for x in self.train_info.keys():
            json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
        with open(train_data_file + '.json', 'w') as json_file:
            json.dump(json_data, json_file)
        with open(train_data_file, 'wb') as td:
            pickle.dump([self.train_info, conf], td, protocol=2)

        '''
        if step % conf.save_step == 0:
            train_model.save(str(os.path.join(
                conf.cachedir, self.runname + '-{}'.format(int(step)))))
        '''

class ValDistLogger(Callback):
    def __init__(self, dsval_kps, logshort, loglong, nbatch_total):
        '''
            dsval_kps: tfdata Dataset that produces (ims, locs). Make sure batchsize
            evenly divides nval. This dsval will be run repeatedly/re-initted via
            hidden K machinery
            
            logshort: filename for summary/txt stats

            loglong: filename for full/binary stats

            nbatch_total: total number of (full) batches in dsval_kps. Could infer this...
        '''

        self.dsval_ims = dsval_kps.map(map_func=lambda ims, locs: ims)
        self.dsval_locs = dsval_kps.map(map_func=lambda ims, locs: locs)
        self.logshort = logshort
        self.loglong = loglong
        self.nbatch = nbatch_total
        self.pred_model = None

        self.metrics = {
            'epoch': [],
            'dallmu': [],
            'dallptl105090': [],
            'dmu': [],
            'dptl105090': [],
        }

    def pass_model(self, dpk_model):
        #logging.info("Set pred_model on APTKerasCbk")
        self.pred_model = dpk_model.predict_model

    def on_epoch_end(self, epoch, logs={}):
        preds = self.pred_model.predict(self.dsval_ims, verbose=0)
        locs = tfdatagen.read_ds_idxed(self.dsval_locs, range(self.nbatch))
        locs = np.concatenate(locs, axis=0)
        predsonly = preds[:, :, :2]
        assert predsonly.shape == locs.shape, \
            "preds.shape ({}) doesn't match locs.shape ({})".format(predsonly.shape, locs.shape)

        d2 = (predsonly - locs) ** 2  # 3rd col of preds is confidence
        d = np.sqrt(np.sum(d2, axis=-1))  # [nval x nkpt]

        PTLS = [10, 50, 90]
        dallmu = np.mean(d)
        dallptl105090 = np.percentile(d, PTLS)
        dmu = np.mean(d, axis=0)  # [nkpt] mean over bches
        dptl105090 = np.percentile(d, PTLS, axis=0)  # [3 x nkpt]

        # total ncols: 1 + 3 + nkpt + 3*nkpt = 4 + 4*nkpt

        # too big for a single giant csv
        # save stuff in running arrays for binary/full log
        self.metrics['epoch'].append(epoch)
        self.metrics['dallmu'].append(dallmu)
        self.metrics['dallptl105090'].append(dallptl105090)
        self.metrics['dmu'].append(dmu)
        self.metrics['dptl105090'].append(dptl105090)

        # log summary/main stats to logshort
        logshort = self.logshort
        tflogexists = os.path.exists(logshort)
        #print("writing {} and {}".format(logshort, self.loglong))
        with open(logshort, 'a') as f:
            writer = csv.writer(f)
            if not tflogexists:
                HEADER = ['epoch', 'vdistallmu', 'vdistall10ptl', 'vdistall50ptl', 'vdistall90ptl']
                writer.writerow(HEADER)
            row = [epoch, dallmu] + list(dallptl105090)
            writer.writerow(row)

        with open(self.loglong, 'wb') as f:
            pickle.dump(self.metrics, f, protocol=2)




def create_callbacks_exp1orig_train(conf):
    # Example callback code adapted from DeepPoseKit by Jake Graving et al
    # https://github.com/jgraving/deepposekit/examples

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
    model_checkpoint = ModelCheckpoint(
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
                                    use_val,
                                    bsize_valdistlogger,
                                    nvalbatch,
                                    runname='deepnet',
                                    ):
    '''
    :param conf:
    :param sdn:
    :param use_val: if true, use val_loss for LR/ES modulation. if false, use loss
    :param bsize_valdistlogger: can be 0 => dont log valdist (eg if no valdist exists)
    :param nvalbatch:
    :param runname:
    :return:
    '''

    monitor_qty = 'val_loss' if use_val else 'loss'

    if conf.dpk_reduce_lr_style == 'ppr':
        lr_patience = 10
        lr_min_delta = .001
    elif conf.dpk_reduce_lr_style == 'ipynb':
        lr_patience = 20
        lr_min_delta = 1e-4
    else:
        assert False
    logr.info('dpk_lr_style: {}, monitor={}'.format(conf.dpk_reduce_lr_style,
                                                    monitor_qty))
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_qty,
        factor=0.2,
        verbose=1,
        patience=lr_patience,
        min_delta=lr_min_delta,
    )

    nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    ckptfile = 'ckpt{}.h5'.format(nowstr)
    ckpt = os.path.join(conf.cachedir, ckptfile)
    model_checkpoint = ModelCheckpoint(
        ckpt,
        monitor=monitor_qty,
        verbose=1,
        save_best_only=True,
    )

    ckpt_reg = 'ckpt{}'.format(nowstr)
    # ckpt_reg += '-{epoch: 05d}-{val_loss: .2f}.h5'
    #
    # don't include val_loss, get KeyError: 'val_loss' I guess bc our save_freq!='epoch'
    # and the metrics get cleared every epoch. note save_freq is in batches, so with
    # save_freq!='epoch' the saving occurs at random points during an epoch. Val metrics
    # are prob computed only at epoch end.
    ckpt_reg += '-{epoch:05d}.h5'
    ckpt_reg = os.path.join(conf.cachedir, ckpt_reg)
    model_checkpoint_reg = ModelCheckpoint(
        ckpt_reg,
        save_freq=conf.save_step,  # save every this many batches
        save_best_only=False,
    )

    if conf.dpk_early_stop_style == 'ppr':
        es_patience = 50
        es_min_delta = 0.0
    elif conf.dpk_early_stop_style == 'ipynb':
        es_patience = 100
        # we have preferred this as it is more conservative. all exp2 runs
        # prior to 20200617 (except *_pprcbks_* have used this)
        es_min_delta = 0.0

        # this is what DPK actually has in its ipynb.
        # es_min_delta = .001
    elif conf.dpk_early_stop_style == 'ipynb2':
        es_patience = 50
        # patience=100 takes forever.
        es_min_delta = 0.0
    else:
        assert False
    logr.info('dpk_early_stop_style: {}, monitor_qty={}'.format(conf.dpk_early_stop_style, monitor_qty))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_qty,
        min_delta=es_min_delta,
        patience=es_patience,
        verbose=1
    )

    logfile = 'trn{}.log'.format(nowstr)
    logfile = os.path.join(conf.cachedir, logfile)
    loggercbk = tf.keras.callbacks.CSVLogger(logfile)

    cbks = [reduce_lr, model_checkpoint, model_checkpoint_reg,
            loggercbk, early_stop]

    if bsize_valdistlogger > 0:
        tgtfr = sdn.train_generator
        dsval_kps = tgtfr(n_outputs=1,
                          batch_size=bsize_valdistlogger,
                          validation=True,
                          confidence=False,
                          infinite=False)
        logfilevdist = 'trn{}.vdist.log'.format(nowstr)
        logfilevdist = os.path.join(conf.cachedir, logfilevdist)
        logfilevdistlong = 'trn{}.vdist.pickle'.format(nowstr)
        logfilevdistlong = os.path.join(conf.cachedir, logfilevdistlong)
        vdistcbk = ValDistLogger(dsval_kps,
                                 logfilevdist,
                                 logfilevdistlong,
                                 nvalbatch)
        cbks.append(vdistcbk)

    return cbks


def create_callbacks_aptsty(conf,
                            sdn,
                            do_val,
                            valbsize,  # only used if do_val==True
                            nvalbatch,  # "
                            runname='deepnet',
                            ):
    '''
    APT-style train
    :param conf:
    :param sdn:
    :param valbsize:
    :param nvalbatch:
    :param runname:
    :return:
    '''

    nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')

    if conf.step_lr:
        logr.info("step_lr is ON. lr_drop_step={}".format(
            conf.lr_drop_step))
        # LR multiplier shuld be applied to dpk_base_lr_used
        lr_cbk = create_lr_sched_callback_dropstep(
            conf.dpk_epochs_used,
            conf.dpk_base_lr_used,
            conf.lr_drop_step
        )
    else:
        logr.info("step_lr is OFF")
        lr_cbk = create_lr_sched_callback(
                    conf.display_step,
                    conf.dpk_base_lr_used,
                    conf.gamma,
                    conf.decay_steps)

    '''
    use regular keras MCpt instead of dpk.MCpt here.
    dpk.ModelCheckPoint derives from k.ModelCheckpoint. Instead of the regular K machinery presumably calling 
    cbk.set_model (eg earlyish during model.fit()), cbk gets manually 'activated'. this sets .model to be the SDN 
    model and not the train_model. the effect is, when the model is saved, it is the sdn.save->save_model. this first 
    calls the regular keras model save on the .train_model, but then adds stuff (traingenerator config info) to the h5. 
    so each h5 has more stuff in it.

    the bug is that in my version of tf/K, the SDN model doesn't have a certain attrib expected of k.Model and so the 
    dpk.ModelCheckPoint cbk harderrs when training ends.
    - soln1: don't use dpk.ModelCkpt; use regular k.ModelCheckpoint. don't 'activate' the cbk. set_model gets called in 
    the normal way at the startish of K.model.fit(). now, the regular train_model.save() gets called which just does a 
    regular K model save w/out the addnl DPK config info stuff. which should be fine.
    - soln2: add a prop _ckpt_saved_epoch property to SDN.model which is always None. This probably fixes the breakage 
    on train_end.
    - I like soln1 more, as we don't really know what _ckpt_saved_epoch might do in other parts of the code. Its 
    probably fine, since we have been running this way successfully, but theoretically there could be some interaction 
    or behavior that is not transparent. Soln1 sticks to raw/vanilla Keras. However this broke our current load codepath.
    - soln3: add custom callback that just saves the model. Hope callbacks exec in order so it executes before the 
    ModelCpt cbk crashes.
    - soln4: overload modelcpt.on_train_end to do nothing
    
    '''


    #ckpt_reg = 'deepnet{}'
    #ckpt_reg = 'deepnet{}'.format(nowstr)
    # ckpt_reg += '-{epoch: 05d}-{val_loss: .2f}.h5'
    #
    # don't include val_loss, get KeyError: 'val_loss' I guess bc our save_freq!='epoch'
    # and the metrics get cleared every epoch. note save_freq is in batches, so with
    # save_freq!='epoch' the saving occurs at random points during an epoch. Val metrics
    # are prob computed only at epoch end.
    ckpt_reg = 'deepnet-{batch:08d}.h5'
    ckpt_reg = os.path.join(conf.cachedir, ckpt_reg)
    ckpt_final = ckpt_reg.format(batch=conf.dl_steps)
    model_checkpoint_reg = APTModelCheckpoint(
        ckpt_final,
        filepath=ckpt_reg,
        save_freq=conf.save_step,  # save every this many batches
        save_best_only=False,
    )

    '''
    class FinalModelSaver(tf.keras.callbacks.Callback):
        def __init__(self, sdn, ckpt_fullpath):
            self.sdn = sdn
            self.ckpt_fpn = ckpt_fullpath

        def on_train_end(self, logs=None):
            self.sdn.save(self.ckpt_fpn)
            logr.info("Saved final model: {}".format(self.ckpt_fpn))

    ckpt_final = ckpt_reg.format(batch=conf.dl_steps)
    final_saver = FinalModelSaver(sdn, ckpt_final)
    logr.info(("Configuring FinalModelSaver with final cpt {}".format(ckpt_final)))
    '''

    #logfile = 'trn{}.log'.format(nowstr)
    #logfile = os.path.join(conf.cachedir, logfile)
    #loggercbk = tf.keras.callbacks.CSVLogger(logfile)
    train_di = sdn.train_generator(sdn.n_outputs,
                                   conf.batch_size,
                                   validation=False,
                                   confidence=True,
                                   shuffle=True,
                                   infinite=True)
    dataits = (train_di, None)
    loggercbk = TrainDataLogger(conf, dataits)

    cbks = [lr_cbk, model_checkpoint_reg, loggercbk]

    if do_val:
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
        vdistcbk = ValDistLogger(dsval_kps,
                                 logfilevdist,
                                 logfilevdistlong,
                                 nvalbatch)
        cbks.append(vdistcbk)

    return cbks


def create_callbacks_OLD(conf, sdn, runname='deepnet'):
    '''

    '''

    logr.warning("configing callbacks")

    # `Logger` evaluates the validation set( or training set if `validation_split = 0` in the `TrainingGenerator`) at the end of each epoch and saves the evaluation data to a HDF5 log file( if `filepath` is set).
    nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    logfile = 'log{}.h5'.format(nowstr)
    '''
    logger = deepposekit.callbacks.Logger(
                    filepath=os.path.join(outtouch,logfile),
                    validation_batch_size=10)
    '''

    assert conf.dpk_reduce_lr_on_plat
    if conf.dpk_reduce_lr_on_plat:
        logr.info("LR callback: using reduceLROnPlateau")
        lr_cbk = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", # monitor="val_loss"
            factor=0.2,
            verbose=1,
            patience=20)
    else:
        lr_cbk = create_lr_sched_callback(
            conf.display_step,
            conf.dpk_base_lr_used,
            conf.gamma,
            conf.decay_steps)

    # training, infinite, shuffling
    train_generator = sdn.train_generator(
        n_outputs=sdn.n_outputs,
        batch_size=conf.batch_size,
        validation=False,
        confidence=True
    )
    # val, infinite, nonshuffled, tgts-as-kpts
    keypoint_generator = sdn.train_generator(
        n_outputs=1,
        batch_size=conf.batch_size,
        validation=True,
        confidence=False,
        infinite=True,  # use "infinite" val generators here, for logging val_dist only
    )
    aptcbk = apt_dpk_callbacks.APTKerasCbk(conf, (train_generator, keypoint_generator),
                                        runname=runname)


    # `ModelCheckpoint` automatically saves the model when the validation loss improves at the end of each epoch. This allows you to automatically save the best performing model during training, without having to evaluate the performance manually.
    '''
    ckptfile = 'ckpt{}.h5'.format(nowstr)
    ckpt = os.path.join(outtouch, ckptfile)
    model_checkpoint = deepposekit.callbacks.ModelCheckpoint(
        ckpt,
        monitor="val_loss", # monitor="val_loss"
        verbose=1,
        save_best_only=True,
    )
    '''

    # `EarlyStopping` automatically stops the training session when the validation loss stops improving for a set number of epochs, which is set with the `patience` argument. This allows you to save time when training your model if there's not more improvment.
    '''
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", # monitor="val_loss"
        min_delta=0.001,
        patience=100,
        verbose=1
    )
    '''

    #callbacks = [early_stop, reduce_lr, model_checkpoint, logger]
    callbacks = [lr_cbk, aptcbk]

    return callbacks


