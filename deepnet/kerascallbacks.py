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

from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras import backend as K

import deepposekit.utils.keypoints as dpkkpts

import PoseTools


def create_lr_sched_callback(iterations_per_epoch, base_lr, gamma, decaysteps):

    def lr_decay(epoch0b):
        initial_lrate = base_lr
        steps = (epoch0b + 1) * iterations_per_epoch
        lrate = initial_lrate * math.pow(gamma, steps / decaysteps)
        return lrate

    lratecbk = LearningRateScheduler(lr_decay)
    return lratecbk


class APTKerasCbk(Callback):
    def __init__(self, conf, dataits):
        # trnDI should produce y=hmaps, n_outputs=<as training>
        # val DI here should produce y=locs, n_outputs=1

        self.train_di, self.val_di = dataits
        self.train_info = {}
        self.train_info['step'] = []
        self.train_info['train_dist'] = []
        self.train_info['train_loss'] = []  # scalar loss (dotted with weightvec)
        self.train_info['train_loss_K'] = []  # scalar loss as reported by K
        self.train_info['train_loss_full'] = []  # full loss, layer by layer
        self.train_info['val_dist'] = []
        #self.train_info['val_loss'] = []  # etc
        #self.train_info['val_loss_K'] = []
        #self.train_info['val_loss_full'] = []
        self.train_info['lr'] = []
        self.config = conf
        self.pred_model = None
        # self.model is training model
        self.save_start = time.time()

    def pass_model(self, dpkmodel):
        logging.info("Set pred_model on APTKerasCbk")
        self.pred_model = dpkmodel.predict_model

    def on_epoch_end(self, epoch, logs={}):
        iterations_per_epoch = self.config.display_step
        batch_size = self.config.batch_size
        train_model = self.model
        pred_model = self.pred_model
        step = (epoch + 1) * iterations_per_epoch

        train_x, train_y = next(self.train_di)
        val_x, val_y = next(self.val_di)

        assert isinstance(train_x, list) and len(train_x) == 1
        assert isinstance(val_x, list) and len(val_x) == 1
        assert isinstance(train_y, list) and len(train_y) >= 2, \
            "Expect training model to have n_output>=2"
        assert not isinstance(val_y, list)

        # training loss
        train_loss_full = train_model.evaluate(train_x, train_y,
                                               batch_size=batch_size,
                                               steps=1,
                                               verbose=0)
        [train_loss_K, *train_loss_full] = train_loss_full
        #train_loss = dot(train_loss_full, loss_weights_vec)
        train_loss = np.nan
        train_out = train_model.predict(train_x, batch_size=batch_size)
        assert len(train_out) == len(train_y)
        npts = self.config.n_classes
        # final output/map, first npts maps are keypoint confs
        tt1 = PoseTools.get_pred_locs(train_out[-1][..., :npts]) - \
              PoseTools.get_pred_locs(train_y[-1][..., :npts])
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        train_dist = np.nanmean(tt1)
        # average over batches/pts, *in output space/coords*

        # val dist
        val_x = val_x[0]
        assert val_x.shape[0] == batch_size
        val_out = pred_model.predict_on_batch(val_x)
        val_out = val_out[:, :, :2]  # cols should be x, y, max/conf
        assert val_y.shape == val_out.shape, "val_y {} val_out {}".format(val_y.shape, val_out.shape)
        _, val_dist, _, _, _ = dpkkpts.keypoint_errors(val_y, val_out)
        # valdist should be in input-space
        assert val_dist.shape == (batch_size, self.config.n_classes)
        val_dist = np.mean(val_dist)  # all batches, all pts

        lr = K.eval(train_model.optimizer.lr)

        self.train_info['val_dist'].append(val_dist)
        self.train_info['train_dist'].append(train_dist)
        self.train_info['train_loss'].append(train_loss)
        self.train_info['train_loss_K'].append(train_loss_K)
        self.train_info['train_loss_full'].append(train_loss_full)
        self.train_info['step'].append(int(step))
        self.train_info['lr'].append(lr)

        p_str = ''
        for k in self.train_info.keys():
            lastval = self.train_info[k][-1]
            if k == 'lr':
                p_str += '{:s}:{:.4g} '.format(k, lastval)
            elif isinstance(lastval, list):
                p_str += '{:s}:<list {} els>'.format(k, len(lastval))
            else:
                p_str += '{:s}:{:.2f} '.format(k, lastval)
        logging.info(p_str)

        conf = self.config
        train_data_file = os.path.join(self.config.cachedir, 'traindata')

        json_data = {}
        for x in self.train_info.keys():
            json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
        with open(train_data_file + '.json', 'w') as json_file:
            json.dump(json_data, json_file)
        with open(train_data_file, 'wb') as td:
            pickle.dump([self.train_info, conf], td, protocol=2)

        name = 'deepnet'

        if step % conf.save_step == 0:
            train_model.save(str(os.path.join(
                conf.cachedir, name + '-{}'.format(int(step)))))
