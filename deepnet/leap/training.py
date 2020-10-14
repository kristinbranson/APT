''' Modified by Mayank Kabra
From LEAP https://github.com/talmo/leap by Talmo Pereira
'''
import numpy as np
import h5py
import os
from time import time
from scipy.io import loadmat, savemat
import re
import shutil
# import clize
import json
import PoseTools
import math
import pickle
import logging
import contextlib

import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback,LearningRateScheduler
from keras.callbacks import Callback
import keras.backend as K

from leap import models
from leap.image_augmentation import PairedImageAugmenter, MultiInputOutputPairedImageAugmenter
#from leap.viz import show_pred, show_confmap_grid, plot_history
from leap.utils import load_dataset


def train_val_split(X, Y, val_size=0.15, shuffle=True):
    """ Splits datasets into training and validation sets. """

    if val_size < 1:
        val_size = int(np.round(len(X) * val_size))

    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)

    val_idx = idx[:val_size]
    idx = idx[val_size:]

    return X[idx], Y[idx], X[val_idx], Y[val_idx], idx, val_idx


def create_run_folders(run_name, base_path="models", clean=False):
    """ Creates subfolders necessary for outputs of training. """

    def is_empty_run(run_path):
        weights_path = os.path.join(run_path, "weights")
        has_weights_folder = os.path.exists(weights_path)
        return not has_weights_folder or len(os.listdir(weights_path)) == 0

    run_path = os.path.join(base_path, run_name)

    if not clean:
        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path): #and not is_empty_run(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1

    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    os.makedirs(run_path)
    os.makedirs(os.path.join(run_path, "weights"))
    os.makedirs(os.path.join(run_path, "viz_pred"))
    os.makedirs(os.path.join(run_path, "viz_confmaps"))
    print("Created folder:", run_path)

    return run_path



class LossHistory(keras.callbacks.Callback):
    def __init__(self, run_path):
        super(LossHistory,self).__init__()
        self.run_path = run_path

    def on_train_begin(self, logs={}):
        self.history = []

    def on_epoch_end(self, batch, logs={}):
        # Append to log list
        self.history.append(logs.copy())

        # Save history so far to MAT file
        savemat(os.path.join(self.run_path, "history.mat"),
                {k: [x[k] for x in self.history] for k in self.history[0].keys()})

        # Plot graph
        # plot_history(self.history, save_path=os.path.join(self.run_path, "history.png"))


def create_model(net_name, img_size, output_channels, **kwargs):
    """ Wrapper for initializing a network for training. """
    # compile_model = getattr(models, net_name)

    compile_model = dict(
        leap_cnn=models.leap_cnn,
        hourglass=models.hourglass,
        stacked_hourglass=models.stacked_hourglass,
        ).get(net_name)
    if compile_model == None:
        return None

    return compile_model(img_size, output_channels, **kwargs)

def train(data_path,
    base_output_path="models",
    run_name='deepnet',
    data_name=None,
    net_name="leap_cnn",
    clean=False,
    box_dset="box",
    confmap_dset="confmaps",
    val_size=0.15,
    preshuffle=True,
    filters=64,
    rotate_angle=15,
    epochs=50,
    batch_size=32,
    batches_per_epoch=50,
    val_batches_per_epoch=10,
    viz_idx=0,
    reduce_lr_factor=0.1,
    reduce_lr_patience=3,
    reduce_lr_min_delta=1e-5,
    reduce_lr_cooldown=0,
    reduce_lr_min_lr=1e-10,
    save_every_epoch=False,
    amsgrad=False,
    upsampling_layers=False,
    conf=None
    ):
    """
    Trains the network and saves the intermediate results to an output directory.

    :param data_path: Path to an HDF5 file with box and confmaps datasets
    :param base_output_path: Path to folder in which the run data folder will be saved
    :param run_name: Name of the training run. If not specified, will be formatted according to other parameters.
    :param data_name: Name of the dataset for use in formatting run_name
    :param net_name: Name of the network for use in formatting run_name
    :param clean: If True, deletes the contents of the run output path
    :param box_dset: Name of the box dataset in the HDF5 data file
    :param confmap_dset: Name of the confidence maps dataset in the HDF5 data file
    :param preshuffle: If True, shuffle prior to splitting the dataset, otherwise validation set will be the last frames
    :param val_size: Fraction of dataset to use as validation
    :param filters: Number of filters to use as baseline (see create_model)
    :param rotate_angle: Images will be augmented by rotating by +-rotate_angle
    :param epochs: Number of epochs to train for
    :param batch_size: Number of samples per batch
    :param batches_per_epoch: Number of batches per epoch (validation is evaluated at the end of the epoch)
    :param val_batches_per_epoch: Number of batches for validation
    :param viz_idx: Index of the sample image to use for visualization
    :param reduce_lr_factor: Factor to reduce the learning rate by (see ReduceLROnPlateau)
    :param reduce_lr_patience: How many epochs to wait before reduction (see ReduceLROnPlateau)
    :param reduce_lr_min_delta: Minimum change in error required before reducing LR (see ReduceLROnPlateau)
    :param reduce_lr_cooldown: How many epochs to wait after reduction before LR can be reduced again (see ReduceLROnPlateau)
    :param reduce_lr_min_lr: Minimum that the LR can be reduced down to (see ReduceLROnPlateau)
    :param save_every_epoch: Save weights at every epoch. If False, saves only initial, final and best weights.
    :param amsgrad: Use AMSGrad variant of optimizer. Can help with training accuracy on rare examples (see Reddi et al., 2018)
    :param upsampling_layers: Use simple bilinear upsampling layers as opposed to learned transposed convolutions
    """

    # Load
    use_leap_lr = conf.get('leap_use_default_lr', False)

    print("data_path:", data_path)
    box, confmap = load_dataset(data_path, X_dset=box_dset, Y_dset=confmap_dset)

    if use_leap_lr:
        box, confmap, val_box, val_confmap, train_idx, val_idx = train_val_split(box, confmap, val_size=val_size, shuffle=preshuffle)
    else:
        val_box   = box
        val_confmap = confmap
        train_idx = np.array([0])
        val_idx = np.array([0])

    print("box.shape:", box.shape)
    print("val_box.shape:", val_box.shape)

    # Pull out metadata
    img_size = np.array(box.shape[1:])
    img_size[0] = img_size[0]//conf.rescale
    img_size[1] = img_size[1]//conf.rescale

    num_output_channels = conf.n_classes

    # Build run name if needed
    if data_name == None:
        data_name = os.path.splitext(os.path.basename(data_path))[0]
    if run_name == None:
        # Ex: "WangMice-DiegoCNN_v1.0_filters=64_rot=15_lrfactor=0.1_lrmindelta=1e-05"
        # run_name = "%s-%s_filters=%d_rot=%d_lrfactor=%.1f_lrmindelta=%g" % (data_name, net_name, filters, rotate_angle, reduce_lr_factor, reduce_lr_min_delta)
        run_name = "%s-%s_epochs=%d" % (data_name, net_name, epochs)
    print("data_name:", data_name)
    print("run_name:", run_name)

    # Create network
    if isinstance(net_name, keras.models.Model):
        model = net_name
        net_name = model.name
    else:
        model = create_model(net_name, img_size, num_output_channels, filters=filters, amsgrad=amsgrad, upsampling_layers=upsampling_layers, summary=True)
    if model == None:
        print("Could not find model:", net_name)
        return

    # Initialize run
    run_path = base_output_path
    savemat(os.path.join(run_path, "training_info.mat"),
            {"data_path": data_path, "val_idx": val_idx, "train_idx": train_idx,
             "base_output_path": base_output_path, "run_name": run_name, "data_name": data_name,
             "net_name": net_name, "clean": clean, "box_dset": box_dset, "confmap_dset": confmap_dset,
             "preshuffle": preshuffle, "val_size": val_size, "filters": filters, "rotate_angle": rotate_angle,
             "epochs": epochs, "batch_size": batch_size, "batches_per_epoch": batches_per_epoch,
             "val_batches_per_epoch": val_batches_per_epoch, "viz_idx": viz_idx, "reduce_lr_factor": reduce_lr_factor,
             "reduce_lr_patience": reduce_lr_patience, "reduce_lr_min_delta": reduce_lr_min_delta,
             "reduce_lr_cooldown": reduce_lr_cooldown, "reduce_lr_min_lr": reduce_lr_min_lr,
             "save_every_epoch": save_every_epoch, "amsgrad": amsgrad, "upsampling_layers": upsampling_layers})

    # Save initial network
    model.save(str(os.path.join(run_path, "initial_model.h5")))

    # Data generators/augmentation
    input_layers = model.input_names
    output_layers = model.output_names
    if len(input_layers) > 1 or len(output_layers) > 1:
        train_datagen = MultiInputOutputPairedImageAugmenter(input_layers, output_layers, box, confmap, conf, shuffle=True)
        val_datagen = MultiInputOutputPairedImageAugmenter(input_layers, output_layers, val_box, val_confmap, conf, shuffle=True)
    else:
        train_datagen = PairedImageAugmenter(box, confmap, conf, shuffle=True)
        val_datagen = PairedImageAugmenter(val_box, val_confmap, conf,shuffle=True)

    # For debugging
    gg = iter(train_datagen)
    xx = next(gg)

    # history_callback = LossHistory(run_path=run_path)

    initial_lr = conf.get('leap_base_lr',0.0001)
    lr_drop_step_frac = conf.get('lr_drop_step', 0.15)

    def step_decay(cur_epoch):
        lrate = initial_lr if cur_epoch < ((1-lr_drop_step_frac) * epochs) else initial_lr/10
        return  lrate

    if not use_leap_lr:
        reduce_lr_callback = LearningRateScheduler(step_decay)
    else:
        reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=reduce_lr_factor, patience=reduce_lr_patience, verbose=1, mode="auto", epsilon=reduce_lr_min_delta, cooldown=reduce_lr_cooldown, min_lr=reduce_lr_min_lr)


    # checkpointer = ModelCheckpoint(model_file, verbose=0, save_best_only=False,period=save_step)
    save_time = conf.get('save_time', None)
    class OutputObserver(Callback):
        def __init__(self, conf, dis):
            self.train_di, self.val_di = dis
            self.train_info = {}
            self.train_info['step'] = []
            self.train_info['train_dist'] = []
            self.train_info['train_loss'] = []
            self.train_info['val_dist'] = []
            self.train_info['val_loss'] = []
            self.config = conf
            self.force = False
            self.train_ndx = 0
            self.val_ndx  = 0
            self.start_time = time()

        def on_epoch_end(self, epoch, logs={}):
            step = epoch*conf.display_step
            val_x, val_y = self.val_di[self.val_ndx]
            self.val_ndx += 1
            if self.val_ndx >= len(self.val_di):
                self.val_ndx = 0
            val_out = self.model.predict(val_x)
            val_loss = self.model.evaluate(val_x, val_y, verbose=0)
            train_x, train_y = self.train_di[self.train_ndx]
            self.train_ndx += 1
            if self.train_ndx >= len(self.train_di):
                self.train_ndx = 0
            train_out = self.model.predict(train_x)
            train_loss = self.model.evaluate(train_x, train_y, verbose=0)

            # dist only for last layer
            tt1 = PoseTools.get_pred_locs(val_out) - \
                  PoseTools.get_pred_locs(val_y)
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            val_dist = np.nanmean(tt1)
            tt1 = PoseTools.get_pred_locs(train_out) - \
                  PoseTools.get_pred_locs(train_y)
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            train_dist = np.nanmean(tt1)
            self.train_info['val_dist'].append(val_dist)
            self.train_info['val_loss'].append(val_loss*10000)
            self.train_info['train_dist'].append(train_dist)
            self.train_info['train_loss'].append(train_loss*10000)
            self.train_info['step'].append(int(step))

            p_str = ''
            for k in self.train_info.keys():
                p_str += '{:s}:{:.2f} '.format(k, self.train_info[k][-1])
            print(p_str)

            if run_name == 'deepnet':
                train_data_file = os.path.join( self.config.cachedir, 'traindata')
            else:
                train_data_file = os.path.join( self.config.cachedir, self.config.expname + '_' + run_name + '_traindata')

            json_data = {}
            for x in self.train_info.keys():
                json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
            with open(train_data_file + '.json', 'w') as json_file:
                json.dump(json_data, json_file)
            with open(train_data_file, 'wb') as train_data_file:
                pickle.dump([self.train_info, conf], train_data_file, protocol=2)

            if save_time is None:
                if step % conf.save_step == 0:
                    model.save(str(os.path.join(run_path,run_name + '-{}'.format(step))))
            else:
                if time() - self.start_time > conf.save_time*60:
                    self.start_time = time()
                    model.save(str(os.path.join(run_path, run_name + '-{}'.format(step))))

    obs = OutputObserver(conf,[train_datagen,val_datagen])

    # Train!
    epoch0 = 0
    t0_train = time()
    use_multiprocessing = False if box.shape[0] < 300 else True
    model.save(str(os.path.join(run_path, run_name + '-{}'.format(0))))
    training = model.fit_generator(
            train_datagen,
            initial_epoch=epoch0,
            epochs=epochs,
            verbose=0,
    #         use_multiprocessing=True,
    #         workers=8,
            steps_per_epoch=batches_per_epoch,
            max_queue_size=512,
            shuffle=False,
            validation_data=val_datagen,
            validation_steps=val_batches_per_epoch,
            callbacks = [
                reduce_lr_callback,
                # checkpointer,
                obs
            ]
        )

    # Compute total elapsed time for training
    elapsed_train = time() - t0_train
    print("Total runtime: %.1f mins" % (elapsed_train / 60))

    # Save final model
    model.save(str(os.path.join(run_path, run_name + '-{}'.format(int(conf.dl_steps)))))
    obs.on_epoch_end(epochs)
    K.clear_session()


def get_read_fn(conf, data_path):

    batch_size = 1
    rotate_angle = 0
    net_name = conf.leap_net_name
    box_dset="box"
    confmap_dset="joints"
    filters=64

    box, confmap = load_dataset(data_path, X_dset=box_dset, Y_dset=confmap_dset)

    # Pull out metadata
    img_size = box.shape[1:]
    num_output_channels = confmap.shape[-1]

    # Create network
    model = create_model(net_name, img_size, num_output_channels, filters=filters, amsgrad=False, upsampling_layers=False, summary=False)
    if model is None:
        print("Could not find model:", net_name)
        return

    input_layers = model.input_names
    output_layers = model.output_names
    if len(input_layers) > 1 or len(output_layers) > 1:
        datagen = MultiInputOutputPairedImageAugmenter(input_layers, output_layers, box, confmap, conf, shuffle=False)
    else:
        datagen = PairedImageAugmenter(box, confmap, conf, shuffle=False)

    cur_ex = [0]
    def read_fn():
        im, hmap = datagen[cur_ex[0]]
        cur_ex[0] += 1
        locs = PoseTools.get_pred_locs(hmap)
        info = [0,0,0]
        return im, locs, info

    n_db = box.shape[0]

    return read_fn, n_db



def get_pred_fn(conf, model_file=None,name='deepnet',tmr_pred=None):

    if tmr_pred is None:
        tmr_pred = contextlib.suppress()

    if model_file is None:
        latest_model_file = PoseTools.get_latest_model_file_keras(conf,name)
    else:
        latest_model_file = model_file
    model = keras.models.load_model(str(latest_model_file))

    def pred_fn(all_f):
        newY = int(np.ceil(float(all_f.shape[1]) / 32) * 32)
        newX = int(np.ceil(float(all_f.shape[2]) / 32) * 32)
        X1 = np.zeros([all_f.shape[0], newY, newX, all_f.shape[3]]).astype('float32')
        X1[:, :all_f.shape[1], :all_f.shape[2], :] = all_f

        X1, _ = PoseTools.preprocess_ims(X1, in_locs=np.zeros([X1.shape[0], conf.n_classes, 2]), conf=conf, distort=False, scale=conf.rescale)


        X1 = X1.astype("float32") / 255.
        with tmr_pred:
            pred = model.predict(X1,batch_size = X1.shape[0])
        pred = np.stack(pred)
        pred = pred[:,:all_f.shape[1],:all_f.shape[2],:]
        base_locs = PoseTools.get_pred_locs(pred)
        base_locs = base_locs * conf.rescale

        ret_dict = {}
        ret_dict['locs'] = base_locs
        ret_dict['hmaps'] = pred
        ret_dict['conf'] = np.max(pred, axis=(1, 2))
        return ret_dict

    close_fn = K.clear_session

    return pred_fn, close_fn, latest_model_file


def model_files(conf, name):
    latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    if latest_model_file is None:
        return None
    traindata_file = PoseTools.get_train_data_file(conf,name)
    return [latest_model_file, traindata_file + '.json']


if __name__ == "__main__":
    # Turn interactive plotting off
    # plt.ioff()

    # Wrapper for running from commandline
    #clize.run(train)
    pass
