import logging
import threading
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from .config import load_config
from deepcut.nnet.net_factory import pose_net
from .nnet.pose_net import get_batch_spec
from .util.mylogging import setup_logging
import urllib
import os
import PoseTools
import numpy as np
import deepcut.predict as predict
from .pose_dataset import Batch, PoseDataset
import json
import pickle
from easydict import EasyDict as edict
import deepcut.config as config
import  tarfile

# name = 'deepnet'

class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg.multi_step
        self.current_step = 0
        self.n_steps = cfg.dl_steps
        self.gamma = cfg.gamma

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1
        #lr = 0.0001 * (self.gamma ** (iteration*3/ self.n_steps))
        return lr


def setup_preloading(batch_spec):
    placeholders = {name: tf.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20

    q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    try:
        while not coord.should_stop():
            batch_np = dataset.next_batch()
            food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
            sess.run(enqueue_op, feed_dict=food)
    except tf.errors.CancelledError:
        pass


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t


def get_optimizer(loss_op, cfg):
    learning_rate = tf.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(cfg.adam_lr)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op


def save_td(cfg, train_info,name):
    if name == 'deepnet':
        train_data_file = os.path.join(cfg.cachedir, 'traindata')
    else:
        train_data_file = os.path.join(cfg.cachedir, name + '_traindata')

    # train_data_file = os.path.join( cfg.cachedir, 'traindata')
    json_data = {}
    for x in train_info.keys():
        json_data[x] = np.array(train_info[x]).astype(np.float64).tolist()
    with open(train_data_file + '.json', 'w') as json_file:
        json.dump(json_data, json_file)
    with open(train_data_file, 'wb') as train_data_file:
        pickle.dump([train_info, cfg], train_data_file, protocol=2)


def set_deepcut_defaults(cfg):
    cfg.batch_size = 1
    cfg.display_step = 500
    cfg.dc_scale = 0.8
    cfg.dl_steps = 1030000
    cfg.save_step = 5000

def get_read_fn(cfg, data_path):
    cfg = edict(cfg.__dict__)
    cfg = config.convert_to_deepcut(cfg)
    cfg.batch_size = 1
    cfg.shuffle = False

    dataset = PoseDataset(cfg, data_path, distort=False)
    n = dataset.num_images
    def read_fn():
        batch_np = dataset.next_batch()
        loc_in = batch_np[Batch.locs]
        ims = batch_np[Batch.inputs]
        if cfg.img_dim == 1:
            ims = ims[:,:,:,0:1]
        info = [0, 0, 0]
        return ims, loc_in, info

    return read_fn, n


def train(cfg,name='deepnet'):
#    setup_logging()

    cfg = edict(cfg.__dict__)
    cfg = config.convert_to_deepcut(cfg)

    if name == 'deepnet':
        train_data_file = os.path.join(cfg.cachedir, 'traindata')
    else:
        train_data_file = os.path.join(cfg.cachedir, name + '_traindata')

    with open(train_data_file, 'wb') as td_file:
        pickle.dump(cfg, td_file, protocol=2)
    logging.info('Saved config to {}'.format(train_data_file))

    dirname = os.path.dirname(os.path.dirname(__file__))
    init_weights = os.path.join(dirname, 'pretrained','resnet_v1_50.ckpt')

    if not os.path.exists(init_weights):
        # Download and save the pretrained resnet weights.
        logging.info('Downloading pretrained resnet 50 weights ...')
        urllib.urlretrieve('http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz', os.path.join(dirname,'pretrained','resnet_v1_50_2016_08_28.tar.gz'))
        tar = tarfile.open(os.path.join(dirname,'pretrained','resnet_v1_50_2016_08_28.tar.gz'))
        tar.extractall(path=os.path.join(dirname,'pretrained'))
        tar.close()
        logging.info('Done downloading pretrained weights')

    db_file_name = os.path.join(cfg.cachedir, 'train_data.p')
    dataset = PoseDataset(cfg, db_file_name,distort=True)
    train_info = {'train_dist':[],'train_loss':[],'val_dist':[],'val_loss':[],'step':[]}

    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    net = pose_net(cfg)
    losses = net.train(batch)
    total_loss = losses['total_loss']
    outputs = [net.heads['part_pred'], net.heads['locref']]

    for k, t in losses.items():
        tf.summary.scalar(k, t)
    merged_summaries = tf.summary.merge_all()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=50,save_relative_paths=True)

    sess = tf.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    learning_rate, train_op = get_optimizer(total_loss, cfg)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, init_weights)

    #max_iter = int(cfg.multi_step[-1][1])
    max_iter = int(cfg.dl_steps)
    display_iters = cfg.display_step
    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    model_name = os.path.join( cfg.cachedir, name)
    ckpt_file = os.path.join(cfg.cachedir, name + '_ckpt')

    start = time.time()
    save_start = time.time()
    for it in range(max_iter+1):
        current_lr = lr_gen.get_lr(it)
        [_, loss_val] = sess.run([train_op, total_loss], # merged_summaries],
                                          feed_dict={learning_rate: current_lr})
        cum_loss += loss_val
 #       train_writer.add_summary(summary, it)

        if it % display_iters == 0:
            cur_out, batch_out = sess.run([outputs, batch], feed_dict={learning_rate: current_lr})
            scmap, locref = predict.extract_cnn_output(cur_out, cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            loc_pred = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            loc_in = batch_out[Batch.locs]
            dd = np.sqrt(np.sum(np.square(loc_pred[:,:,:2]-loc_in),axis=-1))
            dd = dd*cfg.dlc_rescale
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            end = time.time()
            # print('Time to train: {}'.format(end-start))
            start = end
            print("iteration: {} loss: {} dist: {}  lr: {}"
                         .format(it, "{0:.4f}".format(average_loss),
                                 '{0:.2f}'.format(dd.mean()), current_lr))
            train_info['step'].append(it)
            train_info['train_loss'].append(loss_val)
            train_info['val_loss'].append(loss_val)
            train_info['val_dist'].append(dd.mean())
            train_info['train_dist'].append(dd.mean())

            save_td(cfg, train_info,name)

        # Save snapshot
        if 'save_time' in cfg.keys() and cfg['save_time'] is not None:
            if (time.time() - save_start) > cfg['save_time']*60:
                saver.save(sess, model_name, global_step=it,
                           latest_filename=os.path.basename(ckpt_file))
                save_start = time.time()
        else:
            if (it % cfg.save_step == 0 ) or it == max_iter:
                saver.save(sess, model_name, global_step=it,
                       latest_filename=os.path.basename(ckpt_file))

    coord.request_stop()
    coord.join([thread],stop_grace_period_secs=60)
    sess.close()


def get_pred_fn(cfg, model_file=None,name='deepnet'):

    cfg = edict(cfg.__dict__)
    cfg = config.convert_to_deepcut(cfg)

    if model_file is None:
        ckpt_file = os.path.join(cfg.cachedir, name + '_ckpt')
        latest_ckpt = tf.train.get_checkpoint_state(cfg.cachedir, ckpt_file)
        model_file = latest_ckpt.model_checkpoint_path
        init_weights = model_file
    else:
        init_weights = model_file

    tf.reset_default_graph()
    sess, inputs, outputs = predict.setup_pose_prediction(cfg, init_weights)

    def pred_fn(all_f):
        if cfg.img_dim == 1:
            cur_im = np.tile(all_f,[1,1,1,3])
        else:
            cur_im = all_f
        cur_im, _ = PoseTools.preprocess_ims(cur_im, in_locs=np.zeros([cur_im.shape[0], cfg.n_classes, 2]), conf=cfg, distort=False, scale=cfg.dlc_rescale)

        cur_out = sess.run(outputs, feed_dict={inputs: cur_im})
        scmap, locref = predict.extract_cnn_output(cur_out, cfg)
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        pose = pose[:,:,:2]*cfg.dlc_rescale
        ret_dict = {}
        ret_dict['locs'] = pose
        ret_dict['hmaps'] = scmap
        ret_dict['conf'] = np.max(scmap, axis=(1, 2))
        return ret_dict

    def close_fn():
        sess.close()

    return pred_fn, close_fn, model_file

def model_files(cfg, name='deepnet'):
    ckpt_file = os.path.join(cfg.cachedir, name + '_ckpt')
    if not os.path.exists(ckpt_file):
        return []
    latest_ckpt = tf.train.get_checkpoint_state(cfg.cachedir, ckpt_file)
    latest_model_file = latest_ckpt.model_checkpoint_path
    import glob
    all_model_files = glob.glob(latest_model_file + '.*')
    train_data_file = os.path.join( cfg.cachedir, 'traindata')
    all_model_files.extend([ckpt_file, train_data_file])

    return all_model_files


if __name__ == '__main__':
    train()
