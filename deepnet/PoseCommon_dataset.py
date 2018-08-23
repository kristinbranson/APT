from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object
import tensorflow as tf
import os
import PoseTools
import multiResData
from enum import Enum
import numpy as np
import re
import pickle
import sys
import math
from past.utils import old_div
from tensorflow.contrib.layers import batch_norm
# from batch_norm import batch_norm_mine_old as batch_norm
from matplotlib import pyplot as plt
import copy
import cv2
import gc
import resource
import json
import math
import threading
import logging
from six import reraise as raise_


renorm = False
def conv_relu(x_in, kernel_shape, train_phase):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, is_training=train_phase, renorm=renorm)
    return tf.nn.relu(conv + biases)


def conv_relu3(x_in, n_filt, train_phase, keep_prob=None,use_leaky=False):
    in_dim = x_in.get_shape().as_list()[3]
    kernel_shape = [3, 3, in_dim, n_filt]
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, decay=0.99, is_training=train_phase, renorm=renorm)

    if keep_prob is not None:
        conv = tf.nn.dropout(conv, keep_prob)

    if use_leaky:
        return tf.nn.leaky_relu(conv+biases)
    else:
        return tf.nn.relu(conv + biases)


def conv_shortcut(x_in, n_filt, train_phase, keep_prob=None,use_leaky=False):
    # shortcut connections for residual
    in_dim = x_in.get_shape().as_list()[3]
    kernel_shape = [1, 1, in_dim, n_filt]
    with tf.variable_scope('shortcut'):
        weights = tf.get_variable("weights", kernel_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", kernel_shape[-1],
                                 initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, decay=0.99, is_training=train_phase, renorm=renorm)

    return conv



def conv_relu3_noscaling(x_in, n_filt, train_phase):
    in_dim = x_in.get_shape().as_list()[3]
    kernel_shape = [3, 3, in_dim, n_filt]
    weights = tf.get_variable(
        "weights", kernel_shape,
        initializer=
        tf.contrib.layers.variance_scaling_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, decay=0.99, is_training=train_phase, renorm=renorm)
    return tf.nn.relu(conv + biases)


def print_train_data(cur_dict):
    p_str = ''
    for k in cur_dict.keys():
        p_str += '{:s}:{:.2f} '.format(k, cur_dict[k])
    print(p_str)


def initialize_remaining_vars(sess):
    var_list = set(sess.run(tf.report_uninitialized_variables()))
    vlist = [v for v in tf.global_variables() if \
             v.name.split(':')[0] in var_list]
    sess.run(tf.variables_initializer(vlist))


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def l2_dist(x,y):
    return np.sqrt(np.sum((x-y)**2,axis=-1))



class PoseCommon(object):

    class DBType(Enum):
        Train = 1
        Val = 2


    def __init__(self, conf, name):
        self.coord = None
        self.threads = None
        self.conf = conf
        self.name = name
        self.net_name = 'pose_common'
        self.train_type = None
        self.train_data = None
        self.val_data = None
        self.train_queue = None
        self.val_queue = None
        self.xs = None
        self.locs = None
        self.info = None
        self.ph = {}
        self.fd = {}
        self.train_info = {}
        self.cost = None
        self.pred = None
        self.opt = None
        self.saver = None
        self.dep_nets = []
        self.joint = False
        self.edge_ignore = 0 # amount of edge to ignore while computing prediction locations
        self.train_loss = np.zeros(500) # keep track of past loss for importance sampling
        self.step = 0 # might be required for mdn.
        self.extra_data = None
        self.db_name = ''
        self.read_and_decode = multiResData.read_and_decode
        self.scale = 1
        self.extra_info_sz = 1
        self.ckpt_file = os.path.join( conf.cachedir, conf.expname + '_' + name + '_ckpt')

        self.q_placeholder_spec = []
        self.q_placeholders = []
        self.q_fns = []
        self.for_training = 0
        self.train_data_name = None
        self.td_fields = ['dist','loss']
        self.input_dtypes = [tf.float32, tf.float32, tf.float32, tf.float32]


    def get_latest_model_file(self):
        latest_ckpt = tf.train.get_checkpoint_state(
            self.conf.cachedir, self.ckpt_file)
        return latest_ckpt.model_checkpoint_path


    def get_var_list(self):
        var_list = tf.global_variables(self.net_name)
        for dep_net in self.dep_nets:
            var_list += dep_net.get_var_list()
        return var_list

    def create_saver(self):
        saver = {}
        name = self.name
        net_name = self.net_name
        saver['out_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name)
        if self.train_data_name is None:
            saver['train_data_file'] = os.path.join(
                self.conf.cachedir,
                self.conf.expname + '_' + name + '_traindata')
        else:
            saver['train_data_file'] = os.path.join(
                self.conf.cachedir,
                self.train_data_name)

        saver['ckpt_file'] = self.ckpt_file
        var_list = self.get_var_list()
        saver['saver'] = (tf.train.Saver(var_list=var_list,
                                         max_to_keep=self.conf.maxckpt,
                                         save_relative_paths=True))
        self.saver = saver

        for dep_net in self.dep_nets:
            dep_net.create_saver()


    def restore(self, sess, model_file=None):
        saver = self.saver
        if model_file is not None:
            latest_model_file = model_file
            saver['saver'].restore(sess, model_file)
        else:
            grr = os.path.split(self.ckpt_file) # angry that get_checkpoint_state doesnt accept complete path to ckpt file. Damn idiots!
            latest_ckpt = tf.train.get_checkpoint_state(grr[0],grr[1])
            latest_model_file = latest_ckpt.model_checkpoint_path
            saver['saver'].restore(sess, latest_model_file)
        return latest_model_file       


    def save(self, sess, step):
        saver = self.saver
        out_file = saver['out_file'].replace('\\', '/')
        saver['saver'].save(sess, out_file, global_step=step,
                            latest_filename=os.path.basename(saver['ckpt_file']))
        print('Saved state to %s-%d' % (out_file, step))


    def init_td(self):
        ex_td_fields = ['step']
        for t_f in self.td_fields:
            ex_td_fields.append('train_' + t_f)
            ex_td_fields.append('val_' + t_f)
        train_info = {}
        for t_f in ex_td_fields:
            train_info[t_f] = []
        self.train_info = train_info


    def restore_td(self):
        saver = self.saver
        train_data_file = saver['train_data_file'].replace('\\', '/')
        with open(train_data_file, 'rb') as td_file:
            if sys.version_info.major == 3:
                in_data = pickle.load(td_file, encoding='latin1')
            else:
                in_data = pickle.load(td_file)

            if not isinstance(in_data, dict):
                train_info, load_conf = in_data
                print('Parameters that do not match for {:s}:'.format(train_data_file))
                PoseTools.compare_conf(self.conf, load_conf)
            else:
                print("No config was stored for base. Not comparing conf")
                train_info = in_data
        self.train_info = train_info


    def save_td(self):
        saver = self.saver
        train_data_file = saver['train_data_file']
        with open(train_data_file, 'wb') as td_file:
            pickle.dump([self.train_info, self.conf], td_file, protocol=2)
        json_data = {}
        for x in self.train_info.keys():
            json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
        with open(train_data_file+'.json','w') as json_file:
            json.dump(json_data, json_file)


    def update_td(self, cur_dict):
        for k in cur_dict.keys():
            self.train_info[k].append(cur_dict[k])
        print_train_data(cur_dict)


    def create_ph_fd(self):
        self.ph = {}
        learning_rate_ph = tf.placeholder(
            tf.float32, shape=[], name='learning_r')
        self.ph['learning_rate'] = learning_rate_ph
        self.ph['phase_train'] = tf.placeholder(
            tf.bool, name='phase_train')
        self.ph['is_train'] = tf.placeholder(tf.bool,name='is_train')
        self.fd = {self.ph['learning_rate']:1, self.ph['phase_train']:False, self.ph['is_train']: False}


    def fd_train(self):
        self.fd[self.ph['phase_train']] = True
        self.fd[self.ph['is_train']] = True

    def fd_val(self):
        self.fd[self.ph['phase_train']] = False
        self.fd[self.ph['is_train']] = False


    def create_datasets(self):

        conf = self.conf
        def _parse_function(serialized_example):
            features = tf.parse_single_example(
                serialized_example,
                features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                          'width': tf.FixedLenFeature([], dtype=tf.int64),
                          'depth': tf.FixedLenFeature([], dtype=tf.int64),
                          'trx_ndx': tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
                          'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                          'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                          'ts': tf.FixedLenFeature([], dtype=tf.float32),
                          'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                          })
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            trx_ndx = tf.cast(features['trx_ndx'], tf.int64)
            image = tf.reshape(image, conf.imsz + (conf.imgDim,))

            locs = tf.cast(features['locs'], tf.float32)
            exp_ndx = tf.cast(features['expndx'], tf.float32)
            ts = tf.cast(features['ts'], tf.float32)  # tf.constant([0]); #
            info = tf.stack([exp_ndx, ts, tf.cast(trx_ndx, tf.float32)])
            return image, locs, info

        train_db = os.path.join(self.conf.cachedir, self.conf.trainfilename) + '.tfrecords'
        train_dataset = tf.data.TFRecordDataset(train_db)
        val_db = os.path.join(self.conf.cachedir, self.conf.valfilename) + '.tfrecords'
        if os.path.exists(val_db):
            print("Val DB exists: Data for validation from:{}".format(val_db))
        else:
            print("Val DB does not exists: Data for validation from:{}".format(train_db))
            val_db = train_db
        val_dataset = tf.data.TFRecordDataset(val_db)

        train_dataset = train_dataset.map(map_func=_parse_function,num_parallel_calls=5)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(buffer_size=100)
        train_dataset = train_dataset.batch(self.conf.batch_size)
        train_dataset = train_dataset.map(map_func=self.train_py_map,num_parallel_calls=5)
        train_dataset = train_dataset.prefetch(buffer_size=100)

        val_dataset = val_dataset.map(map_func=_parse_function,num_parallel_calls=2)
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(self.conf.batch_size)
        val_dataset = val_dataset.map(map_func=self.val_py_map,num_parallel_calls=2)
        val_dataset = val_dataset.prefetch(buffer_size=100)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        train_iter = train_dataset.make_one_shot_iterator()
        train_next = train_iter.get_next()

        val_iter = val_dataset.make_one_shot_iterator()
        val_next = val_iter.get_next()

        self.inputs = []
        for ndx in range(len(train_next)):
            self.inputs.append(tf.cond(self.ph['is_train'], lambda: tf.identity(train_next[ndx]), lambda: tf.identity(val_next[ndx])))


    def create_input_ph(self):
        # when we want to manually feed in data
        self.inputs = []
        for cur_d in self.input_dtypes:
            self.inputs.append(tf.placeholder(cur_d,None))


    def setup_train(self):
        self.create_ph_fd()
        self.create_datasets()

    def setup_pred(self):
        self.create_ph_fd()
        self.create_input_ph()

    def initialize_net(self, sess):
        name = self.net_name
        sess.run(tf.variables_initializer(PoseTools.get_vars(name)),
                 feed_dict=self.fd)
        print("Not loading {:s} variables. Initializing them".format(name))
        self.init_td()
        for dep_net in self.dep_nets:
            dep_net.initialize_net(sess)
        initialize_remaining_vars(sess)


    def train_step(self, step, sess, learning_rate, training_iters):
        cur_step = float(step)

        # n_steps = self.conf.n_steps
        # cur_lr = learning_rate * (self.conf.gamma ** (cur_step*n_steps/ training_iters))
        # self.fd[self.ph['learning_rate']] = cur_lr

        # cosine decay restarts
        t_mul = 2.0
        m_mul = 1.0
        alpha = 0.0
        n_steps = (t_mul)**self.conf.cos_steps - 1
        first_decay_steps = int(math.ceil(float(training_iters)/n_steps))+1
        completed_fraction = step / first_decay_steps

        def compute_step(completed_fraction, geometric=False):
            if geometric:
                i_restart = math.floor(
                    math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                    math.log(t_mul))

                sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
                completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart
            else:
                i_restart = math.floor(completed_fraction)
                completed_fraction = completed_fraction - i_restart
            return i_restart, completed_fraction

        if t_mul == 1.0:
            i_restart, completed_fraction = compute_step(completed_fraction, geometric=False)
        else:
            i_restart, completed_fraction = compute_step(completed_fraction, geometric=True)

        m_fac = m_mul ** i_restart
        cosine_decayed = 0.5 * m_fac * (
                1.0 + math.cos(math.pi * completed_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha
        self.fd[self.ph['learning_rate']] = learning_rate* decayed

        self.fd_train()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        sess.run(self.opt, self.fd,options=run_options)


    def create_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
#            plain vanilla.
#            self.opt = tf.train.AdamOptimizer(
#                learning_rate=self.ph['learning_rate']).minimize(self.cost)

            # clipped gradients.
            #optimizer = tf.train.RMSPropOptimizer(
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.ph['learning_rate'])
            gradients, variables = zip(*optimizer.compute_gradients(self.cost))
            gradients = [None if gradient is None else
                         tf.clip_by_norm(gradient, 5.0)
                         for gradient in gradients]
            self.opt = optimizer.apply_gradients(zip(gradients, variables))

    def compute_dist(self, preds, locs):
        tt1 = PoseTools.get_pred_locs(preds,self.edge_ignore) - locs
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        return np.nanmean(tt1)


    def compute_train_data(self, sess, db_type):
        self.fd_train() if db_type is self.DBType.Train \
            else self.fd_val()
        cur_loss, cur_pred, self.cur_inputs = \
            sess.run( [self.cost, self.pred, self.inputs], self.fd)
        cur_dist = self.compute_dist(cur_pred, self.cur_inputs[1])
        return cur_loss, cur_dist


    def train(self, create_network,
              loss, learning_rate):

        self.setup_train()
        self.pred = create_network()
        self.cost = loss(self.inputs, self.pred)
        self.create_optimizer()
        self.create_saver()
        training_iters = self.conf.dl_steps
        num_val_rep = self.conf.numTest / self.conf.batch_size + 1

        with tf.Session() as sess:
            self.initialize_net(sess)
            #self.restore_pretrained(sess,'/home/mayank/work/poseTF/cache/stephen_dataset/head_pose_umdn_joint-20000')
            #initialize_remaining_vars(sess)
            #self.init_td()

            for step in range(0, training_iters + 1):
                self.train_step(step, sess, learning_rate, training_iters)
                if step % self.conf.display_step == 0:
                    train_loss, train_dist = self.compute_train_data(sess, self.DBType.Train)
                    val_loss = 0.
                    val_dist = 0.
                    for _ in range(num_val_rep):
                       cur_loss, cur_dist = self.compute_train_data(sess, self.DBType.Val)
                       val_loss += cur_loss
                       val_dist += cur_dist
                    val_loss = val_loss / num_val_rep
                    val_dist = val_dist / num_val_rep
                    cur_dict = {'step': step,
                               'train_loss': train_loss, 'val_loss': val_loss,
                               'train_dist': train_dist, 'val_dist': val_dist}
                    self.update_td(cur_dict)
                if step % self.conf.save_step == 0:
                    self.save(sess, step)
                if step % self.conf.save_td_step == 0:
                    self.save_td()
            print("Optimization Finished!")
            self.save(sess, training_iters)
            self.save_td()



    def restore_net_common(self, create_network_fn, model_file=None):
        print('--- Loading the model by reconstructing the graph ---')
        self.setup_pred()
        self.pred = create_network_fn()
        self.create_saver()
        sess = tf.Session()
        latest_model_file = self.restore(sess, model_file)
        initialize_remaining_vars(sess)

        try:
            self.restore_td()
        except AttributeError:  # If the conf file has been modified
            print("Couldn't load train data because the conf has changed!")
            self.init_td()

        for i in self.inputs:
            self.fd[i] = np.zeros(i.get_shape().as_list())

        return sess, latest_model_file


    def restore_meta_common(self, model_file):
        print('--- Loading the model using the saved graph ---')
        self.create_ph_fd()
        sess = tf.Session()
        try:
            latest_model_file = self.restore_meta(self.name, sess, model_file)
        except tf.errors.NotFoundError:
            pass

        return sess, latest_model_file


    def restore_meta(self, name, sess, model_file=None):
        if model_file is None:
            ckpt_file = self.ckpt_file
            latest_ckpt = tf.train.get_checkpoint_state(ckpt_file)
            saver = tf.train.import_meta_graph(latest_ckpt.model_checkpoint_path+'.meta')
            latest_model_file =latest_ckpt.model_checkpoint_path
        else:
            saver = tf.train.import_meta_graph(model_file + '.meta')
            latest_model_file = model_file

        saver.restore(sess, latest_model_file)
        return latest_model_file


    def classify_val(self,train_type=0, at_step=-1):

        if train_type is 0:
            val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        else:
            val_file = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename + '.tfrecords')
        num_val = 0
        for record in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        self.init_train(train_type)
        self.pred = self.create_network()
        saver = self.create_saver()

        with tf.Session() as sess:
            start_at = self.init_and_restore(sess, True, ['loss', 'dist'],at_step)

            val_dist = []
            val_ims = []
            val_preds = []
            val_predlocs = []
            val_locs = []
            for step in range(num_val/self.conf.batch_size):
                self.setup_val(sess)
                cur_pred = sess.run(self.pred, self.fd)
                cur_predlocs = PoseTools.get_pred_locs(cur_pred)
                cur_dist = np.sqrt(np.sum(
                    (cur_predlocs-self.locs) ** 2, 2))
                val_dist.append(cur_dist)
                val_ims.append(self.xs)
                val_locs.append(self.locs)
                val_preds.append(cur_pred)
                val_predlocs.append(cur_predlocs)
            self.close_cursors()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape( (-1,) + in_a.shape[2:])
        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_predlocs = val_reshape(val_predlocs)
        val_locs = val_reshape(val_locs)

        return val_dist, val_ims, val_preds, val_predlocs, val_locs

    def plot_results(self,n=50):
        # saver = {}
        # saver['train_data_file'] = os.path.join(
        #     self.conf.cachedir,
        #     self.conf.expname + '_' + self.name + '_traindata')
        # self.saver = saver
        self.create_saver()
        self.restore_td()
        plt.figure()
        plt.plot(moving_average(self.train_info['val_dist'],n),c='r')
        plt.plot(moving_average(self.train_info['train_dist'],n),c='g')

    def iter_res_image(self, max_iter, shape, train_type=0,
                       perc=[90,95,97,99],min_iter=0):
        ptiles = []
        for iter in range(min_iter,max_iter+1,self.conf.save_step):
            tf.reset_default_graph()
            A = self.classify_val(train_type,iter)
            ptiles.append(np.percentile(A[0],perc,axis=0))

        niter = len(ptiles)
        if A[1].shape[3] == 1:
            im = A[1][0,:,:,0]
        else:
            im = A[1][0,...]
        iszx = A[1].shape[2]
        iszy = A[1].shape[1]
        im = np.tile(im,shape)
        locs = A[4][0,...]
        alocs = []
        for yy in range(shape[1]):
            for xx in range(shape[0]):
                alocs.append(locs + [xx*iszx, yy*iszy])

        alocs = np.array(alocs).reshape([-1,2])
        nperc = ptiles[0].shape[0]
        ptiles_a = np.array(ptiles).transpose([1,0,2]).reshape([nperc,-1])
        PoseTools.create_result_image(im,alocs,ptiles_a)
        return ptiles


class PoseCommonMulti(PoseCommon):

    def create_cursors(self, sess):
        train_ims, train_locs, train_info = \
            multiResData.read_and_decode_multi(self.train_queue, self.conf)
        val_ims, val_locs, val_info = \
            multiResData.read_and_decode_multi(self.val_queue, self.conf)
        self.train_data = [train_ims, train_locs, train_info]
        self.val_data = [val_ims, val_locs, val_info]
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def compute_dist_m(self,preds,locs):
        pred_locs = PoseTools.get_pred_locs_multi(
            preds, self.conf.max_n_animals,
            self.conf.label_blur_rad * 2)

        dist = np.zeros(locs.shape[:-1])
        dist[:] = np.nan

        for ndx in range(self.conf.batch_size):
            for cls in range(self.conf.n_classes):
                for i_ndx in range(self.info[ndx][2][0]):
                    cur_locs = locs[ndx, i_ndx, cls, ...]
                    if np.isnan(cur_locs[0]):
                        continue
                    cur_dist = l2_dist(pred_locs[ndx, :, cls, :], cur_locs)
                    closest = np.argmin(cur_dist)
                    dist[ndx,i_ndx,cls] += cur_dist.min()

    def classify_val_m(self):
        val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        num_val = 0
        for record in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        self.init_train(train_type=0)
        self.pred = self.create_network()
        saver = self.create_saver()

        with tf.Session() as sess:
            start_at = self.init_and_restore(sess, True, ['loss', 'dist'])

            val_dist = []
            val_ims = []
            val_preds = []
            val_locs = []
            for step in range(num_val/self.conf.batch_size):
                self.setup_val(sess)
                cur_pred = sess.run(self.pred, self.fd)
                cur_dist = self.compute_dist_m(cur_pred,self.locs)
                val_ims.append(self.xs)
                val_locs.append(self.locs)
                val_preds.append(cur_pred)
                val_dist.append(cur_dist)
            self.close_cursors()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape( (-1,) + in_a.shape[2:])

        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_locs = val_reshape(val_locs)

        return val_dist, val_ims, val_preds, val_locs


class PoseCommonTime(PoseCommon):

    def create_cursors(self, sess):
        train_ims, train_locs, train_info = \
            multiResData.read_and_decode_time(self.train_queue, self.conf)
        val_ims, val_locs, val_info = \
            multiResData.read_and_decode_time(self.val_queue, self.conf)
        self.train_data = [train_ims, train_locs, train_info]
        self.val_data = [val_ims, val_locs, val_info]
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def read_images(self, db_type, distort, sess, shuffle=None):
        conf = self.conf
        cur_data = self.val_data if (db_type == self.DBType.Val)\
            else self.train_data
        xs = []
        locs = []
        info = []

        if shuffle is None:
            shuffle = distort

        # Tfrecords doesn't allow shuffling. Skipping a random
        # number of records
        # as a way to simulate shuffling. very hacky.

        for _ in range(conf.batch_size):
            if shuffle:
                for _ in range(np.random.randint(100)):
                    sess.run(cur_data)
            [cur_xs, cur_locs, cur_info] = sess.run(cur_data)
            xs.append(cur_xs)
            locs.append(cur_locs)
            info.append(cur_info)
        xs = np.array(xs)
        tw = (2*conf.time_window_size + 1)
        b_sz = conf.batch_size * tw
        xs = xs.reshape( (b_sz, ) + xs.shape[2:])
        locs = np.array(locs)
        locs = multiResData.sanitize_locs(locs)

        xs = PoseTools.adjust_contrast(xs, conf)

        # ideally normalize_mean should be here, but misc.imresize in scale_images
        # messes up dtypes. It converts float64 back to uint8.
        # so for now it'll be in update_fd.
        # xs = PoseTools.normalize_mean(xs, conf)
        if distort:
            if conf.horzFlip:
                xs, locs = PoseTools.randomly_flip_lr(xs, locs, tw)
            if conf.vertFlip:
                xs, locs = PoseTools.randomly_flip_ud(xs, locs, tw)
            xs, locs = PoseTools.randomly_rotate(xs, locs, conf, tw)
            xs, locs = PoseTools.randomly_translate(xs, locs, conf, tw)
            xs = PoseTools.randomly_adjust(xs, conf, tw)
        # else:
        #     rows, cols = xs.shape[2:]
        #     for ndx in range(xs.shape[0]):
        #         orig_im = copy.deepcopy(xs[ndx, ...])
        #         ii = copy.deepcopy(orig_im).transpose([1, 2, 0])
        #         mat = np.float32([[1, 0, 0], [0, 1, 0]])
        #         ii = cv2.warpAffine(ii, mat, (cols, rows))
        #         if ii.ndim == 2:
        #             ii = ii[..., np.newaxis]
        #         ii = ii.transpose([2, 0, 1])
        #         xs[ndx, ...] = ii

        self.xs = xs
        self.locs = locs
        self.info = info


class PoseCommonRNN(PoseCommonTime):

    def create_cursors(self, sess):
        train_ims, train_locs, train_info = \
            multiResData.read_and_decode_rnn(self.train_queue, self.conf)
        val_ims, val_locs, val_info = \
            multiResData.read_and_decode_rnn(self.val_queue, self.conf)
        self.train_data = [train_ims, train_locs, train_info]
        self.val_data = [val_ims, val_locs, val_info]
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def read_images(self, db_type, distort, sess, shuffle=None):
        conf = self.conf
        cur_data = self.val_data if (db_type == self.DBType.Val)\
            else self.train_data
        xs = []
        locs = []
        info = []

        if shuffle is None:
            shuffle = distort

        for _ in range(conf.batch_size):
            if shuffle:
                for _ in range(np.random.randint(100)):
                    sess.run(cur_data)
            [cur_xs, cur_locs, cur_info] = sess.run(cur_data)
            xs.append(cur_xs)
            locs.append(cur_locs)
            info.append(cur_info)
        xs = np.array(xs)
        tw = (conf.rnn_before + conf.rnn_after + 1)
        b_sz = conf.batch_size * tw
        xs = xs.reshape( (b_sz, ) + xs.shape[2:])
        locs = np.array(locs)
        locs = multiResData.sanitize_locs(locs)

        xs = PoseTools.adjust_contrast(xs, conf)

        # ideally normalize_mean should be here, but misc.imresize in scale_images
        # messes up dtypes. It converts float64 back to uint8.
        # so for now it'll be in update_fd.
        # xs = PoseTools.normalize_mean(xs, conf)
        if distort:
            if conf.horzFlip:
                xs, locs = PoseTools.randomly_flip_lr(xs, locs, tw)
            if conf.vertFlip:
                xs, locs = PoseTools.randomly_flip_ud(xs, locs, tw)
            xs, locs = PoseTools.randomly_rotate(xs, locs, conf, tw)
            xs, locs = PoseTools.randomly_translate(xs, locs, conf, tw)
            xs = PoseTools.randomly_adjust(xs, conf, tw)
        # else:
        #     rows, cols = xs.shape[2:]
        #     for ndx in range(xs.shape[0]):
        #         orig_im = copy.deepcopy(xs[ndx, ...])
        #         ii = copy.deepcopy(orig_im).transpose([1, 2, 0])
        #         mat = np.float32([[1, 0, 0], [0, 1, 0]])
        #         ii = cv2.warpAffine(ii, mat, (cols, rows))
        #         if ii.ndim == 2:
        #             ii = ii[..., np.newaxis]
        #         ii = ii.transpose([2, 0, 1])
        #         xs[ndx, ...] = ii

        self.xs = xs
        self.locs = locs
        self.info = info

