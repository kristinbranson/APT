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
# from batch_norm import batch_norm_mine as batch_norm
from matplotlib import pyplot as plt
import copy
import cv2
import gc
import resource
import json
import threading
import logging
from six import reraise as raise_


def conv_relu(x_in, kernel_shape, train_phase):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, train_phase)
    return tf.nn.relu(conv + biases)


def conv_relu3(x_in, n_filt, train_phase, keep_prob=None,use_leaky=False):
    in_dim = x_in.get_shape().as_list()[3]
    kernel_shape = [3, 3, in_dim, n_filt]
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, decay=0.99, is_training=train_phase)

    if keep_prob is not None:
        conv = tf.nn.dropout(conv, keep_prob)

    if use_leaky:
        return tf.nn.leaky_relu(conv+biases)
    else:
        return tf.nn.relu(conv + biases)


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
    conv = batch_norm(conv, decay=0.99, is_training=train_phase)
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
        self.dep_nets = None
        self.joint = False
        self.edge_ignore = 0 # amount of edge to ignore while computing prediction locations
        self.train_loss = np.zeros(500) # keep track of past loss for importance sampling
        self.step = 0 # might be required for mdn.
        self.extra_data = None
        self.db_name = ''
        self.read_and_decode = multiResData.read_and_decode
        self.scale = 1
        self.extra_info_sz = 1

        self.q_placeholder_spec = []
        self.q_placeholders = []
        self.q_fns = []
        self.for_training = 0
        self.train_data_name = None


    def create_q_specs(self):
        # preloading queues specifications.
        extra_info_sz = self.extra_info_sz
        scale = self.scale
        imsz = self.conf.imsz
        self.q_placeholder_spec.append(['images', [self.conf.batch_size, imsz[0]//scale, imsz[1]//scale, self.conf.imgDim]])
        self.q_placeholder_spec.append(['locs',  [self.conf.batch_size, self.conf.n_classes, 2]])
        self.q_placeholder_spec.append(['orig_images', [self.conf.batch_size, imsz[0], imsz[1], self.conf.imgDim]])
        self.q_placeholder_spec.append(['orig_locs',  [self.conf.batch_size, self.conf.n_classes, 2]])
        self.q_placeholder_spec.append(['info',  [self.conf.batch_size, 3]])
        self.q_placeholder_spec.append(['extra_info', [self.conf.batch_size, extra_info_sz]])
        self.q_fns.append(self.create_update_fd_fn())


    def open_db_meta(self):
        # When we reload using the graph, we don't need to recreate the queues.
        graph = tf.get_default_graph()
        self.train_enqueue_op = graph.get_operation_by_name('trainq_enqueue')
        self.train_dequeue_op = graph.get_operation_by_name('trainq_Dequeue')
        self.val_enqueue_op = graph.get_operation_by_name('valq_enqueue')
        self.val_dequeue_op = graph.get_operation_by_name('valq_Dequeue')

        saved_batch_size = self.train_enqueue_op.inputs[1].get_shape().as_list()[0]
        if saved_batch_size != self.conf.batch_size:
            self.conf.batch_size = saved_batch_size
            logging.info('Over-riding the batch size with the batch size that was used to train the model')

        self.create_q_specs()
        self.q_placeholders = []
        for name,_ in self.q_placeholder_spec:
            cur_tensor = graph.get_tensor_by_name('{}:0'.format(name))
            self.q_placeholders.append([name,cur_tensor])

        self.locs_op = graph.get_tensor_by_name('locs_op:0')
        self.info_op = graph.get_tensor_by_name('info_op:0')
        self.orig_xs_op = graph.get_tensor_by_name('orig_images_op:0')
        self.orig_locs_op = graph.get_tensor_by_name('orig_locs_op:0')
        self.extra_data_op = graph.get_tensor_by_name('extra_data_op:0')


    def open_dbs(self):
        # name is legacy. Now it just sets up the pre loading threads

        self.create_q_specs()
        assert self.train_type is not None, 'traintype has not been set'
        # if self.train_type == 0:
        #     train_filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + self.db_name + '.tfrecords'
        #     val_filename = os.path.join(self.conf.cachedir, self.conf.valfilename) + self.db_name + '.tfrecords'
        #     self.train_queue = tf.train.string_input_producer([train_filename])
        #     self.val_queue = tf.train.string_input_producer([val_filename])
        # else:
        #     train_filename = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename) + self.db_name + '.tfrecords'
        #     val_filename = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename) + self.db_name + '.tfrecords'
        #     self.train_queue = tf.train.string_input_producer([train_filename])
        #     self.val_queue = tf.train.string_input_producer([val_filename])
        #
        placeholders = [[name, tf.placeholder(tf.float32, shape=spec,name=name)] \
                        for (name, spec) in self.q_placeholder_spec] #.items()}
        self.q_placeholders = placeholders
        names = []
        shapes = []
        for k,v in self.q_placeholder_spec:
            names.append(k)
            shapes.append(v)
        placeholders_list = [v for k,v in placeholders]

        QUEUE_SIZE = 50

        q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(names), shapes=shapes, name='trainq')
        enqueue_op = q.enqueue(placeholders_list)
        dequeue_op = q.dequeue()

        self.train_enqueue_op = enqueue_op
        self.train_dequeue_op = dequeue_op

#        self.train_qr = tf.train.QueueRunner(q, [enqueue_op]*4)

        q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(names), shapes=shapes, name='valq')
        enqueue_op = q.enqueue(placeholders_list)
        dequeue_op = q.dequeue()

        self.val_enqueue_op = enqueue_op
        self.val_dequeue_op = dequeue_op
 #       self.val_qr = tf.train.QueueRunner(q, [enqueue_op]*4)


    def create_cursors(self, sess, distort, shuffle):
        # start the preloading threads.

        # tdata = self.read_and_decode(self.train_queue, self.conf)
        # vdata = self.read_and_decode(self.val_queue, self.conf)
        # self.train_data = tdata
        # self.val_data = vdata
        self.coord = tf.train.Coordinator()
        scale = self.scale

        train_threads = []
        val_threads = []

        if self.for_training == 0:
            # for training
            n_threads = 10
        elif self.for_training == 1:
            # for prediction
            n_threads = 0
        elif self.for_training == 2:
            # for cross validation
            n_threads = 1
        else:
            traceback = sys.exc_info()[2]
            raise_(ValueError, "Inocrrect value for for_training", traceback)

        for _ in range(n_threads):

            train_t = threading.Thread(target=self.read_image_thread,
                                 args=(sess, self.DBType.Train, distort, shuffle, scale))
            train_t.start()
            train_threads.append(train_t)

            val_t = threading.Thread(target=self.read_image_thread,
                                       args=(sess, self.DBType.Val, False, False, scale))
            val_t.start()
            val_threads.append(val_t)

        # self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)
        # self.val_threads1 = self.val_qr.create_threads(sess, coord=self.coord, start=True)
        # self.train_threads1 = self.train_qr.create_threads(sess, coord=self.coord, start=True)
        self.train_threads = train_threads
        self.val_threads = val_threads


    def close_cursors(self):
        try:
            self.coord.request_stop()
            self.coord.join(self.threads)
            self.coord.join(self.train_threads)
            self.coord.join(self.val_threads)
        except RuntimeError as e:
            pass


    def read_image_thread(self, sess, db_type, distort, shuffle, scale):
        # Thread that does the pre processing.

        if self.train_type == 0:
            if db_type == self.DBType.Val:
                filename = os.path.join(self.conf.cachedir, self.conf.valfilename) + '.tfrecords'
            elif db_type == self.DBType.Train:
                filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + '.tfrecords'
            else:
                traceback = sys.exc_info()[2]
                raise_(IOError, "Unspecified DB Type", traceback)

        else:
            filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + '.tfrecords'

        cur_db = multiResData.tf_reader(self.conf, filename, shuffle)
        placeholders = self.q_placeholders

        print('Starting preloading thread of type ... {}'.format(db_type))
        batch_np = {}
        while not self.coord.should_stop():
            batch_in = cur_db.next()
            batch_np['orig_images'] = batch_in[0]
            batch_np['orig_locs'] = batch_in[1]
            batch_np['info'] = batch_in[2]
            batch_np['extra_info'] = batch_in[3]
            xs, locs = PoseTools.preprocess_ims(batch_np['orig_images'], batch_np['orig_locs'], self.conf,
                                                distort, scale)

            batch_np['images'] = xs
            batch_np['locs'] = locs

            for fn in self.q_fns:
                fn(batch_np)

            food = {pl: batch_np[name] for (name, pl) in placeholders}

            success = False
            run_options = tf.RunOptions(timeout_in_ms=30000)
            try:
                while not success:

                    if sess._closed or self.coord.should_stop():
                        return

                    try:
                        if db_type == self.DBType.Val:
                            sess.run(self.val_enqueue_op, feed_dict=food,options=run_options)
                        elif db_type == self.DBType.Train:
                            sess.run(self.train_enqueue_op, feed_dict=food, options=run_options)
                        success = True

                    except tf.errors.DeadlineExceededError:
                        pass

            except (tf.errors.CancelledError,) as e:
                return
            except Exception as e:
                logging.exception('Error in preloading thread')
                self.close_cursors()
                sys.exit(1)
                return


    def dequeue_thread_op(self, sess, db_type):
        if db_type == self.DBType.Val:
            batch_out = sess.run(self.val_dequeue_op)
        elif db_type == self.DBType.Train:
            batch_out = sess.run(self.train_dequeue_op)
        else:
            raise IOError, 'Unspecified DB Type'

        names = [k for k,v in self.q_placeholders]
        batch = {}
        for idx, name in enumerate(names):
            batch[name] = batch_out[idx]

        self.orig_xs = batch['orig_images']
        self.orig_locs = batch['orig_locs']
        self.xs = batch['images']
        self.locs = batch['locs']
        self.info = batch['info']
        self.extra_data = batch['extra_info']

        return batch


    def read_images(self, db_type, distort, sess, shuffle=None, scale = 1):
        conf = self.conf
        cur_data = self.val_data if (db_type == self.DBType.Val)\
            else self.train_data
        xs = []
        locs = []
        info = []
        extra_data = []

        if shuffle is None:
            shuffle = distort

        # Tfrecords doesn't allow shuffling. Skipping a random
        # number of records
        # as a way to simulate shuffling. very hacky.

        for _ in range(conf.batch_size):
            if shuffle:
                for _ in range(np.random.randint(30)):
                    sess.run(cur_data)
            A = sess.run(cur_data)
            xs.append(A[0])
            locs.append(A[1])
            info.append(A[2])
            if len(A)>3:
                extra_data.append(A[3:])
        xs = np.array(xs)
        locs = np.array(locs)
        locs = multiResData.sanitize_locs(locs)

        xs, locs = PoseTools.preprocess_ims(xs, locs, conf, distort, scale)
        self.xs = xs
        self.locs = locs
        self.info = info
        self.extra_data = extra_data


    def get_latest_model_file(self):
        ckpt_file = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + self.name + '_ckpt')
        latest_ckpt = tf.train.get_checkpoint_state(
            self.conf.cachedir, ckpt_file)
        return latest_ckpt.model_checkpoint_path


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

        saver['ckpt_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name + '_ckpt')
        saver['saver'] = (tf.train.Saver(var_list=PoseTools.get_vars(net_name+ '/'),
                                         max_to_keep=self.conf.maxckpt,
                                         save_relative_paths=True))
        self.saver = saver
        if self.dep_nets:
            self.dep_nets.create_joint_saver(self.name)


    def create_joint_saver(self, o_name):
        saver = {}
        name = o_name + '_' + self.name
        saver['out_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name)
        saver['train_data_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name + '_traindata')
        saver['ckpt_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name + '_ckpt')
        saver['saver'] = (tf.train.Saver(var_list=PoseTools.get_vars(self.net_name + '/'),
                                         max_to_keep=self.conf.maxckpt,
                                         save_relative_paths=True))
        self.saver = saver
        if self.dep_nets:
            self.dep_nets.create_joint_saver(name)


    def restore(self, sess, do_restore, at_step=-1):
        saver = self.saver
        name = self.net_name
        out_file = saver['out_file'].replace('\\', '/')
        latest_ckpt = tf.train.get_checkpoint_state(
            self.conf.cachedir, saver['ckpt_file'])
        if not latest_ckpt or not do_restore:
            start_at = 0
            sess.run(tf.variables_initializer(
                PoseTools.get_vars(name)),
                feed_dict=self.fd)
            print("Not loading {:s} variables. Initializing them".format(name))
        else:
            if at_step < 0:
                saver['saver'].restore(sess, latest_ckpt.model_checkpoint_path)
                match_obj = re.match(out_file + '-(\d*)', latest_ckpt.model_checkpoint_path)
                start_at = int(match_obj.group(1)) + 1
            else:
                aa = latest_ckpt.all_model_checkpoint_paths
                model_file = ''
                for a in aa:
                    match_obj = re.match(out_file + '-(\d*)', a)
                    step = int(match_obj.group(1))
                    if step >= at_step:
                        model_file = a
                        break
                saver['saver'].restore(sess, model_file)
                match_obj = re.match(out_file + '-(\d*)', model_file)
                start_at = int(match_obj.group(1)) + 1

        if self.dep_nets:
            self.dep_nets.restore_joint(sess, self.name, self.joint, do_restore)

        return start_at


    def restore_joint(self, sess, o_name, joint_train, do_restore):
        # when to restore and from to restore is kinda complicated.
        # if not doing joint training, then always restore from trained dependent model.
        # if doing joint training, but not restoring then again load from trained dependent model.
        # if doing joint training, and restoring but nothing exists then again load from trained dependent model.
        # in other words only load from jointly trained dependent model, if joint_train, do_restore and
        # an earlier saved model exists
        saver = self.saver
        name = o_name + '_' + self.name
        if joint_train and do_restore:
            latest_ckpt = tf.train.get_checkpoint_state(
                self.conf.cachedir, saver['ckpt_file'])
            if not latest_ckpt:
                ckpt_file = os.path.join(self.conf.cachedir,
                    self.conf.expname + '_' + self.name + '_ckpt')
                latest_ckpt = tf.train.get_checkpoint_state(
                    self.conf.cachedir, ckpt_file)
                assert latest_ckpt, 'dependent network {} hasnt been trained'.format(self.name)
        else:
            ckpt_file = os.path.join(self.conf.cachedir,
                                     self.conf.expname + '_' + self.name + '_ckpt')
            latest_ckpt = tf.train.get_checkpoint_state(
                self.conf.cachedir, ckpt_file)
            assert latest_ckpt, 'dependent network {} hasnt been trained'.format(self.name)

        saver['saver'].restore(sess, latest_ckpt.model_checkpoint_path)
        print("Loading {:s} variables from {:s}".format(name, latest_ckpt.model_checkpoint_path))


    def save(self, sess, step):
        saver = self.saver
        out_file = saver['out_file'].replace('\\', '/')
        saver['saver'].save(sess, out_file, global_step=step,
                            latest_filename=os.path.basename(saver['ckpt_file']))
        print('Saved state to %s-%d' % (out_file, step))
        if self.dep_nets and self.joint:
            self.dep_nets.save(sess, step)


    def init_td(self, td_fields):
        ex_td_fields = ['step']
        for t_f in td_fields:
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
        if self.dep_nets:
            self.dep_nets.create_ph_fd()

    # def create_fd(self):
    #     # to be subclassed
    #     return None

    def fd_train(self):
        assert False, 'Write fd_train for the subclass'

    def fd_val(self):
        assert False, 'Write fd_val for the subclass'

    def update_fd(self, db_type, sess, distort):
        assert False, 'Write update_fd for the subclass'

    def create_update_fd_fn(self):
        assert False, 'Write create_fd_fn for the subclass'


    def init_train(self, train_type):
        self.train_type = train_type
        self.open_dbs()
        self.create_ph_fd()
#        self.create_fd()


    def init_and_restore(self, sess, restore, td_fields, distort, shuffle, at_step=-1):
        self.create_cursors(sess,distort,shuffle)
        #self.update_fd(db_type=self.DBType.Train, sess=sess, distort=True)
        start_at = self.restore(sess, restore, at_step)
        initialize_remaining_vars(sess)

        try:
            self.init_td(td_fields) if start_at is 0 else self.restore_td()
        except AttributeError: # If the conf file has been modified
            print('----------------')
            print("Couldn't load train data because the conf has changed!")
            print('----------------')
            self.init_td(td_fields)

        return start_at


    def train_step(self, step, sess, learning_rate, training_iters, n_steps):
        cur_step = float(step)
        # cur_lr = learning_rate * self.conf.gamma ** math.floor(old_div(ex_count, self.conf.step_size))

        # for most cases the learning rate should decay to 1/100 - 1/1000 by the end of the training.

        cur_lr = learning_rate * (self.conf.gamma ** (cur_step*n_steps/ training_iters))
        self.fd[self.ph['learning_rate']] = cur_lr
        self.fd_train()
        self.update_fd(db_type=self.DBType.Train, sess=sess, distort=True)
        # doTrain = False
        # while not doTrain: # importance sampling
        #     self.update_fd(self.DBType.Train, sess, True)
        #     cur_loss = sess.run(self.cost,self.fd)
        #     x0 = np.median(self.train_loss)
        #     # x1 = np.percentile(self.train_loss,75)
        #     cur_tr = 1/(1+np.exp(-(cur_loss-x0)/x0))
        #     doTrain = (np.random.rand() < cur_tr)
        #     # cur_tr = np.percentile(self.train_loss,50)*np.random.rand()*2
        #     # if loss is approx 80 perc, then train half the time.
        #     # if loss > 2 * 80 perc, then always train.
        #     # doTrain = (cur_loss > cur_tr)
        #     self.train_loss[:-1] = self.train_loss[1:]
        #     self.train_loss[-1] = cur_loss

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        sess.run(self.opt, self.fd,options=run_options)


    def setup_train(self, sess, distort=True, treat_as_val = False):
        if treat_as_val:
            self.fd_val()
        else:
            self.fd_train()
        self.update_fd(db_type=self.DBType.Train, sess=sess, distort=distort)


    def setup_val(self, sess):
        self.fd_val()
        self.update_fd(db_type=self.DBType.Val, sess=sess, distort=False)


    def create_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # plain vanilla.
            # self.opt = tf.train.AdamOptimizer(
            #     learning_rate=self.ph['learning_rate']).minimize(self.cost)

            # clipped gradients.
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.ph['learning_rate'])
            gradients, variables = zip(*optimizer.compute_gradients(self.cost))
            gradients = [ None if gradient is None else
                          tf.clip_by_norm(gradient, 5.0)
                    for gradient in gradients]
            self.opt = optimizer.apply_gradients(zip(gradients,variables))


    def compute_dist(self, preds, locs):
        tt1 = PoseTools.get_pred_locs(preds,self.edge_ignore) - locs
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        return np.nanmean(tt1)


    def compute_train_data(self, sess, db_type):
        self.setup_train(sess) if db_type is self.DBType.Train \
            else self.setup_val(sess)
        cur_loss, cur_pred, self.locs, self.info, self.extra_data, cur_im, cur_label = \
            sess.run( [self.cost, self.pred, self.locs_op, self.info_op, self.extra_data_op,
                       self.ph['x'], self.ph['y']], self.fd)

        cur_dist = self.compute_dist(cur_pred, self.locs)
        return cur_loss, cur_dist


    def train(self, restore, train_type, create_network,
              training_iters, loss, pred_in_key, learning_rate,
              td_fields=('loss', 'dist')):

        self.init_train(train_type)
        self.pred = create_network()
        self.cost = loss(self.pred, self.ph[pred_in_key])
        self.create_optimizer()
        self.create_saver()
        num_val_rep = self.conf.numTest / self.conf.batch_size + 1

        with tf.Session() as sess:
            start_at = self.init_and_restore( sess, restore, td_fields, True, True)

            if start_at < training_iters:
                for step in range(start_at, training_iters + 1):
                    self.train_step(step, sess, learning_rate, training_iters, self.conf.n_steps)
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
            self.close_cursors()


    def init_net_common(self, create_network_fn, train_type=0, restore=True):
        print('--- Loading the model by reconstructing the graph ---')
        self.init_train(train_type=train_type)
        self.pred = create_network_fn()
        saver = self.create_saver()
        self.joint = True
        sess = tf.InteractiveSession()
        start_at = self.init_and_restore(sess, restore, ['loss', 'dist'])
        return sess


    def init_net_meta(self, train_type, model_file):
        print('--- Loading the model using the saved graph ---')
        sess = tf.Session()
        self.train_type = train_type
        try:
          # self.open_dbs()
#            self.create_ph_fd()
            latest_model_file = self.restore_meta(self.name, sess, model_file)
            self.open_db_meta()
            self.create_cursors(sess,distort=False, shuffle=False)
        except tf.errors.NotFoundError:
            pass

        return sess, latest_model_file



    def restore_meta(self, name, sess, model_file=None):
        if self.dep_nets:
            if type(self.dep_nets) is list:
                for dd in self.dep_nets:
                    dd.restore_meta(name + '_' + dd.name, sess)
            else:
                self.dep_nets.restore_meta(name + '_' + self.dep_nets.name, sess)

        if model_file is None:
            ckpt_file = os.path.join( self.conf.cachedir, self.conf.expname + '_' + name + '_ckpt')
            latest_ckpt = tf.train.get_checkpoint_state( self.conf.cachedir, ckpt_file)
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

    def plot_results(self):
        saver = {}
        saver['train_data_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + self.name + '_traindata')
        self.saver = saver
        self.restore_td()
        n = 50
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

