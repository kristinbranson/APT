from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

from PoseTrain import PoseTrain
import os
import sys
import tensorflow as tf
import PoseTools
import re
import pickle
from tensorflow.contrib import slim
from tensorflow.contrib.layers import batch_norm
from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
import edward as ed
import math
import numpy as np
import mymap
# from batch_norm import batch_norm_2D
import convNetBase as CNB


def conv_relu(X, kernel_shape, trainPhase, sc_name,stride=1):
    with tf.variable_scope(sc_name):
        weights = tf.get_variable("weights", kernel_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", kernel_shape[-1],
                                 initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(X, weights,
                            strides=[1, stride, stride, 1], padding='SAME')
        conv = batch_norm(conv, decay=0.99,
                          is_training=trainPhase)
    return tf.nn.relu(conv + biases)


def fully_conn(X, out_wts, trainPhase, sc_name):
    X_flat = slim.flatten(X)
    in_dim = X_flat.get_shape().as_list()[-1]
    with tf.variable_scope(sc_name):
        weights = tf.get_variable("weights", [in_dim, out_wts],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", [out_wts],
                                 initializer=tf.constant_initializer(0))
        hidden = tf.matmul(X_flat, weights)
        hidden = batch_norm(hidden, decay=0.99,
                            is_training=trainPhase)
    return tf.nn.relu(hidden + biases)

class PoseMDN(PoseTrain):
    def __init__(self, conf):
        super(self.__class__, self).__init__(conf)
        self.mdn_saver = None
        self.mdn_start_at = 0
        self.mdn_train_data = {}
        self.baseLayers = {}
        self.basePred = None
        self.mdn_locs = None
        self.mdn_scales = None
        self.mdn_logits = None
        self.trainType = None
        self.loss = None
        self.opt = None

    def create_saver(self):
        self.mdn_saver = tf.train.Saver(var_list=PoseTools.get_vars('mdn'),
                                        max_to_keep=self.conf.maxckpt)

    def restore(self, sess, restore):
        out_filename = os.path.join(self.conf.cachedir, self.conf.expname + '_MDN')
        out_filename = out_filename.replace('\\', '/')
        train_data_filename = os.path.join(self.conf.cachedir, self.conf.expname + '_MDN_traindata')
        ckpt_filename = self.conf.expname + '_MDN_ckpt'
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                                    latest_filename=ckpt_filename)
        if not latest_ckpt or not restore:
            self.mdn_start_at = 0
            self.mdn_train_data = {'train_err': [], 'val_err': [], 'step_no': [],
                                   'train_dist': [], 'val_dist': []}
            sess.run(tf.variables_initializer(PoseTools.get_vars('mdn')), feed_dict=self.feed_dict)
            print("Not loading MDN variables. Initializing them")
            return False
        else:
            self.mdn_saver.restore(sess, latest_ckpt.model_checkpoint_path)
            match_obj = re.match(out_filename + '-(\d*)', latest_ckpt.model_checkpoint_path)
            self.mdn_start_at = int(match_obj.group(1)) + 1
            with open(train_data_filename, 'rb') as td_file:
                if sys.version_info.major == 3:
                    in_data = pickle.load(td_file, encoding='latin1')
                else:
                    in_data = pickle.load(td_file)

                if not isinstance(in_data, dict):
                    self.mdn_train_data, load_conf = in_data
                    print('Parameters that dont match for base:')
                    PoseTools.compare_conf(self.conf, load_conf)
                else:
                    print("No config was stored for base. Not comparing conf")
                    self.mdn_train_data = in_data
            print("Loading MDN variables from %s" % latest_ckpt.model_checkpoint_path)
            return True


    def save(self, sess, step):
        out_filename = os.path.join(self.conf.cachedir, self.conf.expname + '_MDN')
        out_filename = out_filename.replace('\\', '/')
        train_data_filename = os.path.join(self.conf.cachedir, self.conf.expname + '_MDN_traindata')
        ckpt_filename = self.conf.expname + '_MDN_ckpt'
        self.mdn_saver.save(sess, out_filename, global_step=step,
                            latest_filename=ckpt_filename)
        print('Saved state to %s-%d' % (out_filename, step))
        with open(train_data_filename, 'wb') as td_file:
            pickle.dump([self.mdn_train_data, self.conf], td_file, protocol=2)


    def restore_base_full(self, sess, full):
        outfilename = os.path.join(self.conf.cachedir, self.conf.expname + self.conf.baseName+'_MDN')
        outfilename = outfilename.replace('\\', '/')
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.basedataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                        latest_filename=self.conf.expname + self.conf.baseName +'ckpt_MDN')
        if not latest_ckpt or not full:
            self.restoreBase(sess,True)
        else:
            self.basesaver.restore(sess, latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)', latest_ckpt.model_checkpoint_path)
            self.basestartat = int(matchObj.group(1)) + 1
            with open(traindatafilename, 'rb') as tdfile:
                if sys.version_info.major == 3:
                    inData = pickle.load(tdfile, encoding='latin1')
                else:
                    inData = pickle.load(tdfile)

                if not isinstance(inData, dict):
                    self.basetrainData, loadconf = inData
                    print('Parameters that dont match for base:')
                    PoseTools.compare_conf(self.conf, loadconf)
                else:
                    print("No config was stored for base. Not comparing conf")
                    self.basetrainData = inData
            print("Loading base variables from %s" % latest_ckpt.model_checkpoint_path)

    def save_base_full(self, sess, step):
        outfilename = os.path.join(self.conf.cachedir, self.conf.expname + self.conf.baseName +'_MDN')
        self.basesaver.save(sess, outfilename, global_step=step,
                            latest_filename=self.conf.expname + self.conf.baseName +'ckpt_MDN')
        print('Saved state to %s-%d' % (outfilename, step))


    def mdn_network_slim(self, X, K, n_out,max_loc):
        """loc, scale, logits = NN(x; theta)"""
        # 2 hidden layers with 15 hidden units
        x_flat = slim.flatten(X)
        hidden1 = slim.fully_connected(x_flat, 400,
                                       normalizer_fn=slim.batch_norm,
                                       normalizer_params={'is_training': self.ph['phase_train_mdn'],
                                                                         'decay': 0.95}
                                       )
        hidden2 = slim.fully_connected(hidden1, 400,
                                       normalizer_fn=slim.batch_norm,
                                       normalizer_params = {'is_training': self.ph['phase_train_mdn'],
                                                            'decay': 0.95})
        o_locs = slim.fully_connected(hidden2, 2 * K * n_out,
                                    activation_fn=None)
        min_loc = 0.
        # locs = tf.minimum(max_loc, tf.maximum(min_loc, o_locs))
        locs = o_locs
        o_scales = slim.fully_connected(hidden2, 2 * K * n_out,
                                        activation_fn=tf.exp)
        min_sig = self.conf.mdn_min_sigma
        max_sig = self.conf.mdn_max_sigma
        scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
        logits = slim.fully_connected(hidden2, K * n_out,
                                      activation_fn=None)
        locs = tf.reshape(locs, [-1, K, n_out, 2])
        scales = tf.reshape(scales, [-1, K, n_out, 2])
        logits = tf.reshape(logits, [-1, K, n_out])
        return locs, scales, logits, hidden1, hidden2

    def mdn_network(self, X, K, n_out,max_loc,reuse):
        """loc, scale, logits = NN(x; theta)"""
        # 2 hidden layers with 15 hidden units
        x_flat = slim.flatten(X)
        grid_size = 16

        with tf.variable_scope('layer_1') as scope:
            in_dim = x_flat.get_shape().as_list()[1]
            in_dim_x = X.get_shape().as_list()[3]
            kernel_shape = [grid_size, grid_size, in_dim_x, 400]
            weights1 = tf.get_variable("weights", kernel_shape,
                                      initializer=tf.contrib.layers.xavier_initializer())
            weights1 = tf.reshape(weights1, [16*16*in_dim_x,400])
            # weights1 = tf.get_variable("weights1", [in_dim,400],
            #                            initializer=tf.contrib.keras.initializers.TruncatedNormal(stddev=0.004))
            #                           # initializer=tf.contrib.layers.xavier_initializer())
            biases1 = tf.get_variable("biases", 400,
                                     initializer=tf.constant_initializer(0))
            hidden1_b = tf.matmul(x_flat, weights1)
            with tf.variable_scope('BatchNorm') as cur_scope:
                hidden1 = batch_norm(hidden1_b, decay=0.999,
                                     is_training=self.ph['phase_train_mdn'],
                                     reuse=reuse, scope=cur_scope)
            hidden1 = tf.nn.relu(hidden1 + biases1)

        with tf.variable_scope('layer_2') as scope:
            # weights2 = tf.get_variable("weights2", [400,400],
            #                           initializer=tf.contrib.layers.xavier_initializer())
            kernel_shape = [1, 1, 400, 400]
            weights2 = tf.get_variable("weights", kernel_shape,
                                      initializer=tf.contrib.layers.xavier_initializer())
            weights2 = tf.reshape(weights2, [400,400])
            biases2 = tf.get_variable("biases", 400,
                                     initializer=tf.constant_initializer(0))
            hidden2_b = tf.matmul(hidden1, weights2)
            with tf.variable_scope('BatchNorm') as cur_scope:
                hidden2 = batch_norm(hidden2_b, decay=0.999,
                                     is_training=self.ph['phase_train_mdn'],
                                     reuse=reuse,scope=cur_scope)
            hidden2 = tf.nn.relu(hidden2 + biases2)

        with tf.variable_scope('locs'):
            # weights_locs = tf.get_variable("weights_locs", [400,2 *K *n_out],
            #                           initializer=tf.contrib.layers.xavier_initializer())
            weights_locs = tf.get_variable("weights_locs", [1, 1, 400, 2 * K * n_out],
                                           initializer=tf.contrib.layers.xavier_initializer())
            weights_locs = tf.reshape(weights_locs, [400, 2*K*n_out])
            biases_locs = tf.get_variable("biases_locs", 2*K*n_out,
                                     initializer=tf.constant_initializer(0))
            o_locs = tf.matmul(hidden2,weights_locs)+biases_locs

        with tf.variable_scope('scales'):
            # weights_scales = tf.get_variable("weights_scales", [400,K *n_out],
            #                           initializer=tf.contrib.layers.xavier_initializer())
            weights_scales = tf.get_variable("weights_scales", [1, 1, 400, K * n_out],
                                             initializer=tf.contrib.layers.xavier_initializer())
            weights_scales = tf.reshape(weights_scales, [400, K*n_out])
            biases_scales = tf.get_variable("biases_scales", K*n_out,
                                     initializer=tf.constant_initializer(0))
            o_scales = tf.exp(tf.matmul(hidden2,weights_scales)+biases_scales)
            min_sig = self.conf.mdn_min_sigma
            max_sig = self.conf.mdn_max_sigma
            o_scales = (o_scales - 1) * (max_sig - min_sig) / 2
            o_scales = o_scales + (max_sig - min_sig) / 2 + min_sig
            scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))

        with tf.variable_scope('logits'):
            # weights_logits = tf.get_variable("weights_logits", [400,K *n_out],
            #                           initializer=tf.contrib.layers.xavier_initializer())
            weights_logits = tf.get_variable("weights_logits", [1, 1, 400, K * n_out],
                                             initializer=tf.contrib.layers.xavier_initializer())
            weights_logits = tf.reshape(weights_logits, [400, K*n_out])
            biases_logits = tf.get_variable("biases_logits", K*n_out,
                                     initializer=tf.constant_initializer(0))
            logits = tf.matmul(hidden2,weights_logits) + biases_logits

        o_locs = tf.reshape(o_locs, [-1, K, n_out, 2])
        o_locs = o_locs + max_loc/2
        # o_locs = tf.exp(o_locs/max_loc/8)
        # o_locs = 1.2 * max_loc *(o_locs/(1+o_locs))
        # locs = o_locs - 0.1*max_loc
        # min_loc = 0.
        # locs = tf.minimum(max_loc, tf.maximum(min_loc, o_locs))
        locs = o_locs
        # scales = tf.reshape(scales, [-1, K, n_out, 2])
        scales = tf.reshape(scales, [-1, K, n_out])
        logits = tf.reshape(logits, [-1, K, n_out])
        return locs, scales, logits, hidden1, hidden2, hidden1_b, hidden2_b

    def create_network_ed(self, l7_layer):

        n_nets = 8
        k = 4
        conf = self.conf


        # l7_layer = tf.stop_gradient(self.baseLayers['conv7'])
        # with tf.variable_scope('mdn_in'):
        #     l7_layer= CNB.conv_relu(l7_layer, [1, 1, conf.nfcfilt, 30],
        #                             0.005, 1, doBatchNorm=True,
        #                             trainPhase=self.ph['phase_train_mdn'])


        l7_shape = l7_layer.get_shape().as_list()
        l7_layer_x_sz = l7_shape[2]
        l7_layer_y_sz = l7_shape[1]

        # self.ph['layer7'] = tf.placeholder(tf.float32, l7_shape)
        # l7_layer = self.ph['layer7']

        x_pad = (-l7_layer_x_sz) % n_nets
        y_pad = (-l7_layer_y_sz) % n_nets
        pad = [[0, 0], [0, y_pad], [0, x_pad], [0, 0]]
        l7_layer = tf.pad(l7_layer,pad)
        l7_shape = l7_layer.get_shape().as_list()
        l7_layer_x_sz = l7_shape[2]
        l7_layer_y_sz = l7_shape[1]

        all_locs = []
        all_scales = []
        all_logits = []

        x_psz = l7_layer_x_sz / n_nets
        y_psz = l7_layer_y_sz / n_nets

        assert x_psz == y_psz

        with tf.variable_scope('mdn'):
            for xx in range(n_nets):
                for yy in range(n_nets):
                    x_start = int(xx * x_psz)
                    x_stop = int((xx + 1) * x_psz)
                    y_start = int(yy * y_psz)
                    y_stop = int((yy + 1) * y_psz)
                    cur_patch = l7_layer[:, y_start:y_stop, x_start:x_stop, :]
                    with tf.variable_scope('nn') as scope:
                        if xx or yy:
                            scope.reuse_variables()
                            locs, scales, logits, hidden1, hidden2, \
                                hidden1_b, hidden2_b= self.mdn_network(
                                cur_patch, k, conf.n_classes, [x_psz, y_psz],
                                reuse=True)
                        else:
                            locs, scales, logits, hidden1, hidden2, \
                            hidden1_b, hidden2_b= self.mdn_network(
                                cur_patch, k, conf.n_classes, [x_psz, y_psz],
                                reuse=False)
                            self.hidden1 = hidden1
                            self.hidden2 = hidden2
                            self.hidden1_b = hidden1_b
                            self.hidden2_b = hidden2_b


                    locs += np.array([x_start, y_start])
                    all_locs.append(locs)
                    all_scales.append(scales)
                    all_logits.append(logits)
            all_locs = tf.concat(all_locs, 1)
            all_scales = tf.concat(all_scales, 1)
            all_logits = tf.concat(all_logits, 1)
        self.mdn_locs = all_locs
        self.mdn_scales = all_scales
        self.mdn_logits = all_logits

        y_cls = []
        self.components = []
        for cls in range(self.conf.n_classes):
            cat = Categorical(logits=all_logits[:, :, cls])
            # components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
            #               in zip(tf.unstack(tf.transpose(all_locs[:, :, cls, :], [1, 0, 2])),
            #                      tf.unstack(tf.transpose(all_scales[:, :, cls, :], [1, 0, 2])))]
            components = [MultivariateNormalDiag(loc=loc, scale_identity_multiplier=scale) for loc, scale
                          in zip(tf.unstack(tf.transpose(all_locs[:, :, cls, :], [1, 0, 2])),
                                 tf.unstack(tf.transpose(all_scales[:, :, cls], [1, 0])))]
            self.components.append(components)
            # components = [self.my_multi_normal(loc=loc, scale=scale) for loc, scale
            #               in zip(tf.unstack(tf.transpose(all_locs[:, :, cls, :], [1, 0, 2])),
            #                      tf.unstack(tf.transpose(all_scales[:, :, cls, :], [1, 0, 2])))]
            # temp_tensor = tf.zeros([self.conf.batch_size, 2]) # not sure what this stupidity is
            y_cls.append(Mixture(cat=cat, components=components))
                                 # value=tf.zeros_like(temp_tensor)))

        return y_cls


    def create_network_grid(self, l7_layer):
        # Earlier implementation, in which batch norm variance does
        #  not get computed properly.

        n_nets = 8
        k = 4
        conf = self.conf


        # l7_layer = tf.stop_gradient(self.baseLayers['conv7'])
        # with tf.variable_scope('mdn_in'):
        #     l7_layer= CNB.conv_relu(l7_layer, [1, 1, conf.nfcfilt, 30],
        #                             0.005, 1, doBatchNorm=True,
        #                             trainPhase=self.ph['phase_train_mdn'])


        l7_shape = l7_layer.get_shape().as_list()
        l7_layer_x_sz = l7_shape[2]
        l7_layer_y_sz = l7_shape[1]

        # self.ph['layer7'] = tf.placeholder(tf.float32, l7_shape)
        # l7_layer = self.ph['layer7']

        x_pad = (-l7_layer_x_sz) % n_nets
        y_pad = (-l7_layer_y_sz) % n_nets
        pad = [[0, 0], [0, y_pad], [0, x_pad], [0, 0]]
        l7_layer = tf.pad(l7_layer,pad)
        l7_shape = l7_layer.get_shape().as_list()
        l7_layer_x_sz = l7_shape[2]
        l7_layer_y_sz = l7_shape[1]

        all_locs = []
        all_scales = []
        all_logits = []

        x_psz = l7_layer_x_sz / n_nets
        y_psz = l7_layer_y_sz / n_nets

        assert x_psz == y_psz

        with tf.variable_scope('mdn'):
            all_hidden1 = []
            all_hidden2 = []
            all_hidden1_b = []
            all_hidden2_b = []
            for xx in range(n_nets):
                for yy in range(n_nets):
                    x_start = int(xx * x_psz)
                    x_stop = int((xx + 1) * x_psz)
                    y_start = int(yy * y_psz)
                    y_stop = int((yy + 1) * y_psz)
                    cur_patch = l7_layer[:, y_start:y_stop, x_start:x_stop, :]
                    max_loc = np.array([x_psz,y_psz])
                    with tf.variable_scope('nn') as scope:
                        if xx or yy:
                            scope.reuse_variables()
                            locs, scales, logits, hidden1, hidden2, \
                                hidden1_b, hidden2_b= self.mdn_network(
                                cur_patch, k, conf.n_classes, max_loc,
                                reuse=True)
                        else:
                            locs, scales, logits, hidden1, hidden2, \
                            hidden1_b, hidden2_b= self.mdn_network(
                                cur_patch, k, conf.n_classes, max_loc,
                                reuse=False)
                        all_hidden1.append(hidden1)
                        all_hidden2.append(hidden2)
                        all_hidden1_b.append(hidden1_b)
                        all_hidden2_b.append(hidden2_b)


                    locs += np.array([x_start, y_start])
                    all_locs.append(locs)
                    all_scales.append(scales)
                    all_logits.append(logits)
            all_locs = tf.concat(all_locs, 1)
            all_scales = tf.concat(all_scales, 1)
            all_logits = tf.concat(all_logits, 1)
            self.hidden1 = tf.stack(all_hidden1, axis=1)
            self.hidden2 = tf.stack(all_hidden2, axis=1)
            self.hidden1_b = tf.stack(all_hidden1_b, axis=1)
            self.hidden2_b = tf.stack(all_hidden2_b, axis=1)
        self.mdn_locs = all_locs
        self.mdn_scales = all_scales
        self.mdn_logits = all_logits

    def create_network(self, l7_layer):

        k = 4
        grid_size = 64
        stride_size = 16

        # l7_layer = tf.stop_gradient(self.baseLayers['conv7'])
        # with tf.variable_scope('mdn_in'):
        #     l7_layer= CNB.conv_relu(l7_layer, [1, 1, conf.nfcfilt, 30],
        #                             0.005, 1, doBatchNorm=True,
        #                             trainPhase=self.ph['phase_train_mdn'])

        l7_shape = l7_layer.get_shape().as_list()
        n_out = self.conf.n_classes

        with tf.variable_scope('mdn/nn'):
            with tf.variable_scope('layer_1'):
                kernel_shape = [grid_size, grid_size, l7_shape[3], 400]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(l7_layer, weights,
                                    strides=[1, stride_size, stride_size, 1], padding='SAME')
                l1_b = conv
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train_mdn'])
            mdn_l1 = tf.nn.relu(conv + biases)

            with tf.variable_scope('layer_2'):
                kernel_shape = [1, 1, 400, 400]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(mdn_l1, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                l2_b = conv
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train_mdn'])
            mdn_l2 = tf.nn.relu(conv + biases)

            with tf.variable_scope('locs'):
                weights_locs = tf.get_variable("weights_locs", [1, 1, 400, 2 * k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
                                              initializer=tf.constant_initializer(0))
                o_locs = tf.nn.conv2d(mdn_l2, weights_locs,
                                      [1, 1, 1, 1], padding='SAME') + biases_locs

                loc_shape = o_locs.get_shape().as_list()
                n_x = loc_shape[2]
                n_y = loc_shape[1]
                o_locs = tf.reshape(o_locs,[-1, n_y, n_x, k, n_out, 2])
                # when initialized o_locs will be centered around 0.
                # with adding grid_size/2, o_locs initially will be centered
                # in the center of the grid.
                o_locs += stride_size/2

                # adding offset of each grid location.
                x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
                x_off = x_off * stride_size
                y_off = y_off * stride_size
                x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                x_locs = o_locs[:,:,:,:,:,0] + x_off
                y_locs = o_locs[:,:,:,:,:,1] + y_off
                o_locs = tf.stack([x_locs, y_locs], axis=5)
                o_locs = tf.reshape(o_locs,[-1, n_x*n_y*k,n_out,2])

            with tf.variable_scope('scales'):
                weights_scales = tf.get_variable("weights_scales", [1, 1, 400, k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_scales = tf.get_variable("biases_scales", k * self.conf.n_classes,
                                              initializer=tf.constant_initializer(0))
                o_scales = tf.exp(tf.nn.conv2d(mdn_l2, weights_scales,
                                      [1, 1, 1, 1], padding='SAME') + biases_scales)
                # when initialized o_scales will be centered around exp(0) and
                # mostly between [exp(-1), exp(1)] = [0.3 2.7]
                # so adding appropriate offsets to make it lie between the wanted range
                min_sig = self.conf.mdn_min_sigma
                max_sig = self.conf.mdn_max_sigma
                o_scales = (o_scales-1) * (max_sig-min_sig)/2
                o_scales = o_scales + (max_sig-min_sig)/2 + min_sig
                scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
                scales = tf.reshape(scales, [-1, n_x*n_y,k,n_out])
                scales = tf.reshape(scales,[-1, n_x*n_y*k,n_out])

            with tf.variable_scope('logits'):
                weights_logits = tf.get_variable("weights_logits", [1, 1, 400, k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_logits = tf.get_variable("biases_logits", k * self.conf.n_classes,
                                              initializer=tf.constant_initializer(0))
                logits = tf.nn.conv2d(mdn_l2, weights_logits,
                                      [1, 1, 1, 1], padding='SAME') + biases_logits
                logits = tf.reshape(logits, [-1, n_x * n_y, k, n_out])
                logits = tf.reshape(logits, [-1, n_x * n_y * k, n_out])


        all_locs = o_locs
        self.mdn_locs = all_locs
        self.mdn_scales = scales
        self.mdn_logits = logits
        self.hidden1_b = l1_b
        self.hidden2_b = l2_b
        self.hidden1 = mdn_l1
        self.hidden2 = mdn_l2

    def create_network_joint(self, l7_layer):

        k = 4
        grid_size = 8
        stride_size = 4
        grid_size2 = 8
        stride_size2 = 2
        grid_size3 = 8
        stride_size3 = 2
        locs_offset = stride_size * stride_size2 * stride_size3

        # l7_layer = tf.stop_gradient(self.baseLayers['conv7'])
        # with tf.variable_scope('mdn_in'):
        #     l7_layer= CNB.conv_relu(l7_layer, [1, 1, conf.nfcfilt, 30],
        #                             0.005, 1, doBatchNorm=True,
        #                             trainPhase=self.ph['phase_train_mdn'])

        l7_shape = l7_layer.get_shape().as_list()
        n_out = self.conf.n_classes

        with tf.variable_scope('mdn/nn'):
            with tf.variable_scope('layer_1'):
                kernel_shape = [grid_size, grid_size, l7_shape[3], 400]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(l7_layer, weights,
                                    strides=[1, stride_size, stride_size, 1], padding='SAME')
                l1_b = conv
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train_mdn'])
            mdn_l1a = tf.nn.relu(conv + biases)

            with tf.variable_scope('layer_1b'):
                kernel_shape = [grid_size2, grid_size2, 400, 400]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(mdn_l1a, weights,
                                    strides=[1, stride_size2, stride_size2, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train_mdn'])
            mdn_l1b = tf.nn.relu(conv + biases)

            with tf.variable_scope('layer_1c'):
                kernel_shape = [grid_size3, grid_size3, 400, 400]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(mdn_l1b, weights,
                                    strides=[1, stride_size3, stride_size3, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train_mdn'])
            mdn_l1c = tf.nn.relu(conv + biases)

            # with tf.variable_scope('layer_1d'):
            #     kernel_shape = [grid_size3, grid_size3, 400, 400]
            #     weights = tf.get_variable("weights", kernel_shape,
            #                               initializer=tf.contrib.layers.xavier_initializer())
            #     biases = tf.get_variable("biases", kernel_shape[-1],
            #                              initializer=tf.constant_initializer(0))
            #     conv = tf.nn.conv2d(mdn_l1c, weights,
            #                         strides=[1, stride_size3, stride_size3, 1], padding='SAME')
            #     conv = batch_norm(conv, decay=0.99,
            #                       is_training=self.ph['phase_train_mdn'])
            # mdn_l1c = tf.nn.relu(conv + biases)
            #
            with tf.variable_scope('layer_2'):
                kernel_shape = [1, 1, 400, 400]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(mdn_l1c, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                l2_b = conv
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train_mdn'])
            mdn_l2 = tf.nn.relu(conv + biases)

            with tf.variable_scope('locs'):

                with tf.variable_scope('layer_3'):
                    kernel_shape = [1, 1, 400, 400]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(mdn_l2, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train_mdn'])
                mdn_l3 = tf.nn.relu(conv + biases)

                weights_locs = tf.get_variable("weights_locs", [1, 1, 400, 2 * k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
                                              initializer=tf.constant_initializer(0))
                o_locs = tf.nn.conv2d(mdn_l3, weights_locs,
                                      [1, 1, 1, 1], padding='SAME') + biases_locs

                loc_shape = o_locs.get_shape().as_list()
                n_x = loc_shape[2]
                n_y = loc_shape[1]
                o_locs = tf.reshape(o_locs,[-1, n_y, n_x, k, n_out, 2])
                # when initialized o_locs will be centered around 0.
                # with adding grid_size/2, o_locs initially will be centered
                # in the center of the grid.
                o_locs += locs_offset/2

                # adding offset of each grid location.
                x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
                x_off = x_off * locs_offset
                y_off = y_off * locs_offset
                x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                x_locs = o_locs[:,:,:,:,:,0] + x_off
                y_locs = o_locs[:,:,:,:,:,1] + y_off
                o_locs = tf.stack([x_locs, y_locs], axis=5)
                o_locs = tf.reshape(o_locs,[-1, n_x*n_y*k,n_out,2])

            with tf.variable_scope('scales'):
                with tf.variable_scope('layer_3'):
                    kernel_shape = [1, 1, 400, 400]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(mdn_l2, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train_mdn'])
                mdn_l3 = tf.nn.relu(conv + biases)

                weights_scales = tf.get_variable("weights_scales", [1, 1, 400, k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_scales = tf.get_variable("biases_scales", k * self.conf.n_classes,
                                              initializer=tf.constant_initializer(0))
                o_scales = tf.exp(tf.nn.conv2d(mdn_l3, weights_scales,
                                      [1, 1, 1, 1], padding='SAME') + biases_scales)
                # when initialized o_scales will be centered around exp(0) and
                # mostly between [exp(-1), exp(1)] = [0.3 2.7]
                # so adding appropriate offsets to make it lie between the wanted range
                min_sig = self.conf.mdn_min_sigma
                max_sig = self.conf.mdn_max_sigma
                o_scales = (o_scales-1) * (max_sig-min_sig)/2
                o_scales = o_scales + (max_sig-min_sig)/2 + min_sig
                scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
                scales = tf.reshape(scales, [-1, n_x*n_y,k,n_out])
                scales = tf.reshape(scales,[-1, n_x*n_y*k,n_out])

            with tf.variable_scope('logits'):
                with tf.variable_scope('layer_3'):
                    kernel_shape = [1, 1, 400, 400]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(mdn_l2, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train_mdn'])
                mdn_l3 = tf.nn.relu(conv + biases)

                weights_logits = tf.get_variable("weights_logits", [1, 1, 400, k],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_logits = tf.get_variable("biases_logits", k ,
                                              initializer=tf.constant_initializer(0))
                logits = tf.nn.conv2d(mdn_l3, weights_logits,
                                      [1, 1, 1, 1], padding='SAME') + biases_logits
                logits = tf.reshape(logits, [-1, n_x * n_y, k])
                logits = tf.reshape(logits, [-1, n_x * n_y * k])


        all_locs = o_locs
        self.mdn_locs = all_locs
        self.mdn_scales = scales
        self.mdn_logits = logits
        self.hidden1_b = l1_b
        self.hidden2_b = l2_b
        self.hidden1 = mdn_l1a
        self.hidden2 = mdn_l2

    def create_network_vgg(self,l7_layer):

        k = 4
        conf = self.conf
        n_out = self.conf.n_classes
        locs_offset = 32

        # l7_layer = tf.stop_gradient(self.baseLayers['conv7'])
        # with tf.variable_scope('mdn_in'):
        #     l7_layer= CNB.conv_relu(l7_layer, [1, 1, conf.nfcfilt, 30],
        #                             0.005, 1, doBatchNorm=True,
        #                             trainPhase=self.ph['phase_train_mdn'])

        l7_shape = l7_layer.get_shape().as_list()
        in_dim = l7_shape[-1]

        with tf.variable_scope('mdn'):

            net = conv_relu(l7_layer,[3,3,in_dim,64],
                            self.ph['phase_train_mdn'],'conv1_1')
            net = conv_relu(net,[3,3,64,64],self.ph['phase_train_mdn'],'conv1_2')
            net = conv_relu(net,[5,5,64,64],self.ph['phase_train_mdn'],'conv1_3',stride=2)
            net = conv_relu(net,[3,3,64,128],self.ph['phase_train_mdn'],'conv2_1')
            net = conv_relu(net,[5,5,128,128],self.ph['phase_train_mdn'],'conv2_2',stride=2)
            net = conv_relu(net,[3,3,128,256],self.ph['phase_train_mdn'],'conv3_1')
            net = conv_relu(net,[3,3,256,256],self.ph['phase_train_mdn'],'conv3_2')
            net = conv_relu(net,[5,5,256,256],self.ph['phase_train_mdn'],'conv3_3', stride=2)
            net = conv_relu(net,[3,3,256,512],self.ph['phase_train_mdn'],'conv4_1')
            net = conv_relu(net,[3,3,512,512],self.ph['phase_train_mdn'],'conv4_2')
            net = conv_relu(net,[5,5,512,512],self.ph['phase_train_mdn'],'conv4_3', stride=2)
            net = conv_relu(net,[3,3,512,512],self.ph['phase_train_mdn'],'conv5_1')
            net = conv_relu(net,[3,3,512,512],self.ph['phase_train_mdn'],'conv5_2')
            net = conv_relu(net,[5,5,512,512],self.ph['phase_train_mdn'],'conv5_3', stride=2)
            net = conv_relu(net,[3,3,512,512],self.ph['phase_train_mdn'],'conv6_1')
            # net = fully_conn(net, 512,self.ph['phase_train_mdn'], sc_name='fc6')
            # net = fully_conn(net, 512,self.ph['phase_train_mdn'], sc_name='fc7')

            with tf.variable_scope('locs'):

                with tf.variable_scope('layer_1'):
                    kernel_shape = [1, 1, 512, 400]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(net, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train_mdn'])
                mdn_l3 = tf.nn.relu(conv + biases)

                weights_locs = tf.get_variable("weights_locs", [1, 1, 400, 2 * k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
                                              initializer=tf.constant_initializer(0))
                o_locs = tf.nn.conv2d(mdn_l3, weights_locs,
                                      [1, 1, 1, 1], padding='SAME') + biases_locs

                loc_shape = o_locs.get_shape().as_list()
                n_x = loc_shape[2]
                n_y = loc_shape[1]
                o_locs = tf.reshape(o_locs,[-1, n_y, n_x, k, n_out, 2])
                # when initialized o_locs will be centered around 0.
                # with adding grid_size/2, o_locs initially will be centered
                # in the center of the grid.
                o_locs += locs_offset/2

                # adding offset of each grid location.
                x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
                x_off = x_off * locs_offset
                y_off = y_off * locs_offset
                x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                x_locs = o_locs[:,:,:,:,:,0] + x_off
                y_locs = o_locs[:,:,:,:,:,1] + y_off
                o_locs = tf.stack([x_locs, y_locs], axis=5)
                o_locs = tf.reshape(o_locs,[-1, n_x*n_y*k,n_out,2])

            with tf.variable_scope('scales'):
                with tf.variable_scope('layer_1'):
                    kernel_shape = [1, 1, 512, 400]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(net, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train_mdn'])
                mdn_l3 = tf.nn.relu(conv + biases)

                weights_scales = tf.get_variable("weights_scales", [1, 1, 400, k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_scales = tf.get_variable("biases_scales", k * self.conf.n_classes,
                                              initializer=tf.constant_initializer(0))
                o_scales = tf.exp(tf.nn.conv2d(mdn_l3, weights_scales,
                                      [1, 1, 1, 1], padding='SAME') + biases_scales)
                # when initialized o_scales will be centered around exp(0) and
                # mostly between [exp(-1), exp(1)] = [0.3 2.7]
                # so adding appropriate offsets to make it lie between the wanted range
                min_sig = self.conf.mdn_min_sigma
                max_sig = self.conf.mdn_max_sigma
                o_scales = (o_scales-1) * (max_sig-min_sig)/2
                o_scales = o_scales + (max_sig-min_sig)/2 + min_sig
                scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
                scales = tf.reshape(scales, [-1, n_x*n_y,k,n_out])
                scales = tf.reshape(scales,[-1, n_x*n_y*k,n_out])

            with tf.variable_scope('logits'):
                with tf.variable_scope('layer_1'):
                    kernel_shape = [1, 1, 512, 400]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(net, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train_mdn'])
                mdn_l3 = tf.nn.relu(conv + biases)

                weights_logits = tf.get_variable("weights_logits", [1, 1, 400, k],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_logits = tf.get_variable("biases_logits", k ,
                                              initializer=tf.constant_initializer(0))
                logits = tf.nn.conv2d(mdn_l3, weights_logits,
                                      [1, 1, 1, 1], padding='SAME') + biases_logits
                logits = tf.reshape(logits, [-1, n_x * n_y, k])
                logits = tf.reshape(logits, [-1, n_x * n_y * k])


        all_locs = o_locs
        self.mdn_locs = all_locs
        self.mdn_scales = scales
        self.mdn_logits = logits
        self.hidden1 = net


    def my_loss(self):
        loss = 0
        for cls in range(self.conf.n_classes):
            ll = tf.nn.softmax(self.mdn_logits[...,cls], dim=1)

            cur_scales = self.mdn_scales[:,:,cls]
            pp = self.mdn_label[:,cls:cls+1,:]
            kk = tf.reduce_sum(tf.square(pp - self.mdn_locs[:,:,cls,:]), axis=2)
            cur_comp = tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2))

            # cur_comp = [i.prob(self.mdn_label[:,cls,:]) for i in self.components[cls]]
            # cur_comp = tf.stack(cur_comp)
            pp = (cur_comp * ll) + 1e-30
            cur_loss = tf.log(tf.reduce_sum(pp,axis=1))
            loss -= tf.reduce_sum(cur_loss)
        return loss

    def my_loss_joint(self):
        cur_comp = []
        ll = tf.nn.softmax(self.mdn_logits, dim=1)
# All gaussians in the mixture have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        ll = tf.cond(self.ph['phase_train_mdn'],lambda: ll+logit_eps,lambda: tf.identity(ll))
        ll = ll/tf.reduce_sum(ll,axis=1,keep_dims=True)
        for cls in range(self.conf.n_classes):

            cur_scales = self.mdn_scales[:,:,cls]
            pp = self.mdn_label[:,cls:cls+1,:]
            kk = tf.reduce_sum(tf.square(pp - self.mdn_locs[:,:,cls,:]), axis=2)
            cur_comp.append(tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2)))

        cur_comp = tf.stack(cur_comp,1)
        cur_loss = tf.reduce_prod(cur_comp,axis=1)
        pp = (cur_loss * ll) + 1e-30
        loss = -tf.log(tf.reduce_sum(pp,axis=1))
        return tf.reduce_sum(loss)


    def my_multi_normal(self, loc, scale):
        components = [ MultivariateNormalDiag(loc, scale_diag=scale),
                       MultivariateNormalDiag(loc, scale_diag=5*scale)]
        temp_tensor = tf.zeros([self.conf.batch_size, 2])  # not sure what this stupidity is
        probs = np.tile( [ 0.95, 0.05 ], [self.conf.batch_size, 1]).astype('float32')
        with tf.variable_scope('my_norm'):
            mm = Mixture( cat=Categorical(probs=probs),
                        components=components,
                        value=tf.zeros_like(temp_tensor) )
        return mm


    def create_ph(self):
        super(self.__class__, self).createPH()
        self.ph['step'] = tf.placeholder(tf.int64)
        self.ph['phase_train_mdn'] = tf.placeholder(tf.bool)
        self.ph['base_locs'] = tf.placeholder(tf.float32,
                                              [None, self.conf.n_classes, 2])
        # self.ph['y_mdn'] = tf.placeholder(tf.float32, [None, self.conf.n_classes, 2])

    def mdn_pred(self,sess):
        pred_weights, pred_means, pred_std = \
            sess.run([tf.nn.softmax(self.mdn_logits,dim=1),
                      self.mdn_locs, self.mdn_scales],
                     feed_dict=self.feed_dict)
        conf = self.conf
        osz = self.conf.imsz[0] // self.conf.rescale // self.conf.pool_scale
        mdn_pred_out = np.zeros([self.conf.batch_size, osz, osz, conf.n_classes])
        for sel in range(conf.batch_size):
            for cls in range(conf.n_classes):
                for ndx in range(pred_means.shape[1]):
                    if pred_weights[sel,ndx,cls] < 0.02:
                        continue
                    cur_locs = pred_means[sel:sel + 1, ndx:ndx + 1, cls, :].astype('int')
                    # cur_scale = pred_std[sel, ndx, cls, :].mean().astype('int')
                    cur_scale = pred_std[sel, ndx, cls].astype('int')
                    curl = (PoseTools.create_label_images(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
                    mdn_pred_out[sel,:, :, cls] += pred_weights[sel, ndx, cls] * curl[0, ..., 0]
        return  mdn_pred_out

    def mdn_pred_joint(self,sess):
        pred_weights, pred_means, pred_std = \
            sess.run([tf.nn.softmax(self.mdn_logits,dim=1),
                      self.mdn_locs, self.mdn_scales],
                     feed_dict=self.feed_dict)
        conf = self.conf
        osz = self.conf.imsz[0] // self.conf.rescale // self.conf.pool_scale
        mdn_pred_out = np.zeros([self.conf.batch_size, osz, osz, conf.n_classes])
        for sel in range(conf.batch_size):
            for cls in range(conf.n_classes):
                for ndx in range(pred_means.shape[1]):
                    if pred_weights[sel,ndx] < 0.02:
                        continue
                    cur_locs = pred_means[sel:sel + 1, ndx:ndx + 1, cls, :].astype('int')
                    # cur_scale = pred_std[sel, ndx, cls, :].mean().astype('int')
                    cur_scale = pred_std[sel, ndx, cls].astype('int')
                    curl = (PoseTools.create_label_images(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
                    mdn_pred_out[sel,:, :, cls] += pred_weights[sel, ndx] * curl[0, ..., 0]
        return  mdn_pred_out


    def train(self, restore, trainType):
        self.conf.trange = self.conf.imsz[0] // 25

        mdn_dropout = 1.
        self.create_ph()
        self.createFeedDict()
        self.feed_dict[self.ph['keep_prob']] = mdn_dropout
        self.feed_dict[self.ph['phase_train_base']] = False
        self.feed_dict[self.ph['phase_train_mdn']] = True
        self.feed_dict[self.ph['learning_rate']] = 0.
        self.trainType = trainType

        with tf.variable_scope('base'):
            super(self.__class__, self).createBaseNetwork(doBatch=True)

        # l7_layer = tf.stop_gradient(self.basePred)
        l7_layer = self.basePred
        self.create_network(l7_layer)
        self.openDBs()
        self.createBaseSaver()
        self.create_saver()


        y_label = self.ph['base_locs'] / self.conf.rescale / self.conf.pool_scale
        self.mdn_label = y_label
        # data_dict = {}
        # for ndx in range(self.conf.n_classes):
        #     data_dict[y[ndx]] = y_label[:,ndx,:]
        # inference = mymap.MAP(data=data_dict)
        # inference.initialize(var_list=PoseTools.getvars('mdn'))
        # self.loss = inference.loss
        self.loss = self.my_loss()

        starter_learning_rate = 0.0001
        decay_steps = 12000 * 8 / self.conf.batch_size
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate,self.ph['step'],decay_steps, 0.9)
        # decay_steps = 5000 / 8 * self.conf.batch_size
        # learning_rate = tf.train.exponential_decay(
        #     starter_learning_rate,self.ph['step'],decay_steps, 0.1,
        #     staircase=True)

        mdn_steps = 50000 * 8 / self.conf.batch_size

        # self.opt = tf.train.AdamOptimizer(
        #     learning_rate=learning_rate).minimize(self.loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.loss)

        with tf.Session() as sess:
            self.createCursors(sess)
            self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
            self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                             self.conf.n_classes,2])
            sess.run(tf.global_variables_initializer())
            self.restoreBase(sess, restore=True)
            self.restore(sess,restore)
            # self.initializeRemainingVars(sess)
            l7_shape = self.baseLayers['conv7'].get_shape().as_list()
            l7_shape[0] = self.conf.batch_size

            for step in range(self.mdn_start_at,mdn_steps):
                self.updateFeedDict(self.DBType.Train, sess=sess,
                                    distort=False)
                self.feed_dict[self.ph['keep_prob']] = mdn_dropout
                self.feed_dict[self.ph['step']] = step
                self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                                 self.conf.n_classes, 2])
                base_pred = sess.run(self.basePred, self.feed_dict)
                self.feed_dict[self.ph['phase_train_mdn']] = True
                self.feed_dict[self.ph['base_locs']] = \
                    PoseTools.get_base_pred_locs(base_pred, self.conf)
                # self.feed_dict[self.ph['layer7']] = np.zeros(l7_shape)
                # l7_cur = sess.run(self.baseLayers['conv7'], feed_dict=self.feed_dict)
                # self.feed_dict[self.ph['layer7']] = l7_cur
                sess.run(self.opt, self.feed_dict)

                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train, sess=sess,
                                        distort=False)
                    self.feed_dict[self.ph['phase_train_mdn']] = False
                    self.feed_dict[self.ph['keep_prob']] = mdn_dropout
                    self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                                                     self.conf.n_classes, 2])
                    base_pred = sess.run(self.basePred, self.feed_dict)
                    self.feed_dict[self.ph['base_locs']] = \
                        PoseTools.get_base_pred_locs(base_pred, self.conf)
                    # self.feed_dict[self.ph['layer7']] = np.zeros(l7_shape)
                    # l7_cur = sess.run(self.baseLayers['conv7'], feed_dict=self.feed_dict)
                    # self.feed_dict[self.ph['layer7']] = l7_cur
                    tr_loss = sess.run(self.loss, feed_dict=self.feed_dict)
                    self.mdn_train_data['train_err'].append(tr_loss)
                    self.mdn_train_data['step_no'].append(step)

                    mdn_pred = self.mdn_pred(sess)
                    bee = PoseTools.get_base_error(
                        self.feed_dict[self.ph['base_locs']],
                        mdn_pred, self.conf)
                    tt1 = np.sqrt(np.sum(np.square(bee), 2))
                    nantt1 = np.invert(np.isnan(tt1.flatten()))
                    train_dist = tt1.flatten()[nantt1].mean()
                    self.mdn_train_data['train_dist'].append(train_dist)

                    numrep = 1 #int(self.conf.numTest / self.conf.batch_size) + 1
                    val_loss = 0.
                    val_dist = 0.
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val, sess=sess,
                                            distort=False)
                        self.feed_dict[self.ph['keep_prob']] = 1.
                        self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                                                         self.conf.n_classes, 2])
                        base_pred = sess.run(self.basePred, self.feed_dict)
                        self.feed_dict[self.ph['base_locs']] = \
                            PoseTools.get_base_pred_locs(base_pred, self.conf)
                        # self.feed_dict[self.ph['layer7']] = np.zeros(l7_shape)
                        # l7_cur = sess.run(self.baseLayers['conv7'], feed_dict=self.feed_dict)
                        # self.feed_dict[self.ph['layer7']] = l7_cur
                        cur_te_loss = sess.run(self.loss, feed_dict=self.feed_dict)
                        val_loss += cur_te_loss
                        if rep == 0:
                            mdn_pred = self.mdn_pred(sess)
                            bee = PoseTools.get_base_error(
                                self.feed_dict[self.ph['base_locs']],
                                mdn_pred, self.conf)
                            tt1 = np.sqrt(np.sum(np.square(bee), 2))
                            nantt1 = np.invert(np.isnan(tt1.flatten()))
                            val_dist += tt1.flatten()[nantt1].mean()

                    val_loss /= numrep
                    # val_dist /= numrep
                    self.mdn_train_data['val_err'].append(val_loss)
                    self.mdn_train_data['val_dist'].append(val_dist)

                    print('{}:Train Loss:{:.4f},{:.2f}, Test Loss:{:.4f},{:.2f}'.format(
                        step, tr_loss, train_dist, val_loss, val_dist))

                if step % self.conf.save_step == 0:
                    self.save(sess, step)

            print("Optimization finished!")
            self.save(sess, mdn_steps)
        self.closeCursors()


    def train_joint(self, restore, trainType, full=False):
        # self.conf.trange = self.conf.imsz[0] // 25

        mdn_dropout = 1.0 if full else 0.9
        self.create_ph()
        self.createFeedDict()
        self.feed_dict[self.ph['keep_prob']] = mdn_dropout
        self.feed_dict[self.ph['phase_train_mdn']] = True
        self.feed_dict[self.ph['learning_rate']] = 0.
        self.trainType = trainType

        with tf.variable_scope('base'):
            super(self.__class__, self).createBaseNetwork(doBatch=True)

        if full:
            l7_layer = self.baseLayers['conv7']
            self.feed_dict[self.ph['phase_train_base']] = True
        else:
            l7_layer = tf.stop_gradient(self.baseLayers['conv7'])
            self.feed_dict[self.ph['phase_train_base']] = False
        # if full:
        #     l7_layer = self.basePred
        # else:
        #     l7_layer = tf.stop_gradient(self.basePred)
        self.create_network_joint(l7_layer)
        self.openDBs()
        self.createBaseSaver()
        self.create_saver()


        y_label = self.ph['locs'] / self.conf.rescale / self.conf.pool_scale
        self.mdn_label = y_label
        self.loss = self.my_loss_joint()

        starter_learning_rate = 0.0001
        decay_steps = 12000 * 8 / self.conf.batch_size
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate,self.ph['step'],decay_steps, 0.9)

        if full:
            mdn_steps = 50000 * 8 / self.conf.batch_size
        else:
            mdn_steps = 20000 * 8 / self.conf.batch_size

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.loss)

        with tf.Session() as sess:
            self.createCursors(sess)
            self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
            self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                             self.conf.n_classes,2])
            sess.run(tf.global_variables_initializer())
            self.restore_base_full(sess,full)
            self.restore(sess,restore)
            # self.initializeRemainingVars(sess)
            l7_shape = self.baseLayers['conv7'].get_shape().as_list()
            l7_shape[0] = self.conf.batch_size

            for step in range(self.mdn_start_at,mdn_steps):
                self.updateFeedDict(self.DBType.Train, sess=sess,
                                    distort=False)
                self.feed_dict[self.ph['keep_prob']] = mdn_dropout
                self.feed_dict[self.ph['step']] = step
                self.feed_dict[self.ph['phase_train_mdn']] = True
                sess.run(self.opt, self.feed_dict)

                if step % self.conf.display_step == 0:
                    self.feed_dict[self.ph['phase_train_mdn']] = False
                    self.updateFeedDict(self.DBType.Train, sess=sess,
                                        distort=False)
                    self.feed_dict[self.ph['keep_prob']] = mdn_dropout
                    tr_loss = sess.run(self.loss, feed_dict=self.feed_dict)
                    self.mdn_train_data['train_err'].append(tr_loss)
                    self.mdn_train_data['step_no'].append(step)

                    mdn_pred = self.mdn_pred_joint(sess)
                    bee = PoseTools.get_base_error(
                        self.feed_dict[self.ph['locs']],
                        mdn_pred, self.conf)
                    tt1 = np.sqrt(np.sum(np.square(bee), 2))
                    nantt1 = np.invert(np.isnan(tt1.flatten()))
                    train_dist = tt1.flatten()[nantt1].mean()
                    self.mdn_train_data['train_dist'].append(train_dist)

                    numrep = 1 #int(self.conf.numTest / self.conf.batch_size) + 1
                    val_loss = 0.
                    val_dist = 0.
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val, sess=sess,
                                            distort=False)
                        self.feed_dict[self.ph['keep_prob']] = 1.
                        cur_te_loss = sess.run(self.loss, feed_dict=self.feed_dict)
                        val_loss += cur_te_loss
                        if rep == 0:
                            mdn_pred = self.mdn_pred_joint(sess)
                            bee = PoseTools.get_base_error(
                                self.feed_dict[self.ph['locs']],
                                mdn_pred, self.conf)
                            tt1 = np.sqrt(np.sum(np.square(bee), 2))
                            nantt1 = np.invert(np.isnan(tt1.flatten()))
                            val_dist += tt1.flatten()[nantt1].mean()

                    val_loss /= numrep
                    self.mdn_train_data['val_err'].append(val_loss)
                    self.mdn_train_data['val_dist'].append(val_dist)

                    print('{}:Train Loss:{:.4f},{:.2f}, Test Loss:{:.4f},{:.2f}'.format(
                        step, tr_loss, train_dist, val_loss, val_dist))

                if step % self.conf.save_step == 0:
                    self.save(sess, step)
                    if full:
                        self.save_base_full(sess,step)

            print("Optimization finished!")
            self.save(sess, int(mdn_steps))
            if full:
                self.save_base_full(sess, int(mdn_steps))
        self.closeCursors()

    def create_offline_network(self):

        n_nets = 8
        k = 4
        conf = self.conf

        self.ph['base_pred'] = tf.placeholder(
            tf.float32, [None, 128, 128, self.conf.n_classes])

        l7_layer = self.ph['base_pred']
        l7_shape = l7_layer.get_shape().as_list()
        l7_layer_x_sz = l7_shape[2]
        l7_layer_y_sz = l7_shape[1]

        # self.ph['layer7'] = tf.placeholder(tf.float32, l7_shape)
        # l7_layer = self.ph['layer7']

        x_pad = (-l7_layer_x_sz) % n_nets
        y_pad = (-l7_layer_y_sz) % n_nets
        pad = [[0, 0], [0, y_pad], [0, x_pad], [0, 0]]
        l7_layer = tf.pad(l7_layer,pad)
        l7_shape = l7_layer.get_shape().as_list()
        l7_layer_x_sz = l7_shape[2]
        l7_layer_y_sz = l7_shape[1]

        all_locs = []
        all_scales = []
        all_logits = []

        x_psz = l7_layer_x_sz / n_nets
        y_psz = l7_layer_y_sz / n_nets
        with tf.variable_scope('mdn'):
            for xx in range(n_nets):
                for yy in range(n_nets):
                    x_start = int(xx * x_psz)
                    x_stop = int((xx + 1) * x_psz)
                    y_start = int(yy * y_psz)
                    y_stop = int((yy + 1) * y_psz)
                    cur_patch = l7_layer[:, y_start:y_stop, x_start:x_stop, :]
                    with tf.variable_scope('nn') as scope:
                        if xx or yy:
                            scope.reuse_variables()
                            locs, scales, logits, hidden1 = self.mdn_network(
                                cur_patch, k, conf.n_classes)
                        else:
                            locs, scales, logits, hidden1 = self.mdn_network(
                                cur_patch, k, conf.n_classes)

                    locs += np.array([x_start, y_start])
                    all_locs.append(locs)
                    all_scales.append(scales)
                    all_logits.append(logits)
            all_locs = tf.concat(all_locs, 1)
            all_scales = tf.concat(all_scales, 1)
            all_logits = tf.concat(all_logits, 1)
        self.mdn_locs = all_locs
        self.mdn_scales = all_scales
        self.mdn_logits = all_logits

        y_cls = []
        for cls in range(self.conf.n_classes):
            cat = Categorical(logits=all_logits[:, :, cls])
            components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                          in zip(tf.unstack(tf.transpose(all_locs[:, :, cls, :], [1, 0, 2])),
                                 tf.unstack(tf.transpose(all_scales[:, :, cls, :], [1, 0, 2])))]
            temp_tensor = tf.zeros([self.conf.batch_size, 2]) # not sure what this stupidity is
            y_cls.append(Mixture(cat=cat, components=components,
                                 value=tf.zeros_like(temp_tensor)))

        return y_cls


    def train_offline(self, restore, trainType):
        self.conf.trange = self.conf.imsz[0] // 25
        conf = self.conf

        mdn_dropout = 1.
        self.create_ph()
        self.createFeedDict()
        self.feed_dict[self.ph['keep_prob']] = mdn_dropout
        self.feed_dict[self.ph['phase_train_base']] = False
        self.feed_dict[self.ph['phase_train_mdn']] = True
        self.feed_dict[self.ph['learning_rate']] = 0.
        self.trainType = trainType

        self.create_network()
        # self.mdn_out = y
        self.openDBs()
        self.createBaseSaver()
        self.create_saver()


        y_label = self.ph['base_locs'] / self.conf.rescale / self.conf.pool_scale
        self.mdn_label = y_label
        self.loss = self.my_loss()
        # data_dict = {}
        # for ndx in range(self.conf.n_classes):
        #     data_dict[y[ndx]] = y_label[:,ndx,:]
        # inference = mymap.MAP(data=data_dict)
        # inference.initialize(var_list=PoseTools.getvars('mdn'))
        # self.loss = inference.loss

        starter_learning_rate = 0.0001
        decay_steps = 5000/8*self.conf.batch_size
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate,self.ph['step'],decay_steps, 0.9,
            staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.loss)

        with open(os.path.join(self.conf.cachedir, 'base_predictions'),
                  'rb') as f:
            train_data, test_data = pickle.load(f)

        m_train_data = train_data
        m_test_data = test_data

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        with tf.Session(config=config) as sess:
            self.createCursors(sess)
            self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
            self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                             self.conf.n_classes,2])
            sess.run(tf.global_variables_initializer())
            self.restoreBase(sess, restore=True)
            self.restore(sess,restore)
            # self.initializeRemainingVars(sess)
            l7_shape = self.baseLayers['conv7'].get_shape().as_list()
            l7_shape[0] = self.conf.batch_size

            mdn_steps = 50000*8/self.conf.batch_size
            test_step = 0
            self.updateFeedDict(self.DBType.Train, sess=sess,
                                distort=False)

            for step in range(self.mdn_start_at,mdn_steps):
                self.feed_dict[self.ph['step']] = step
                self.feed_dict[self.ph['phase_train_mdn']] = True

                data_ndx = step % len(m_train_data)
                cur_bpred = m_train_data[data_ndx][0]
                pd = 15
                cur_bpred = np.pad(cur_bpred, [[0, 0], [pd, pd], [pd, pd], [0, 0]],
                                   'constant', constant_values=-1)
                dxx = np.random.randint(pd * 2)
                dyy = np.random.randint(pd * 2)
                cur_bpred = cur_bpred[:, dyy:(128 + dyy), dxx:(128 + dxx), :]
                # self.feed_dict[self.ph['step']] = cur_step
                self.feed_dict[self.ph['base_locs']] = \
                    PoseTools.get_base_pred_locs(cur_bpred, self.conf)
                self.feed_dict[self.ph['base_pred']] = cur_bpred
                sess.run(self.opt, self.feed_dict)

                if step % self.conf.display_step == 0:
                    data_ndx = (step + 1) % len(m_train_data)
                    cur_bpred = m_train_data[data_ndx][0]
                    self.feed_dict[self.ph['base_locs']] = \
                        PoseTools.get_base_pred_locs(cur_bpred, self.conf)
                    self.feed_dict[self.ph['base_pred']] = cur_bpred
                    tr_loss = sess.run(self.loss, feed_dict=self.feed_dict)
                    self.mdn_train_data['train_err'].append(tr_loss)
                    self.mdn_train_data['step_no'].append(step)

                    mdn_pred = self.mdn_pred(sess)
                    bee = PoseTools.get_base_error(
                        self.feed_dict[self.ph['base_locs']], mdn_pred, self.conf)
                    tt1 = np.sqrt(np.sum(np.square(bee), 2))
                    nantt1 = np.invert(np.isnan(tt1.flatten()))
                    train_dist = tt1.flatten()[nantt1].mean()
                    self.mdn_train_data['train_dist'].append(train_dist)

                    data_ndx = (test_step + 1) % len(m_test_data)
                    cur_bpred = m_test_data[data_ndx][0]
                    self.feed_dict[self.ph['base_locs']] = \
                        PoseTools.get_base_pred_locs(cur_bpred, self.conf)
                    self.feed_dict[self.ph['base_pred']] = cur_bpred
                    cur_te_loss = sess.run(self.loss, feed_dict=self.feed_dict)
                    val_loss = cur_te_loss
                    mdn_pred = self.mdn_pred(sess)
                    bee = PoseTools.get_base_error(
                        self.feed_dict[self.ph['base_locs']], mdn_pred, conf)
                    tt1 = np.sqrt(np.sum(np.square(bee), 2))
                    nantt1 = np.invert(np.isnan(tt1.flatten()))
                    val_dist = tt1.flatten()[nantt1].mean()
                    test_step += 1

                    self.mdn_train_data['val_err'].append(val_loss)
                    self.mdn_train_data['val_dist'].append(val_dist)

                    print('{}:Train Loss:{:.4f},{:.2f}, Test Loss:{:.4f},{:.2f}'.format(
                        step, tr_loss, train_dist, val_loss, val_dist))

                if step % self.conf.save_step == 0:
                    self.save(sess, step)

            print("Optimization finished!")
            self.save(sess, mdn_steps)
        self.closeCursors()




def create_offline_data(conf):

    conf.batch_size = 16
    # conf.nfcfilt = 256
    self = PoseTrain.PoseTrain(conf)
    self.createPH()
    self.createFeedDict()
    self.feed_dict[self.ph['keep_prob']] = 1.
    self.feed_dict[self.ph['phase_train_base']] = False
    self.feed_dict[self.ph['learning_rate']] = 0.
    self.trainType = 0

    with tf.variable_scope('base'):
        self.createBaseNetwork(doBatch=True)

    self.conf.trange = self.conf.imsz[0] // 25

    self.openDBs()
    self.createBaseSaver()

    sess = tf.InteractiveSession()
    self.createCursors(sess)
    self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
    sess.run(tf.global_variables_initializer())
    self.restoreBase(sess, restore=True)

    #

    train_file = os.path.join(conf.cachedir,
                             conf.trainfilename) + '.tfrecords'
    num_train = 0
    for t in tf.python_io.tf_record_iterator(train_file):
        num_train += 1

    test_file = os.path.join(conf.cachedir,
                             conf.valfilename) + '.tfrecords'
    num_test = 0
    for t in tf.python_io.tf_record_iterator(test_file):
        num_test += 1

    train_data = []
    for ndx in range(num_train//conf.batch_size):
        self.updateFeedDict(self.DBType.Train,distort=True,
                            sess=sess)
        cur_pred = sess.run(self.basePred, self.feed_dict)
        train_data.append([cur_pred, self.locs, self.info, self.xs])

    test_data = []
    for ndx in range(num_test//conf.batch_size):
        self.updateFeedDict(self.DBType.Val,distort=False,
                            sess=sess)
        cur_pred = sess.run(self.basePred, self.feed_dict)
        test_data.append([cur_pred, self.locs, self.info, self.xs])

    with open(os.path.join(self.conf.cachedir, 'base_predictions'),
              'wb') as f:
        pickle.dump([train_data,test_data], f, protocol=2)

