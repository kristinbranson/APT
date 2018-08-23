import PoseCommon_dataset as PoseCommon
import PoseTools
import tensorflow as tf
import PoseUNet_dataset as PoseUNet
import os
import sys
import math
# from batch_norm import batch_norm_new as batch_norm
from tensorflow.contrib.layers import batch_norm
import convNetBase as CNB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile
from matplotlib import cm
import movies
import multiResData
from scipy import io as sio
import tensorflow.contrib.framework as tfr


renorm = False

def softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

def show_top_preds(im,pred_locs,pred_weights, n=12):

    ord = np.flipud(np.argsort(pred_weights))
    f,ax = plt.subplots(3,4)
    ax = ax.flatten()
    for ndx in range(n):
        if im.ndim == 2:
            ax[ndx].imshow(im, 'gray')
        else:
            ax[ndx].imshow(im)
        ax[ndx].scatter(pred_locs[ord[ndx],:,0],pred_locs[ord[ndx],:,1])

def train_preproc_func(ims, locs, info, conf):
    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, True, conf.rescale)
    hmaps = PoseTools.create_label_images(locs, conf.imsz, conf.rescale, conf.label_blur_rad)
    return ims.astype('float32'), locs.astype('float32'), info.astype('float32'), hmaps.astype('float32')


def val_preproc_func(ims, locs, info, conf):
    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, False, conf.rescale)
    hmaps = PoseTools.create_label_images(locs, conf.imsz, conf.rescale, conf.label_blur_rad)
    return ims.astype('float32'), locs.astype('float32'), info.astype('float32'), hmaps.astype('float32')


def conv_residual(x_in, n_filt, train_phase):
    in_dim = x_in.get_shape().as_list()[3]
    kernel_shape = [3, 3, in_dim, n_filt]
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, decay=0.99, is_training=train_phase)
    return conv


class PoseUMDN(PoseCommon.PoseCommon):

    def __init__(self, conf, name='pose_umdn',net_type='conv',
                 unet_name = 'pose_unet'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.dep_nets = [PoseUNet.PoseUNet(conf, unet_name)]
        self.net_type = net_type
        self.net_name = 'pose_umdn'
        self.train_data_name = 'traindata'
        self.i_locs = None
        self.input_dtypes = [tf.float32, tf.float32, tf.float32, tf.float32]

        def train_pp(ims,locs,info):
            return train_preproc_func(ims,locs,info, conf)
        def val_pp(ims,locs,info):
            return val_preproc_func(ims,locs,info, conf)

        self.train_py_map = lambda ims, locs, info: tuple(tf.py_func( train_pp, [ims, locs, info], self.input_dtypes))
        self.val_py_map = lambda ims, locs, info: tuple(tf.py_func( val_pp, [ims, locs, info], self.input_dtypes))

        if 'mdn_groups' not in self.conf.__dict__:
            self.conf.mdn_groups = [range(self.conf.n_classes)]


    def create_network(self):
        dep_net = self.dep_nets[0]
        dep_net.inputs = self.inputs
        dep_net.ph = self.ph
        X = dep_net.create_network()
        dep_net.pred = X
        # if not self.joint:
        #     X = tf.stop_gradient(X)

        with tf.variable_scope(self.net_name):
            if self.net_type is 'fixed':
                return self.create_network_fixed(X)
            else:
                return self.create_network1(X)


    def create_network1(self, X):

        n_conv = 2
        conv = PoseCommon.conv_relu3
        layers = []
        mdn_prev = None
        n_out = self.conf.n_classes
        k = 2
        extra_layers = self.conf.mdn_extra_layers
        # extra_layers = 1

        dep_net = self.dep_nets[0]
        n_layers_u = len(dep_net.up_layers) + extra_layers
        locs_offset = 1.
        # locs_offset = 2**n_layers_u
        n_groups = len(self.conf.mdn_groups)

        self.mdn_layers1 = []
        self.mdn_layers2 = []

        if 'mdn_cross_pred' in self.conf.__dict__.keys():
            do_cross_pred = self.conf.mdn_cross_pred
            print('Doing cross prediction for MDN')
        else:
            do_cross_pred = False
        # MDN downsample.
        for ndx in range(n_layers_u):

            if ndx < len(dep_net.up_layers):
                cur_ul = dep_net.up_layers[n_layers_u - extra_layers - ndx - 1]
                cur_dl = dep_net.down_layers[ndx]
                cur_l = tf.concat([cur_ul,cur_dl],axis=3)

                n_filt = cur_l.get_shape().as_list()[3]/2

                if mdn_prev is None:
                    X = cur_l
                else:
                    X = tf.concat([mdn_prev, cur_l], axis=3)
                    # X = mdn_prev + cur_l # residual

#            if ndx == 0:
#                n_conv = 1
#            elif ndx == 1:
#                n_conv = 2
#            elif ndx == 2:
#                n_conv = 4
#            else:
#                n_conv = 6

            for c_ndx in range(n_conv):
                sc_name = 'mdn_{}_{}'.format(ndx,c_ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'])
            self.mdn_layers1.append(X)

            # downsample using strides instead of max-pooling
            sc_name = 'mdn_{}_{}'.format(ndx, n_conv)
            with tf.variable_scope(sc_name):
                kernel_shape = [3, 3, n_filt, n_filt]
                weights = tf.get_variable("weights_{}".format(ndx), kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0.))
                cur_conv = tf.nn.conv2d(
                    X, weights,strides=[1, 2, 2, 1], padding='SAME')
                cur_conv = batch_norm(cur_conv, decay=0.99, is_training=self.ph['phase_train'],renorm=renorm)
                X = tf.nn.relu(cur_conv + biases)
            mdn_prev = X
            self.mdn_layers2.append(X)

            # with tf.variable_scope('mdn_{}'.format(ndx)):
            #     X = conv_residual(X, n_filt, self.ph['phase_train'])
            # with tf.variable_scope('mdn_{}_0'.format(ndx)):
            #     X_in = X
            #     X = conv_residual(X, n_filt, self.ph['phase_train'])
            #     X = tf.nn.leaky_relu(X)
            # with tf.variable_scope('mdn_{}_1'.format(ndx)):
            #     X = conv_residual(X, n_filt, self.ph['phase_train'])
            #     X = X + X_in
            #     X = tf.nn.leaky_relu(X)
            #
            # self.mdn_layers1.append(X)

        # few more convolution for the outputs
        n_filt = X.get_shape().as_list()[3]
        with tf.variable_scope('locs'):
            with tf.variable_scope('layer_locs'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv_l = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv_l = batch_norm(conv_l, decay=0.99,
                                  is_training=self.ph['phase_train'],renorm=renorm)
            mdn_l = tf.nn.relu(conv_l + biases)

            weights_locs = tf.get_variable("weights_locs", [1, 1, n_filt, 2 * k * n_out],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
                                          initializer=tf.constant_initializer(0))
            o_locs = tf.nn.conv2d(mdn_l, weights_locs,
                                  [1, 1, 1, 1], padding='SAME') + biases_locs

            loc_shape = o_locs.get_shape().as_list()
            n_x = loc_shape[2]
            n_y = loc_shape[1]
            o_locs = tf.reshape(o_locs,[-1, n_y, n_x, k, n_out, 2])
            # when initialized o_locs will be centered around 0 with var 1.
            # with multiplying grid_size/2, o_locs will have variance grid_size/2
            # with adding grid_size/2, o_locs initially will be centered
            # in the center of the grid.
            o_locs =  ((tf.sigmoid(o_locs)*2) - 0.5) * locs_offset
            self.i_locs = o_locs
            # o_locs *= float(locs_offset)/2
            # o_locs += float(locs_offset)/2

            self.pre_cross_locs = o_locs
            if do_cross_pred:
                cross_pred  = []
                cross_eps = 0.05
                cross_wts = []
                for pt in range(self.conf.n_classes):
                    kk1 = o_locs[:,:,:,:,:pt,:]
                    kk2 = o_locs[:,:,:,:,(pt+1):,:]
                    if pt == 0:
                        locs_in = kk2
                    elif pt== (self.conf.n_classes-1):
                        locs_in = kk1
                    else:
                        locs_in = tf.concat([kk1,kk2],axis=-2)
                    l_f = tf.reshape(locs_in,[-1,n_x,n_y,k,(n_out-1)*2])
                    l1 = tf.contrib.layers.fully_connected(l_f, 24)
                    l2 = tf.contrib.layers.fully_connected(l1, 18)
                    l3 = tf.contrib.layers.fully_connected(l2, 12)
                    l4 = tf.contrib.layers.fully_connected(l3, 8)
                    l_out = tf.contrib.layers.fully_connected(l4, 3, activation_fn=None)
                    wt = tf.sigmoid(l_out[:,:,:,:,0:1])
                    cross_wts.append(wt)
                    pp = l_out[:,:,:,:,1:]

                    rand = tf.random_uniform(shape=wt.get_shape().as_list()) < cross_eps
                    wt = tf.cond(self.ph['phase_train'], lambda: tf.maximum(tf.cast(rand,tf.float32),wt), lambda: tf.identity(wt))
                    #wt = tf.cond(self.ph['phase_train'], lambda: cross_eps + (1-cross_eps)*wt, lambda: tf.identity(wt))

                    wt_pred = (1-wt)*o_locs[:,:,:,:,pt,:] + wt*pp
                    cross_pred.append(wt_pred)

                o_locs = tf.stack(cross_pred,axis=-2)
                self.cross_wts = cross_wts

            # adding offset of each grid location.
            x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
            x_off = x_off * locs_offset
            y_off = y_off * locs_offset
            x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            x_locs = o_locs[:,:,:,:,:,0] + x_off
            y_locs = o_locs[:,:,:,:,:,1] + y_off
            o_locs = tf.stack([x_locs, y_locs], axis=5)
            locs = tf.reshape(o_locs,[-1, n_x*n_y*k,n_out,2],name='locs_final')



        with tf.variable_scope('scales'):
            with tf.variable_scope('layer_scales'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train'],renorm=renorm)
            mdn_l = tf.nn.relu(conv + biases)

            weights_scales = tf.get_variable("weights_scales", [1, 1, n_filt, k * n_out],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_scales = tf.get_variable("biases_scales", k * self.conf.n_classes,
                                          initializer=tf.constant_initializer(0))
            o_scales = tf.exp(tf.nn.conv2d(mdn_l, weights_scales,
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
            scales = tf.reshape(scales,[-1, n_x*n_y*k,n_out],name='scales_final')

        with tf.variable_scope('logits'):
            with tf.variable_scope('layer_logits'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train'],renorm=renorm)
            mdn_l = tf.nn.relu(conv + biases)

            weights_logits = tf.get_variable("weights_logits", [1, 1, n_filt, k * n_groups],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_logits = tf.get_variable("biases_logits", k * n_groups,
                                          initializer=tf.constant_initializer(0))
            logits = tf.nn.conv2d(mdn_l, weights_logits,
                                  [1, 1, 1, 1], padding='SAME') + biases_logits
            logits = tf.reshape(logits, [-1, n_x * n_y, k *n_groups])
            logits = tf.reshape(logits, [-1, n_x * n_y * k, n_groups],name='logits_final')

        return [locs,scales,logits]


    def create_network_fixed(self, X):
        # Downsample to a size between 4x4 and 2x2

        n_conv = 3
        conv = PoseCommon.conv_relu3
        layers = []
        mdn_prev = None
        n_out = self.conf.n_classes
        k = 2 # this is the number of final gaussians.
        k_fc = 10
        dep_net = self.dep_nets[0]
        n_layers_u = len(dep_net.up_layers)
        min_im_sz = min(self.conf.imsz)
        extra_layers = math.floor(np.log2(min_im_sz)) - n_layers_u - 1
        extra_layers = int(extra_layers)
        n_layers = n_layers_u + extra_layers
        locs_offset = 2**(n_layers)
        n_groups = len(self.conf.mdn_groups)

        # MDN downsample.
        for ndx in range(n_layers_u):
            cur_ul = dep_net.up_layers[n_layers_u - ndx - 1]
            cur_dl = dep_net.down_layers[ndx]
            cur_l = tf.concat([cur_ul,cur_dl],axis=3)

            n_filt = cur_dl.get_shape().as_list()[3]

            if mdn_prev is None:
                X = cur_l
            else:
                X = tf.concat([mdn_prev, cur_l], axis=3)

            for c_ndx in range(n_conv-1):
                sc_name = 'mdn_{}_{}'.format(ndx,c_ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'])

            # downsample using strides instead of max-pooling
            sc_name = 'mdn_{}_{}'.format(ndx, n_conv)
            with tf.variable_scope(sc_name):
                kernel_shape = [3, 3, n_filt, n_filt]
                weights = tf.get_variable("weights_{}".format(ndx), kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0.))
                cur_conv = tf.nn.conv2d(X, weights, strides=[1, 2, 2, 1], padding='SAME')
                cur_conv = batch_norm(cur_conv, decay=0.99, is_training=self.ph['phase_train'], renorm=renorm)
                X = tf.nn.relu(cur_conv + biases)
            mdn_prev = X

        # few more convolution for the outputs
        n_filt = X.get_shape().as_list()[3]
        for ndx in range(extra_layers):
            for c_ndx in range(n_conv-1):
                sc_name = 'mdn_extra_{}_{}'.format(ndx,c_ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'])

            with tf.variable_scope('mdn_extra_{}'.format(ndx)):
                kernel_shape = [3, 3, n_filt, n_filt]
                weights = tf.get_variable("weights_{}".format(ndx), kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0.))
                cur_conv = tf.nn.conv2d(X, weights, strides=[1, 2, 2, 1], padding='SAME')
                cur_conv = batch_norm(cur_conv, decay=0.99, is_training=self.ph['phase_train'], renorm=renorm)
                X = tf.nn.relu(cur_conv + biases)

        X_sz = min(X.get_shape().as_list()[1:3])
        assert (X_sz >= 2) and (X_sz<4), 'The net has been reduced too much or not too much'

        # fully connected layers
        # with tf.variable_scope('fc'):
        #     X = tf.contrib.layers.flatten(X)
        #     X = tf.contrib.layers.fully_connected(
        #         X, n_filt*4, normalizer_fn=batch_norm,normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train']})
        #     X = tf.contrib.layers.fully_connected(
        #         X, n_filt*4, normalizer_fn=batch_norm,normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train']})
        #     X = tf.contrib.layers.fully_connected(
        #         X, n_filt*4, normalizer_fn=batch_norm,normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train']})
        #
        X_conv = X
        with tf.variable_scope('locs'):
            with tf.variable_scope('fc'):
                X = tf.contrib.layers.flatten(X_conv)
                X = tf.contrib.layers.fully_connected(
                    X, n_filt * 4, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'], 'renorm':renorm})
                X = tf.contrib.layers.fully_connected(
                    X, n_filt * 4, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train','renorm':renorm]})

            with tf.variable_scope('layer_locs'):
                mdn_l = tf.contrib.layers.fully_connected(
                    X, n_filt * 2, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'],'renorm':renorm})

            locs = tf.contrib.layers.fully_connected(
                mdn_l, k_fc*n_out*2, activation_fn=None)
            # offset= np.mean(self.conf.imsz)
            # locs = locs * offset
            locs = locs + 0.5
            locs = tf.reshape(locs,[-1,k_fc,n_out,2])

                #     kernel_shape = [1, 1, n_filt, n_filt]
            #     weights = tf.get_variable("weights", kernel_shape,
            #                               initializer=tf.contrib.layers.xavier_initializer())
            #     biases = tf.get_variable("biases", kernel_shape[-1],
            #                              initializer=tf.constant_initializer(0))
            #     conv_l = tf.nn.conv2d(X, weights,
            #                         strides=[1, 1, 1, 1], padding='SAME')
            #     conv_l = batch_norm(conv_l, decay=0.99,
            #                       is_training=self.ph['phase_train'])
            # mdn_l = tf.nn.relu(conv_l + biases)
            #
            # weights_locs = tf.get_variable("weights_locs", [1, 1, n_filt, 2 * k * n_out],
            #                                initializer=tf.contrib.layers.xavier_initializer())
            # biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
            #                               initializer=tf.constant_initializer(0))
            # o_locs = tf.nn.conv2d(mdn_l, weights_locs,
            #                       [1, 1, 1, 1], padding='SAME') + biases_locs
            #
            # loc_shape = o_locs.get_shape().as_list()
            # n_x = loc_shape[2]
            # n_y = loc_shape[1]
            # o_locs = tf.reshape(o_locs,[-1, n_y, n_x, k, n_out, 2])
            # # when initialized o_locs will be centered around 0 with var 1.
            # # with multiplying grid_size/2, o_locs will have variance grid_size/2
            # # with adding grid_size/2, o_locs initially will be centered
            # # in the center of the grid.
            # o_locs *= locs_offset/2
            # o_locs += locs_offset/2
            #
            # # adding offset of each grid location.
            # x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
            # x_off = x_off * locs_offset
            # y_off = y_off * locs_offset
            # x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            # y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            # x_locs = o_locs[:,:,:,:,:,0] + x_off
            # y_locs = o_locs[:,:,:,:,:,1] + y_off
            # o_locs = tf.stack([x_locs, y_locs], axis=5)
            # locs = tf.reshape(o_locs,[-1, n_x*n_y*k,n_out,2])
            #
        with tf.variable_scope('scales'):
            with tf.variable_scope('fc'):
                X = tf.contrib.layers.flatten(X_conv)
                X = tf.contrib.layers.fully_connected(
                    X, n_filt * 4, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'],'renorm':renorm})
                X = tf.contrib.layers.fully_connected(
                    X, n_filt * 4, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'],'renorm':renorm})

            with tf.variable_scope('layer_scales'):
                mdn_s = tf.contrib.layers.fully_connected(
                    X, n_filt * 2, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'],'renorm':renorm})

            o_scales = tf.contrib.layers.fully_connected(
                mdn_s, k_fc * n_out, activation_fn = None)
            o_scales = tf.exp(o_scales)
            # when initialized o_scales will be centered around exp(0) and
            # mostly between [exp(-1), exp(1)] = [0.3 2.7]
            # so adding appropriate offsets to make it lie between the wanted range
            min_sig = self.conf.mdn_min_sigma
            max_sig = self.conf.mdn_max_sigma
            o_scales = (o_scales-1) * (max_sig-min_sig)/2
            o_scales = o_scales + (max_sig-min_sig)/2 + min_sig
            scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
            scales = tf.reshape(scales, [-1, k_fc, n_out])

            #     kernel_shape = [1, 1, n_filt, n_filt]
            #     weights = tf.get_variable("weights", kernel_shape,
            #                               initializer=tf.contrib.layers.xavier_initializer())
            #     biases = tf.get_variable("biases", kernel_shape[-1],
            #                              initializer=tf.constant_initializer(0))
            #     conv = tf.nn.conv2d(X, weights,
            #                         strides=[1, 1, 1, 1], padding='SAME')
            #     conv = batch_norm(conv, decay=0.99,
            #                       is_training=self.ph['phase_train'])
            # mdn_l = tf.nn.relu(conv + biases)
            #
            # weights_scales = tf.get_variable("weights_scales", [1, 1, n_filt, k * n_out],
            #                                initializer=tf.contrib.layers.xavier_initializer())
            # biases_scales = tf.get_variable("biases_scales", k * self.conf.n_classes,
            #                               initializer=tf.constant_initializer(0))
            # o_scales = tf.exp(tf.nn.conv2d(mdn_l, weights_scales,
            #                       [1, 1, 1, 1], padding='SAME') + biases_scales)
            # # when initialized o_scales will be centered around exp(0) and
            # # mostly between [exp(-1), exp(1)] = [0.3 2.7]
            # # so adding appropriate offsets to make it lie between the wanted range
            # min_sig = self.conf.mdn_min_sigma
            # max_sig = self.conf.mdn_max_sigma
            # o_scales = (o_scales-1) * (max_sig-min_sig)/2
            # o_scales = o_scales + (max_sig-min_sig)/2 + min_sig
            # scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
            # scales = tf.reshape(scales, [-1, n_x*n_y,k,n_out])
            # scales = tf.reshape(scales,[-1, n_x*n_y*k,n_out])

        with tf.variable_scope('logits'):
            with tf.variable_scope('fc'):
                X = tf.contrib.layers.flatten(X_conv)
                X = tf.contrib.layers.fully_connected(
                    X, n_filt * 4, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'],'renorm':renorm})
                X = tf.contrib.layers.fully_connected(
                    X, n_filt * 4, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'],'renorm':renorm})

            with tf.variable_scope('layer_logits'):
                mdn_w = tf.contrib.layers.fully_connected(
                    X, n_filt * 2, normalizer_fn=batch_norm,
                    normalizer_params={'decay': 0.99, 'is_training': self.ph['phase_train'],'renorm':renorm})

            logits = tf.contrib.layers.fully_connected(
                mdn_w, k_fc*n_groups, activation_fn=None)
            logits = tf.reshape(logits,[-1,k_fc,n_groups])
            #     kernel_shape = [1, 1, n_filt, n_filt]
            #     weights = tf.get_variable("weights", kernel_shape,
            #                               initializer=tf.contrib.layers.xavier_initializer())
            #     biases = tf.get_variable("biases", kernel_shape[-1],
            #                              initializer=tf.constant_initializer(0))
            #     conv = tf.nn.conv2d(X, weights,
            #                         strides=[1, 1, 1, 1], padding='SAME')
            #     conv = batch_norm(conv, decay=0.99,
            #                       is_training=self.ph['phase_train'])
            # mdn_l = tf.nn.relu(conv + biases)
            #
            # weights_logits = tf.get_variable("weights_logits", [1, 1, n_filt, k * n_groups],
            #                                initializer=tf.contrib.layers.xavier_initializer())
            # biases_logits = tf.get_variable("biases_logits",
            #         k * n_groups ,initializer=tf.constant_initializer(0))
            # logits = tf.nn.conv2d(mdn_l, weights_logits,
            #                       [1, 1, 1, 1], padding='SAME') + biases_logits
            # logits = tf.reshape(logits, [-1, n_x * n_y, k * n_groups])
            # logits = tf.reshape(logits, [-1, n_x * n_y * k, n_groups])

        return [locs,scales,logits]


    def restore_pretrained(self, sess, model_file):
        rem_locs = tf.global_variables(self.net_name + '/locs')
        rem_locs += tf.global_variables(self.net_name + '/scales')
        rem_locs += tf.global_variables(self.net_name + '/logits')
        rem_locs += tf.global_variables(self.dep_nets[0].net_name + '/out')

        var_list = self.get_var_list()
        pre_list = tfr.list_variables(model_file)
        pre_list_names = [p[0] for p in pre_list]
        pre_list_shapes = [p[1] for p in pre_list]
        common_vars = []
        for i in var_list:
            if not i.name[:-2] in pre_list_names:
                continue
            ndx = pre_list_names.index(i.name[:-2])
            if pre_list_shapes[ndx] == i.shape.as_list():
                common_vars.append(i)

        c_names = [c.name for c in common_vars]
        r_names = [v.name for v in var_list if v not in common_vars]
        print("-- Loading from pretrained --")
        print('\n'.join(c_names))
        print("-- Not Loading from pretrained --")
        print('\n'.join(r_names))
        # common_vars = [i for i in common_vars if i not in rem_locs]
        pretrained_saver = tf.train.Saver(var_list=common_vars)
        pretrained_saver.restore(sess, model_file)

    def my_loss(self, X, y):

        dep_net = self.dep_nets[0]
        if self.net_type is 'conv':
            extra_layers = self.conf.mdn_extra_layers
            n_layers_u = len(dep_net.up_layers) + extra_layers
            locs_offset = float(2**n_layers_u)
        elif self.net_type is 'fixed':
            locs_offset = np.mean(self.conf.imsz)/2
        else:
            raise Exception('Unknown net type')

        mdn_locs, mdn_scales, mdn_logits = X
        cur_comp = []
        ll = tf.nn.softmax(mdn_logits, axis=1)

        n_preds = mdn_locs.get_shape().as_list()[1]
        # All gaussians in the mixture have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        ll = tf.cond(self.ph['phase_train'], lambda: ll + logit_eps, lambda: tf.identity(ll))
        ll = ll / tf.reduce_sum(ll, axis=1, keepdims=True)
        for cls in range(self.conf.n_classes):
            cur_scales = mdn_scales[:, :, cls]
            pp = y[:, cls:cls + 1, :]/locs_offset
            kk = tf.sqrt(tf.reduce_sum(tf.square(pp - mdn_locs[:, :, cls, :]), axis=2))
            # tf.div is actual correct implementation of gaussian distance.
            # but we run into numerical issues. Since the scales are withing
            # the same range, I'm just ignoring them for now.
            # dd = tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2))
            dd = tf.exp(-kk / cur_scales / 2)
            cur_comp.append(dd)

        for ndx,gr in enumerate(self.conf.mdn_groups):
            sel_comp = [cur_comp[i] for i in gr]
            sel_comp = tf.stack(sel_comp, 1)
            pp = ll[:,:, ndx] * \
                       tf.reduce_prod(sel_comp, axis=1) + 1e-30
            if ndx is 0:
                cur_loss = -tf.log(tf.reduce_sum(pp,axis=1))
            else:
                cur_loss = cur_loss - tf.log(tf.reduce_sum(pp,axis=1))

        # product because we are looking at joint distribution of all the points.
        # pp = cur_loss + 1e-30
        # loss = -tf.log(tf.reduce_sum(pp, axis=1))
        return tf.reduce_sum(cur_loss)


    def l2_loss(self, X, y):

        dep_net = self.dep_nets[0]
        if self.net_type is 'conv':
            extra_layers = self.conf.mdn_extra_layers
            n_layers_u = len(dep_net.up_layers) + extra_layers
            locs_offset = float(2**n_layers_u)
        elif self.net_type is 'fixed':
            locs_offset = np.mean(self.conf.imsz)/2
        else:
            raise Exception('Unknown net type')

        mdn_locs, mdn_scales, mdn_logits = X
        cur_comp = []
        ll = tf.nn.softmax(mdn_logits, axis=1)

        n_preds = mdn_locs.get_shape().as_list()[1]
        # All gaussians in the mixture have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        ll = tf.cond(self.ph['phase_train'], lambda: ll + logit_eps, lambda: tf.identity(ll))
        ll = ll / tf.reduce_sum(ll, axis=1, keepdims=True)
        for cls in range(self.conf.n_classes):
            pp = y[:, cls:cls + 1, :]/locs_offset
            kk = tf.sqrt(tf.reduce_sum(tf.square(pp - mdn_locs[:, :, cls, :]), axis=2))
            # tf.div is actual correct implementation of gaussian distance.
            # but we run into numerical issues. Since the scales are withing
            # the same range, I'm just ignoring them for now.
            # dd = tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2))
            cur_comp.append(kk)

        cur_loss = 0
        for ndx,gr in enumerate(self.conf.mdn_groups):
            sel_comp = [cur_comp[i] for i in gr]
            sel_comp = tf.stack(sel_comp, 1)
            pp = ll[:,:, ndx] * tf.reduce_sum(sel_comp, axis=1)
            cur_loss += pp

        return tf.reduce_sum(cur_loss)

    def compute_dist(self, preds, locs):

        dep_net = self.dep_nets[0]
        if self.net_type is 'conv':
            extra_layers = self.conf.mdn_extra_layers
            n_layers_u = len(dep_net.up_layers) + extra_layers
            locs_offset = float(2**n_layers_u)
        elif self.net_type is 'fixed':
            locs_offset = np.mean(self.conf.imsz)/2
        else:
            raise Exception('Unknown net type')

        locs = locs.copy()
        if locs.ndim == 3:
            locs = locs[:,np.newaxis,:,:]
        val_means, val_std, val_wts = preds
        val_means = val_means * locs_offset
        val_dist = np.zeros(locs.shape[:-1])
        pred_dist = np.zeros(val_means.shape[:-1])
        pred_dist[:] = np.nan
        for ndx in range(val_means.shape[0]):
            for gdx, gr in enumerate(self.conf.mdn_groups):
                for g in gr:
                    if locs.shape[1] == 1:
                        sel_ex = (np.argmax(val_wts[ndx,:,gdx]),)
                    else:
                        sel_ex = val_wts[ndx, :, gdx] > 0
                    mm = val_means[ndx, ...][np.newaxis, sel_ex, :, :]
                    ll = locs[ndx, ...][:, np.newaxis, :, :]
                    jj =  mm[:,:,g,:] - ll[:,:,g,:]
                    # jj has distance between all labels and
                    # all predictions with wts > 0.
                    dd1 = np.sqrt(np.sum(jj ** 2, axis=-1)).min(axis=1)
                    # instead of min it should ideally be matching.
                    # but for now this is ok.
                    dd2 = np.sqrt(np.sum(jj ** 2, axis=-1)).min(axis=0)
                    # dd1 -- to every label is there a close by prediction.
                    # dd2 -- to every prediction is there a close by labels.
                    # or dd1 is fn and dd2 is fp.
                    val_dist[ndx,:, g] = dd1 * self.conf.rescale
                    pred_dist[ndx,sel_ex, g] = dd2 * self.conf.rescale
        val_dist[locs[..., 0] < -5000] = np.nan
        pred_mean = np.nanmean(pred_dist)
        label_mean = np.nanmean(val_dist)
        return (pred_mean+label_mean)/2


    def compute_train_data(self, sess, db_type):
        self.fd_train() if db_type is self.DBType.Train \
            else self.fd_val()
        cur_loss, cur_inputs , pred_means, pred_std, pred_weights = sess.run(
            [self.cost,self.inputs] + self.pred, self.fd)

        pred_weights = softmax(pred_weights, axis=1)
        cur_dist = self.compute_dist( [pred_means, pred_std, pred_weights], cur_inputs[1])
        return cur_loss, cur_dist


    def train_umdn(self):

        self.joint = True
        def loss(inputs, pred):
            # mdn_loss = self.l2_loss(pred, inputs[1])
            mdn_loss = self.my_loss(pred, inputs[1])
            unet_pred = self.dep_nets[0].pred
            unet_loss = tf.nn.l2_loss(inputs[-1]-unet_pred)
            unet_pred_shape = np.array(unet_pred.get_shape().as_list()[1:3]).astype('float32')
            unet_loss_factor = 10
            unet_loss_normalized = unet_loss_factor * unet_loss / unet_pred_shape[0]/ unet_pred_shape[1]
            # unet_loss / unet_pred_shape[0]/ unet_pred_shape[1] is generally around 0.02 while mdn_loss is around 0.4. unet_loss_factor of 10 gives unet_loss half the weight.
            mdn_loss += unet_loss_normalized

            return mdn_loss

        super(self.__class__, self).train(
            create_network=self.create_network,
            loss=loss,
            learning_rate=0.0001)


    def restore_net(self, model_file=None):
        return self.restore_net_common(self.create_network, model_file)


    def restore_net_meta(self):
        sess = PoseCommon.restore_meta_common(self)

        graph = tf.get_default_graph()
        try:
            kp = graph.get_tensor_by_name('keep_prob:0')
        except KeyError:
            kp = graph.get_tensor_by_name('Placeholder:0')

        self.dep_nets.ph['keep_prob'] = kp
        self.dep_nets.ph['x'] = graph.get_tensor_by_name('x:0')
        self.dep_nets.ph['y'] = graph.get_tensor_by_name('y:0')
        self.dep_nets.ph['learning_rate'] = graph.get_tensor_by_name('learning_r_1:0')
        self.dep_nets.ph['phase_train'] = graph.get_tensor_by_name('phase_train_1:0')

        self.ph['x'] = self.dep_nets.ph['x']
        self.ph['learning_rate'] = graph.get_tensor_by_name('learning_r:0')
        self.ph['phase_train'] = graph.get_tensor_by_name('phase_train:0')
        self.ph['locs'] = graph.get_tensor_by_name('locs:0')

        try:
            unet_pred = graph.get_tensor_by_name('pose_unet/unet_pred:0')
        except KeyError:
            unet_pred = graph.get_tensor_by_name('pose_unet/add:0')
        self.dep_nets.pred = unet_pred

        locs = graph.get_tensor_by_name('pose_umdn/locs/locs_final:0')
        logits = graph.get_tensor_by_name('pose_umdn/logits/logits_final:0')
        scales = graph.get_tensor_by_name('pose_umdn/scales/scales_final:0')

        self.pred = [locs,scales,logits]
        self.create_fd()
        return sess


    def get_pred_fn(self, model_file=None):
        sess, latest_model_file = self.restore_net(model_file)

        conf = self.conf

        if self.net_type is 'conv':
            extra_layers = self.conf.mdn_extra_layers
            n_layers_u = len(self.dep_nets[0].up_layers) + extra_layers
            locs_offset = float(2 ** n_layers_u)
        elif self.net_type is 'fixed':
            locs_offset = np.mean(self.conf.imsz) / 2
        else:
            raise Exception('Unknown net type')

        def pred_fn(all_f):
            # this is the function that is used for classification.
            # this should take in an array B x H x W x C of images, and
            # output an array of predicted locations.
            # predicted locations should be B x N x 2
            # PoseTools.get_pred_locs can be used to convert heatmaps into locations.

            bsize = conf.batch_size
            xs, _ = PoseTools.preprocess_ims(
                all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf,
                distort=False, scale=self.conf.unet_rescale)

            self.fd[self.inputs[0]] = xs
            self.fd[self.ph['phase_train']] = False
            self.fd[self.ph['learning_rate']] = 0
            # self.fd[self.ph['keep_prob']] = 1.
            pred, cur_input = sess.run([self.pred, self.inputs], self.fd)

            pred_means, pred_std, pred_weights = pred
            pred_means *= locs_offset
            pred_weights = softmax(pred_weights,axis=1)

            osz = [int(i/conf.unet_rescale) for i in self.conf.imsz]
            mdn_pred_out = np.zeros([bsize, osz[0], osz[1], conf.n_classes])

            for sel in range(bsize):
                for cls in range(conf.n_classes):
                    for ndx in range(pred_means.shape[1]):
                        cur_gr = [l.count(cls) for l in self.conf.mdn_groups].index(1)
                        if pred_weights[sel, ndx, cur_gr] < (0.02 / self.conf.max_n_animals):
                            continue
                        cur_locs = np.round(pred_means[sel:sel + 1, ndx:ndx + 1, cls, :]).astype('int')
                        # cur_scale = pred_std[sel, ndx, cls, :].mean().astype('int')
                        cur_scale = pred_std[sel, ndx, cls].astype('int')
                        curl = (PoseTools.create_label_images(cur_locs, osz, 1, cur_scale) + 1) / 2
                        mdn_pred_out[sel, :, :, cls] += pred_weights[sel, ndx, cur_gr] * curl[0, ..., 0]

            base_locs = PoseTools.get_pred_locs(mdn_pred_out)
            # base_locs = np.zeros([pred_means.shape[0],self.conf.n_classes,2])
            # for ndx in range(pred_means.shape[0]):
            #     for gdx, gr in enumerate(self.conf.mdn_groups):
            #         for g in gr:
            #             sel_ex = np.argmax(pred_weights[ndx, :, gdx])
            #             mm = pred_means[ndx, sel_ex, g, :]
            #             base_locs[ndx, g] = mm
            #
            # base_locs = base_locs * conf.unet_rescale

            return base_locs, mdn_pred_out

        def close_fn():
            sess.close()

        return pred_fn, close_fn, latest_model_file

    def classify_val(self, model_file=None, onTrain = False):
        if not onTrain:
            val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        else:
            val_file = os.path.join(self.conf.cachedir, self.conf.trainfilename + '.tfrecords')

        num_val = 0
        for _ in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        print('--- Loading the model by reconstructing the graph ---')
        self.setup_train()
        self.pred = self.create_network()
        self.create_saver()
        sess = tf.Session()
        self.restore(sess, model_file)
        PoseCommon.initialize_remaining_vars(sess)

        try:
            self.restore_td()
        except AttributeError:  # If the conf file has been modified
            print("Couldn't load train data because the conf has changed!")
            self.init_td()

        p_m, p_s, p_w = self.pred
        conf = self.conf
        osz = self.conf.imsz
        #       self.joint = True

        if self.net_type is 'conv':
            extra_layers = self.conf.mdn_extra_layers
            n_layers_u = len(self.dep_nets[0].up_layers) + extra_layers
            locs_offset = float(2 ** n_layers_u)
        elif self.net_type is 'fixed':
            locs_offset = np.mean(self.conf.imsz) / 2
        else:
            raise Exception('Unknown net type')
        p_m *= locs_offset

        val_dist = []
        val_ims = []
        val_preds = []
        val_predlocs = []
        val_locs = []
        val_means = []
        val_std = []
        val_wts = []
        val_u_preds = []
        val_u_predlocs =[]
        for step in range(num_val/self.conf.batch_size):
            if onTrain:
                self.fd_train()
            else:
                self.fd_val()
            pred_means, pred_std, pred_weights, cur_input, u_pred = sess.run(
                [p_m, p_s, p_w, self.inputs, self.dep_nets[0].pred], self.fd)
            val_means.append(pred_means)
            val_std.append(pred_std)
            val_wts.append(pred_weights)
            pred_weights = softmax(pred_weights,axis=1)
            mdn_pred_out = np.zeros([self.conf.batch_size, osz[0], osz[1], conf.n_classes])

            locs = cur_input[1]
            cur_dist = np.zeros([conf.batch_size,conf.n_classes])
            cur_predlocs = np.zeros(pred_means.shape[0:1] + pred_means.shape[2:])
            for ndx in range(pred_means.shape[0]):
                for gdx, gr in enumerate(self.conf.mdn_groups):
                    for g in gr:
                        sel_ex = np.argmax(pred_weights[ndx,:,gdx])
                        mm = pred_means[ndx, sel_ex, g, :]
                        ll = locs[ndx,g,:]
                        jj =  mm-ll
                        # jj has distance between all labels and
                        # all predictions with wts > 0.
                        dd1 = np.sqrt(np.sum(jj ** 2, axis=-1))
                        cur_dist[ndx, g] = dd1 * self.conf.rescale
                        cur_predlocs[ndx,g,...] = mm

            val_u_preds.append(u_pred)
            u_predlocs = PoseTools.get_pred_locs(u_pred)
            val_u_predlocs.append(u_predlocs)

            # for sel in range(conf.batch_size):
            #     for cls in range(conf.n_classes):
                    # for ndx in range(pred_means.shape[1]):
                    #     cur_gr = [l.count(cls) for l in self.conf.mdn_groups].index(1)
                    #     if pred_weights[sel, ndx, cur_gr] < (0.02/self.conf.max_n_animals):
                    #         continue
                    #     cur_locs = np.round(pred_means[sel:sel + 1, ndx:ndx + 1, cls, :]).astype('int')
                    #     # cur_scale = pred_std[sel, ndx, cls, :].mean().astype('int')
                    #     cur_scale = pred_std[sel, ndx, cls].astype('int')
                    #     curl = (PoseTools.create_label_images(cur_locs, osz, 1, cur_scale) + 1) / 2
                    #     mdn_pred_out[sel, :, :, cls] += pred_weights[sel, ndx, cur_gr] * curl[0, ..., 0]

            # locs = cur_input[1]
            # if locs.ndim == 3:
            #     cur_predlocs = PoseTools.get_pred_locs(mdn_pred_out)
            #     cur_dist = np.sqrt(np.sum(
            #         (cur_predlocs - locs/self.conf.unet_rescale) ** 2, 2))
            # else:
            #     cur_predlocs = PoseTools.get_pred_locs_multi(
            #         mdn_pred_out,self.conf.max_n_animals,
            #         self.conf.label_blur_rad * 7)
            #     curl = locs.copy()/self.conf.unet_rescale
            #     jj = cur_predlocs[:,:,np.newaxis,:,:] - curl[:,np.newaxis,...]
            #     cur_dist = np.sqrt(np.sum(jj**2,axis=-1)).min(axis=1)
            val_dist.append(cur_dist)
            val_ims.append(cur_input[0])
            val_locs.append(cur_input[1])
            val_preds.append(mdn_pred_out)
            val_predlocs.append(cur_predlocs)

        sess.close()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape( (-1,) + in_a.shape[2:])
        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_predlocs = val_reshape(val_predlocs)
        val_locs = val_reshape(val_locs)
        val_means = val_reshape(val_means)
        val_std = val_reshape(val_std)
        val_wts = val_reshape(val_wts)
        val_u_predlocs = val_reshape(val_u_predlocs)
        val_u_preds = val_reshape(val_u_preds)
        tf.reset_default_graph()

        return val_dist, val_ims, val_preds, val_predlocs, val_locs,\
               [val_means,val_std,val_wts], [val_u_predlocs, val_u_preds]

    def classify_movie(self, movie_name, sess, max_frames=-1, start_at=0,flipud=False):
        # maxframes if specificied reads that many frames
        # start at specifies where to start reading.
        conf = self.conf

        cap = movies.Movie(movie_name)
        n_frames = int(cap.get_n_frames())

        # figure out how many frames to read
        if max_frames > 0:
            if max_frames + start_at > n_frames:
                n_frames = n_frames - start_at
            else:
                n_frames = max_frames
        else:
            n_frames = n_frames - start_at

        if self.net_type is 'conv':
            extra_layers = self.conf.mdn_extra_layers
            n_layers_u = len(self.dep_nets.up_layers) + extra_layers
            locs_offset = float(2**n_layers_u)
        elif self.net_type is 'fixed':
            locs_offset = np.mean(self.conf.imsz)/2
        else:
            raise Exception('Unknown net type')

        # pre allocate results
        bsize = conf.batch_size
        n_batches = int(math.ceil(float(n_frames)/ bsize))
        pred_locs = np.zeros([n_frames, conf.n_classes, 2])
        all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))

        for curl in range(n_batches):
            ndx_start = curl * bsize
            ndx_end = min(n_frames, (curl + 1) * bsize)
            ppe = min(ndx_end - ndx_start, bsize)
            for ii in range(ppe):
                fnum = ndx_start + ii + start_at
                frame_in = cap.get_frame(fnum)
                if len(frame_in) == 2:
                    frame_in = frame_in[0]
                    if frame_in.ndim == 2:
                        frame_in = frame_in[:,:,np.newaxis]
                frame_in = PoseTools.crop_images(frame_in, conf)
                if flipud:
                    frame_in = np.flipud(frame_in)
                all_f[ii, ...] = frame_in[..., 0:conf.imgDim]

            # all_f = all_f.astype('uint8')
            # xs = PoseTools.adjust_contrast(all_f, conf)
            # xs = PoseTools.scale_images(xs, conf.unet_rescale, conf)
            # xs = PoseTools.normalize_mean(xs, self.conf)
            xs, _ = PoseTools.preprocess_ims(all_f, np.zeros([bsize, self.conf.n_classes,2]),conf=conf,distort=False,scale=self.conf.unet_rescale)
            self.fd[self.ph['x']] = xs
            self.fd[self.ph['phase_train']] = False
            val_means, val_std, val_wts = sess.run(self.pred, self.fd)

            base_locs = np.zeros([self.conf.batch_size, self.conf.n_classes,2])
            for ndx in range(val_means.shape[0]):
                for gdx, gr in enumerate(self.conf.mdn_groups):
                    for g in gr:
                        sel_ex = np.argmax(val_wts[ndx, :, gdx])
                        base_locs[ndx,g,:] = val_means[ndx, sel_ex, g, :]

            pred_locs[ndx_start:ndx_end, :, :] = base_locs[:ppe,...]*locs_offset * conf.unet_rescale
            sys.stdout.write('.')
            if curl % 20 == 19:
                sys.stdout.write('\n')

        cap.close()
        return pred_locs

    def classify_movie_trx(self, movie_name, trx, sess, max_frames=-1, start_at=0,flipud=False, return_ims=False):
        # maxframes if specificied reads that many frames
        # start at specifies where to start reading.

        conf = self.conf
        cap = movies.Movie(movie_name, interactive=False)
        n_frames = int(cap.get_n_frames())
        T = sio.loadmat(trx)['trx'][0]
        n_trx = len(T)

        end_frames = np.array([x['endframe'][0,0] for x in T])
        first_frames = np.array([x['firstframe'][0,0] for x in T]) - 1 # for converting from 1 indexing to 0 indexing
        if max_frames < 0:
            max_frames = end_frames.max()
        if start_at > max_frames:
            return None
        max_n_frames = max_frames - start_at
        pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
        pred_locs[:] = np.nan

        if return_ims:
            ims = np.zeros([max_n_frames, n_trx, conf.imsz[0]/conf.unet_rescale, conf.imsz[1]/conf.unet_rescale,conf.imgDim])

        bsize = conf.batch_size

        for trx_ndx in range(n_trx):
            cur_trx = T[trx_ndx]
            # pre allocate results
            if first_frames[trx_ndx] > start_at:
                cur_start = first_frames[trx_ndx]
            else:
                cur_start = start_at

            if end_frames[trx_ndx] < max_frames:
                cur_end = end_frames[trx_ndx]
            else:
                cur_end = max_frames

            n_frames = cur_end - cur_start
            n_batches = int(math.ceil(float(n_frames)/ bsize))
            all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))

            for curl in range(n_batches):
                ndx_start = curl * bsize + cur_start - start_at
                ndx_end = min(n_frames, (curl + 1) * bsize) + cur_start - start_at
                ppe = min(ndx_end - ndx_start, bsize)
                for ii in range(ppe):
                    fnum = ndx_start + ii + cur_start
                    frame_in = cap.get_frame(fnum)
                    if len(frame_in) == 2:
                        frame_in = frame_in[0]
                        if frame_in.ndim == 2:
                            frame_in = frame_in[:,:,np.newaxis]
                    if flipud:
                        frame_in = np.flipud(frame_in)

                    trx_fnum = fnum - first_frames[trx_ndx]
                    x = int(round(cur_trx['x'][0,trx_fnum]))-1
                    y = int(round(cur_trx['y'][0,trx_fnum]))-1
                    # -1 for 1-indexing in matlab and 0-indexing in python
                    theta = cur_trx['theta'][0,trx_fnum]
                    assert conf.imsz[0] == conf.imsz[1]

                    frame_in, _ = multiResData.get_patch_trx(frame_in,x,y,theta,conf.imsz[0], np.zeros([2,2]))
                    frame_in = frame_in[:, :, 0:conf.imgDim]
                    all_f[ii, ...] = frame_in[..., 0:conf.imgDim]

                all_f = all_f.astype('uint8')
                xs = PoseTools.adjust_contrast(all_f, conf)
                xs = PoseTools.scale_images(xs, conf.unet_rescale, conf)
                xs = PoseTools.normalize_mean(xs, self.conf)
                self.fd[self.ph['x']] = xs
                self.fd[self.ph['phase_train']] = False
                val_means, val_std, val_wts = sess.run(self.pred, self.fd)

                if return_ims:
                    ims[ndx_start:ndx_end, trx_ndx,:,:,:] = xs[:ppe,...]
                base_locs = np.zeros([self.conf.batch_size, self.conf.n_classes,2])
                for ndx in range(val_means.shape[0]):
                    for gdx, gr in enumerate(self.conf.mdn_groups):
                        for g in gr:
                            sel_ex = np.argmax(val_wts[ndx, :, gdx])
                            base_locs[ndx,g,:] = val_means[ndx, sel_ex, g, :]

                pred_locs[ndx_start:ndx_end, trx_ndx, :, :] = base_locs[:ppe,...]
                sys.stdout.write('.')
                if curl % 20 == 19:
                    sys.stdout.write('\n')

            sys.stdout.write('\n')
        cap.close()
        tf.reset_default_graph()
        if return_ims:
            return pred_locs, ims
        else:
            return pred_locs

    def create_pred_movie(self, movie_name, out_movie, max_frames=-1,flipud=False,trace=True):
        conf = self.conf
        sess = self.setup_net(0, True)
        predLocs = self.classify_movie(movie_name,sess,max_frames=max_frames,flipud=flipud)
        tdir = tempfile.mkdtemp()

        cap = movies.Movie(movie_name)
        nframes = int(cap.get_n_frames())
        if max_frames > 0:
            nframes = max_frames

        fig = mpl.figure.Figure(figsize=(9, 4))
        canvas = FigureCanvasAgg(fig)
        sc = self.conf.unet_rescale

        color = cm.hsv(np.linspace(0, 1 - 1./conf.n_classes, conf.n_classes))
        trace_len = 30
        for curl in range(nframes):
            frame_in = cap.get_frame(curl)
            if len(frame_in) == 2:
                frame_in = frame_in[0]
                if frame_in.ndim == 2:
                    frame_in = frame_in[:,:, np.newaxis]
            frame_in = PoseTools.crop_images(frame_in, conf)

            if flipud:
                frame_in = np.flipud(frame_in)
            fig.clf()
            ax1 = fig.add_subplot(1, 1, 1)
            if frame_in.shape[2] == 1:
                ax1.imshow(frame_in[:,:,0], cmap=cm.gray)
            else:
                ax1.imshow(frame_in)
            ax1.scatter(predLocs[curl, :, 0]*sc,
                        predLocs[curl, :, 1]*sc,
                c=color*0.9, linewidths=0,
                edgecolors='face',marker='+',s=45)
            if trace:
                for ndx in range(conf.n_classes):
                    curc = color[ndx,:].copy()
                    curc[3] = 0.5
                    e = np.maximum(0,curl-trace_len)
                    ax1.plot(predLocs[e:curl,ndx,0]*sc,
                             predLocs[e:curl, ndx, 1] * sc,
                             c = curc,lw=0.8)
            ax1.axis('off')
            fname = "test_{:06d}.png".format(curl)

            # to printout without X.
            # From: http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html
            # The size * the dpi gives the final image size
            #   a4"x4" image * 80 dpi ==> 320x320 pixel image
            canvas.print_figure(os.path.join(tdir, fname), dpi=160)

            # below is the easy way.
        #         plt.savefig(os.path.join(tdir,fname))

        tfilestr = os.path.join(tdir, 'test_*.png')
        mencoder_cmd = "mencoder mf://" + tfilestr + " -frames " + "{:d}".format(
            nframes) + " -mf type=png:fps=15 -o " + out_movie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
        os.system(mencoder_cmd)
        cap.close()
        tf.reset_default_graph()

    def create_pred_movie_trx(self, movie_name, out_movie, trx, fly_num, max_frames=-1, start_at=0, flipud=False,trace=True):
        conf = self.conf
        sess = self.setup_net(0, True)
        predLocs = self.classify_movie_trx(movie_name, trx, sess, max_frames=max_frames,flipud=flipud, start_at=start_at)
        tdir = tempfile.mkdtemp()

        cap = movies.Movie(movie_name,interactive=False)
        T = sio.loadmat(trx)['trx'][0]
        n_trx = len(T)

        end_frames = np.array([x['endframe'][0,0] for x in T])
        first_frames = np.array([x['firstframe'][0,0] for x in T]) - 1
        if max_frames < 0:
            max_frames = end_frames.max()

        nframes = max_frames - start_at
        fig = mpl.figure.Figure(figsize=(8, 8))
        canvas = FigureCanvasAgg(fig)
        sc = self.conf.unet_rescale

        color = cm.hsv(np.linspace(0, 1 - 1./conf.n_classes, conf.n_classes))
        trace_len = 3
        cur_trx = T[fly_num]
        c_x = None
        c_y = None
        for curl in range(nframes):
            fnum = curl + start_at
            frame_in = cap.get_frame(curl+start_at)
            if len(frame_in) == 2:
                frame_in = frame_in[0]
                if frame_in.ndim == 2:
                    frame_in = frame_in[:,:, np.newaxis]

            trx_fnum = fnum - first_frames[fly_num]
            x = int(round(cur_trx['x'][0, trx_fnum])) - 1
            y = int(round(cur_trx['y'][0, trx_fnum])) - 1
            theta = -cur_trx['theta'][0, trx_fnum]
            R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            # -1 for 1-indexing in matlab and 0-indexing in python
            if c_x is None:
                c_x = x; c_y = y;

            if (np.abs(c_x - x) > conf.imsz[0]*3./8.*2.) or (np.abs(c_y - y) > conf.imsz[0]*3./8.*2.):
                c_x = x; c_y = y


            assert conf.imsz[0] == conf.imsz[1]

            frame_in, _ = multiResData.get_patch_trx(frame_in, c_x, c_y, -math.pi/2, conf.imsz[0]*2, np.zeros([2, 2]))
            frame_in = frame_in[:, :, 0:conf.imgDim]

            if flipud:
                frame_in = np.flipud(frame_in)
            fig.clf()
            ax1 = fig.add_subplot(1, 1, 1)
            if frame_in.shape[2] == 1:
                ax1.imshow(frame_in[:,:,0], cmap=cm.gray)
            else:
                ax1.imshow(frame_in)
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()

            hsz_p = conf.imsz[0]/2 # half size for pred
            hsz_s = conf.imsz[0] # half size for showing
            for fndx in range(n_trx):
                ct = T[fndx]
                if (fnum < first_frames[fndx]) or (fnum>=end_frames[fndx]):
                    continue
                trx_fnum = fnum - first_frames[fndx]
                x = int(round(ct['x'][0, trx_fnum])) - 1
                y = int(round(ct['y'][0, trx_fnum])) - 1
                theta = -ct['theta'][0, trx_fnum] - math.pi/2
                R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

                curlocs = np.dot(predLocs[curl,fndx,:,:]-[hsz_p,hsz_p],R)
                ax1.scatter(curlocs[ :, 0]*sc - c_x + x +hsz_s,
                            curlocs[ :, 1]*sc - c_y + y + hsz_s,
                    c=color*0.9, linewidths=0,
                    edgecolors='face',marker='+',s=30)
                if trace:
                    for ndx in range(conf.n_classes):
                        curc = color[ndx,:].copy()
                        curc[3] = 0.5
                        e = np.maximum(0,curl-trace_len)
                        zz = np.dot(predLocs[e:(curl+1),fndx,ndx,:]-[hsz_p,hsz_p],R)
                        ax1.plot(zz[:,0]*sc - c_x + x + hsz_s,
                                 zz[:,1]*sc - c_y + y + hsz_s,
                                 c = curc,lw=0.8,alpha=0.6)
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.axis('off')
            fname = "test_{:06d}.png".format(curl)

            # to printout without X.
            # From: http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html
            # The size * the dpi gives the final image size
            #   a4"x4" image * 80 dpi ==> 320x320 pixel image
            canvas.print_figure(os.path.join(tdir, fname), dpi=300)

            # below is the easy way.
        #         plt.savefig(os.path.join(tdir,fname))

        tfilestr = os.path.join(tdir, 'test_*.png')
        mencoder_cmd = "mencoder mf://" + tfilestr + " -frames " + "{:d}".format(
            nframes) + " -mf type=png:fps=15 -o " + out_movie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
        os.system(mencoder_cmd)
        cap.close()
        tf.reset_default_graph()

    def worst_preds(self,  dist= 10, num_ex = 30, train_type=0, at_step=-1, onTrain=False,):

        val_dist, val_ims, val_preds, val_predlocs, val_locs, val_out = self.classify_val(train_type, at_step, onTrain=False)

        sel = np.where(np.any(val_dist>dist,axis=1))[0]
        yy = np.ceil(np.sqrt(num_ex/ 12) * 4).astype('int')
        xx = np.ceil(num_ex / yy).astype('int')
        f,ax = plt.subplots(xx,yy,sharex=True,sharey=True)
        ax = ax.flatten()
        for ndx in range(num_ex):
            x = np.random.choice(len(sel))
            ix = sel[x]
            dd = np.where(val_dist[ix,:]>dist)[0]
            pt = np.random.choice(dd)
            grp = [i.count(pt) for i in self.conf.mdn_groups ].index(1)
            ax[ndx].imshow(val_ims[ix,:,:,0],'gray')
            ax[ndx].scatter(
                val_out[0][ix,:,pt,0], val_out[0][ix,:,pt,1],
                s=np.maximum( 0.5, (val_out[2][ix,:,grp])*5))
            ax[ndx].scatter( val_locs[ix,pt,0],val_locs[ix,pt,1],marker='+')
            ax[ndx].set_title('{}:{}'.format(x,pt))

class PoseUMDNMulti(PoseUMDN, PoseCommon.PoseCommonMulti):

    def __init__(self, conf, name='pose_umdn_multi',net_type='conv',
                 unet_name = 'pose_unet_multi'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.dep_nets = PoseUNet.PoseUNetMulti(conf, unet_name)
        self.net_type = net_type

    def create_cursors(self, sess):
        PoseCommon.PoseCommonMulti.create_cursors(self, sess)

    def create_ph_fd(self):
        PoseCommon.PoseCommon.create_ph_fd(self)
        self.dep_nets.create_ph_fd()
        self.ph['x'] = self.dep_nets.ph['x']
        self.ph['locs'] = tf.placeholder(tf.float32,
                           [None, self.conf.max_n_animals, self.conf.n_classes, 2],
                           name='locs')

    def loss_multi(self, X, y):
        mdn_locs, mdn_scales, mdn_logits = X
        cur_comp = []

        # softmax of logits
        ll = tf.nn.softmax(mdn_logits, dim=1)
        # All gaussians in the mixture have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        ll = tf.cond(self.ph['phase_train'], lambda: ll + logit_eps, lambda: tf.identity(ll))
        ll = ll / tf.reduce_sum(ll, axis=1, keepdims=True)

        # find distance of each predicted gaussian to all the labels.
        # soft assign each prediction to the labels.
        # compute loss based on this soft assignment.
        # also the total weight that gets assigned to each label should
        # be roughly equal. Add loss for that too.

        labels_valid = tf.cast(y[:, :, 0, 0] > -10000.,tf.float32)
        n_labels = tf.reduce_sum(labels_valid,axis=1)
        # labels that don't exist have high negative values
        # distance of predicted gaussians from such labels should be high
        # and they shouldn't create problems with soft_assign.
        for cls in range(self.conf.n_classes):
            cur_scales = mdn_scales[:, :, cls:cls+1]
            pp = tf.expand_dims(y[:, :, cls, :],axis=1)
            kk = tf.reduce_sum(tf.square(pp - mdn_locs[:, :, cls:cls+1, :]), axis=3)
            # kk now has the distance between all the labels and all the predictions.
            dd = tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2))
            cur_comp.append(dd)

        cur_comp = tf.stack(cur_comp, 1)
        tot_dist = tf.reduce_prod(cur_comp, axis=1)
        # soft assignment to each label.
        soft_assign = tf.expand_dims(ll,axis=2) * tf.nn.softmax(tot_dist,dim=2)
        ind_loss = tot_dist * soft_assign

        # how much weight does each label have.
        # all the valid labels should get equal weights
        # without this, the net might predict only label.
        lbl_weights = tf.reduce_sum(soft_assign,axis=1)
        lbl_weights_loss = (1-lbl_weights*tf.expand_dims(n_labels,axis=1))**2

        lbl_weights_loss = lbl_weights_loss*labels_valid

        # for each label, the predictions are treated as mixture.
        int_loss = -tf.log(tf.reduce_sum(ind_loss, axis=1)+ 1e-30)
        loss = int_loss * lbl_weights_loss
        return tf.reduce_sum(loss)

    def train_umdn(self, restore, train_type=0,joint=False,
                   net_type='conv'):

        self.joint = joint
        self.net_type = net_type
        if joint:
            training_iters = 40000/self.conf.batch_size*8
        else:
            training_iters = 20000/self.conf.batch_size*8

        super(self.__class__, self).train(
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=training_iters,
            loss=self.loss_multi,
            pred_in_key='locs',
            learning_rate=0.0001,
            td_fields=('loss','dist'))


class PoseUMDNTime(PoseUMDN):

    def __init__(self,conf,name = 'pose_umdn_time'):
        PoseUMDN.__init__(self,conf,name)

