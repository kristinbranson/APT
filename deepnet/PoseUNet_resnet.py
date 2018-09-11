import PoseUNet_dataset as PoseUNet
import PoseUMDN_dataset as PoseUMDN
import PoseCommon_dataset as PoseCommon
import convNetBase as CNB
import PoseTools
import sys
import os
import tensorflow as tf
import imageio
import localSetup
from scipy.ndimage.interpolation import zoom
import numpy as np
import cv2
import traceback
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim
from PoseCommon_dataset import conv_relu3
from tensorflow.contrib.layers import batch_norm

class PoseUNet_resnet(PoseUNet.PoseUNet):

    def __init__(self, conf, name='unet_resnet'):
        conf.use_pretrained_weights = True
#        conf.pretrained_weights = '/home/mayank/work/deepcut/pose-tensorflow/models/pretrained/resnet_v1_50.ckpt'
        self.conf = conf
        PoseUNet.PoseUNet.__init__(self, conf, name=name)

    def create_network(self):

        im, locs, info, hmap = self.inputs
        conf = self.conf
        im.set_shape([conf.batch_size, conf.imsz[0]/conf.rescale,conf.imsz[1]/conf.rescale, conf.imgDim])
        hmap.set_shape([conf.batch_size, conf.imsz[0]/conf.rescale, conf.imsz[1]/conf.rescale,conf.n_classes])
        locs.set_shape([conf.batch_size, conf.n_classes,2])
        info.set_shape([conf.batch_size,3])
        if conf.imgDim == 1:
            im = tf.tile(im,[1,1,1,3])

        conv = lambda a, b: conv_relu3(
            a,b,self.ph['phase_train'], keep_prob=None,
            use_leaky=self.conf.unet_use_leaky)

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(im,
                                      global_pool=False, is_training=self.ph['phase_train'])

        with tf.variable_scope(self.net_name):

            l_names = ['conv1', 'block1/unit_2/bottleneck_v1','block2/unit_3/bottleneck_v1', 'block3/unit_5/bottleneck_v1', 'block4']
            down_layers = [end_points['resnet_v1_50/'+x] for x in l_names]
            n_filts = [32, 64, 64, 128, 256, 512]

            ex_down_layers =  conv(self.inputs[0], 64)
            down_layers.insert(0,ex_down_layers)

            prev_in = None
            for ndx in reversed(range(len(down_layers))):

                if prev_in is None:
                    X = down_layers[ndx]
                else:
                    X = tf.concat([prev_in, down_layers[ndx]],axis=-1)

                sc_name = 'layerup_{}_0'.format(ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filts[ndx])

                if ndx is not 0:
                    sc_name = 'layerup_{}_1'.format(ndx)
                    with tf.variable_scope(sc_name):
                        X = conv(X, n_filts[ndx])

                    layers_sz = down_layers[ndx-1].get_shape().as_list()[1:3]
                    with tf.variable_scope('u_{}'.format(ndx)):
                        # X = CNB.upscale('u_{}'.format(ndx), X, layers_sz)
                        X_sh = X.get_shape().as_list()
                        w_mat = np.zeros([4,4,X_sh[-1],X_sh[-1]])
                        for wndx in range(X_sh[-1]):
                            w_mat[:,:,wndx,wndx] = 1.
                        w = tf.get_variable('w', [4, 4, X_sh[-1], X_sh[-1]],initializer=tf.constant_initializer(w_mat))
                        out_shape = [X_sh[0],layers_sz[0],layers_sz[1],X_sh[-1]]
                        X = tf.nn.conv2d_transpose(X, w, output_shape=out_shape, strides=[1, 2, 2, 1], padding="SAME")
                        biases = tf.get_variable('biases', [out_shape[-1]], initializer=tf.constant_initializer(0))
                        conv_b = X + biases

                        bn = batch_norm(conv_b)
                        X = tf.nn.relu(bn)

                prev_in = X

            n_filt = X.get_shape().as_list()[-1]
            n_out = self.conf.n_classes
            weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("out_biases", n_out,
                                     initializer=tf.constant_initializer(0.))
            conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.add(conv, biases, name = 'unet_pred')
            X = 2*tf.sigmoid(X)-1
        # X = conv+biases
        return X

    def get_var_list(self):
        var_list = tf.global_variables(self.net_name)
        for dep_net in self.dep_nets:
            var_list += dep_net.get_var_list()
        var_list += tf.global_variables('resnet_v1_50')
        return var_list


class PoseUMDN_resnet(PoseUMDN.PoseUMDN):

    def __init__(self, conf, name='umdn_resnet'):
        conf.use_pretrained_weights = True
#        conf.pretrained_weights = '/home/mayank/work/deepcut/pose-tensorflow/models/pretrained/resnet_v1_50.ckpt'
        self.conf = conf
        PoseUMDN.PoseUMDN.__init__(self, conf, name=name)
        self.dep_nets = []

    def create_network(self):

        im, locs, info, hmap = self.inputs
        conf = self.conf
        im.set_shape([conf.batch_size, conf.imsz[0]/conf.rescale,conf.imsz[1]/conf.rescale, conf.imgDim])
        hmap.set_shape([conf.batch_size, conf.imsz[0]/conf.rescale, conf.imsz[1]/conf.rescale,conf.n_classes])
        locs.set_shape([conf.batch_size, conf.n_classes,2])
        info.set_shape([conf.batch_size,3])
        if conf.imgDim == 1:
            im = tf.tile(im,[1,1,1,3])

        conv = lambda a, b: conv_relu3(
            a,b,self.ph['phase_train'], keep_prob=None,
            use_leaky=self.conf.unet_use_leaky)

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(im,global_pool=False, is_training=self.ph['phase_train'])

        X = net
        k = 2
        locs_offset = 1.
        n_groups = len(self.conf.mdn_groups)
        n_out = self.conf.n_classes

        with tf.variable_scope(self.net_name + '_unet'):

            l_names = ['conv1', 'block1/unit_2/bottleneck_v1','block2/unit_3/bottleneck_v1', 'block3/unit_5/bottleneck_v1', 'block4']
            down_layers = [end_points['resnet_v1_50/'+x] for x in l_names]
            n_filts = [32, 64, 64, 128, 256, 512]

            ex_down_layers =  conv(self.inputs[0], 64)
            down_layers.insert(0,ex_down_layers)

            prev_in = None
            for ndx in reversed(range(len(down_layers))):

                if prev_in is None:
                    X = down_layers[ndx]
                else:
                    X = tf.concat([prev_in, down_layers[ndx]],axis=-1)

                sc_name = 'layerup_{}_0'.format(ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filts[ndx])

                if ndx is not 0:
                    sc_name = 'layerup_{}_1'.format(ndx)
                    with tf.variable_scope(sc_name):
                        X = conv(X, n_filts[ndx])

                    layers_sz = down_layers[ndx-1].get_shape().as_list()[1:3]
                    with tf.variable_scope('u_{}'.format(ndx)):
                        # X = CNB.upscale('u_{}'.format(ndx), X, layers_sz)
                        X_sh = X.get_shape().as_list()
                        w_mat = np.zeros([4,4,X_sh[-1],X_sh[-1]])
                        for wndx in range(X_sh[-1]):
                            w_mat[:,:,wndx,wndx] = 1.
                        w = tf.get_variable('w', [4, 4, X_sh[-1], X_sh[-1]],initializer=tf.constant_initializer(w_mat))
                        out_shape = [X_sh[0],layers_sz[0],layers_sz[1],X_sh[-1]]
                        X = tf.nn.conv2d_transpose(X, w, output_shape=out_shape, strides=[1, 2, 2, 1], padding="SAME")
                        biases = tf.get_variable('biases', [out_shape[-1]], initializer=tf.constant_initializer(0))
                        conv_b = X + biases

                        bn = batch_norm(conv_b)
                        X = tf.nn.relu(bn)

                prev_in = X

            n_filt = X.get_shape().as_list()[-1]
            n_out = self.conf.n_classes
            weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("out_biases", n_out,
                                     initializer=tf.constant_initializer(0.))
            conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.add(conv, biases, name = 'unet_pred')
            X_unet = 2*tf.sigmoid(X)-1

        X = net
        with tf.variable_scope(self.net_name):

            n_filt_in = X.get_shape().as_list()[3]
            n_filt = 256
            with tf.variable_scope('locs'):
                with tf.variable_scope('layer_locs'):
                    kernel_shape = [1, 1, n_filt_in, n_filt-2]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv_l = tf.nn.conv2d(X, weights,
                                          strides=[1, 1, 1, 1], padding='SAME')
                    conv_l = batch_norm(conv_l, decay=0.99,
                                        is_training=self.ph['phase_train'])
                mdn_l = tf.nn.relu(conv_l + biases)

                loc_shape = mdn_l.get_shape().as_list()
                x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
                x_off = np.tile(x_off[np.newaxis,:,:,np.newaxis],[loc_shape[0],1,1,1])
                y_off = np.tile(y_off[np.newaxis,:,:,np.newaxis], [loc_shape[0],1,1,1])
                mdn_l = tf.concat([mdn_l,x_off,y_off],axis=-1)

                weights_locs = tf.get_variable("weights_locs", [1, 1, n_filt, 2 * k * n_out],
                                               initializer=tf.contrib.layers.xavier_initializer())
                biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
                                              initializer=tf.constant_initializer(0))
                o_locs = tf.nn.conv2d(mdn_l, weights_locs,
                                      [1, 1, 1, 1], padding='SAME') + biases_locs

                n_x = loc_shape[2]
                n_y = loc_shape[1]
                locs = tf.reshape(o_locs, [-1, n_x * n_y * k, n_out, 2], name='locs_final')

                # when initialized o_locs will be centered around 0 with var 1.
                # with multiplying grid_size/2, o_locs will have variance grid_size/2
                # with adding grid_size/2, o_locs initially will be centered
                # in the center of the grid.

                # o_locs = ((tf.sigmoid(o_locs) * 2) - 0.5) * locs_offset
                # self.i_locs = o_locs
                # # o_locs *= float(locs_offset)/2
                # # o_locs += float(locs_offset)/2
                #
                # # adding offset of each grid location.
                # x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
                # x_off = x_off * locs_offset
                # y_off = y_off * locs_offset
                # x_off = x_off[np.newaxis, :, :, np.newaxis, np.newaxis]
                # y_off = y_off[np.newaxis, :, :, np.newaxis, np.newaxis]
                # x_locs = o_locs[:, :, :, :, :, 0] + x_off
                # y_locs = o_locs[:, :, :, :, :, 1] + y_off
                # o_locs = tf.stack([x_locs, y_locs], axis=5)
                # locs = tf.reshape(o_locs, [-1, n_x * n_y * k, n_out, 2], name='locs_final')

            with tf.variable_scope('scales'):
                with tf.variable_scope('layer_scales'):
                    kernel_shape = [1, 1, n_filt_in, n_filt]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(X, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train'])
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
                o_scales = (o_scales - 1) * (max_sig - min_sig) / 2
                o_scales = o_scales + (max_sig - min_sig) / 2 + min_sig
                scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
                scales = tf.reshape(scales, [-1, n_x * n_y, k, n_out])
                scales = tf.reshape(scales, [-1, n_x * n_y * k, n_out], name='scales_final')

            with tf.variable_scope('logits'):
                with tf.variable_scope('layer_logits'):
                    kernel_shape = [1, 1, n_filt_in, n_filt]
                    weights = tf.get_variable("weights", kernel_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable("biases", kernel_shape[-1],
                                             initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(X, weights,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99,
                                      is_training=self.ph['phase_train'])
                mdn_l = tf.nn.relu(conv + biases)

                weights_logits = tf.get_variable("weights_logits", [1, 1, n_filt, k * n_groups],
                                                 initializer=tf.contrib.layers.xavier_initializer())
                biases_logits = tf.get_variable("biases_logits", k * n_groups,
                                                initializer=tf.constant_initializer(0))
                logits = tf.nn.conv2d(mdn_l, weights_logits,
                                      [1, 1, 1, 1], padding='SAME') + biases_logits

                # blur_weights
                logits = tf.reshape(logits, [-1, n_x * n_y, k * n_groups])
                logits = tf.reshape(logits, [-1, n_x * n_y * k, n_groups], name='logits_final')

            return [locs, scales, logits, X_unet]


    def get_var_list(self):
        var_list = tf.global_variables(self.net_name)
        var_list += tf.global_variables('resnet_v1_50')
        return var_list


    def train_umdn(self):

        self.joint = True
        def loss(inputs, pred):
            mdn_loss = self.my_loss(pred, inputs[0])
            unet_loss = tf.losses.mean_squared_error(inputs[-1],pred[-1])
            return mdn_loss + unet_loss/10.

        super(self.__class__, self).train(
            create_network=self.create_network,
            loss=loss,
            learning_rate=0.0001)

    def my_loss(self, X, y):

        locs_offset = float(2**5)

        mdn_locs, mdn_scales, mdn_logits, unet_out = X
        cur_comp = []
        ll = tf.nn.softmax(mdn_logits, axis=1)

        n_preds = mdn_locs.get_shape().as_list()[1]
        # All gaussians in the mixture have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        # ll = tf.cond(self.ph['phase_train'], lambda: ll + logit_eps, lambda: tf.identity(ll))
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


    def compute_dist(self, preds, locs):

        locs_offset = 32

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

    def classify_val(self, model_file=None, onTrain=False):
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

        p_m, p_s, p_w, x_unet = self.pred
        conf = self.conf
        osz = self.conf.imsz
        #       self.joint = True

        locs_offset = 32
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
        val_u_predlocs = []
        for step in range(num_val / self.conf.batch_size):
            if onTrain:
                self.fd_train()
            else:
                self.fd_val()
            pred_means, pred_std, pred_weights, cur_input = sess.run(
                [p_m, p_s, p_w, self.inputs], self.fd)
            val_means.append(pred_means)
            val_std.append(pred_std)
            val_wts.append(pred_weights)
            pred_weights = PoseUMDN.softmax(pred_weights, axis=1)
            mdn_pred_out = np.zeros([self.conf.batch_size, osz[0], osz[1], conf.n_classes])

            locs = cur_input[1]
            cur_dist = np.zeros([conf.batch_size, conf.n_classes])
            cur_predlocs = np.zeros(pred_means.shape[0:1] + pred_means.shape[2:])
            #            for ndx in range(pred_means.shape[0]):
            #                for gdx, gr in enumerate(self.conf.mdn_groups):
            #                    for g in gr:
            #                        sel_ex = np.argmax(pred_weights[ndx,:,gdx])
            #                        mm = pred_means[ndx, sel_ex, g, :]
            #                        ll = locs[ndx,g,:]
            #                        jj =  mm-ll
            #                        # jj has distance between all labels and
            #                        # all predictions with wts > 0.
            #                        dd1 = np.sqrt(np.sum(jj ** 2, axis=-1))
            #                        cur_dist[ndx, g] = dd1 * self.conf.rescale
            #                        cur_predlocs[ndx,g,...] = mm
            for sel in range(conf.batch_size):
                for cls in range(conf.n_classes):
                    for ndx in range(pred_means.shape[1]):
                        cur_gr = [l.count(cls) for l in self.conf.mdn_groups].index(1)
                        if pred_weights[sel, ndx, cur_gr] < (0.02 / self.conf.max_n_animals):
                            continue
                        cur_locs = np.round(pred_means[sel:sel + 1, ndx:ndx + 1, cls, :]).astype('int')
                        cur_scale = pred_std[sel, ndx, cls].astype('int')
                        curl = (PoseTools.create_label_images(cur_locs, osz, 1, cur_scale) + 1) / 2
                        mdn_pred_out[sel, :, :, cls] += pred_weights[sel, ndx, cur_gr] * curl[0, ..., 0]

            locs = cur_input[1]
            if locs.ndim == 3:
                cur_predlocs = PoseTools.get_pred_locs(mdn_pred_out)
                cur_dist = np.sqrt(np.sum(
                    (cur_predlocs - locs / self.conf.unet_rescale) ** 2, 2))
            else:
                cur_predlocs = PoseTools.get_pred_locs_multi(
                    mdn_pred_out, self.conf.max_n_animals,
                    self.conf.label_blur_rad * 7)
                curl = locs.copy() / self.conf.unet_rescale
                jj = cur_predlocs[:, :, np.newaxis, :, :] - curl[:, np.newaxis, ...]
                cur_dist = np.sqrt(np.sum(jj ** 2, axis=-1)).min(axis=1)
            val_dist.append(cur_dist)
            val_ims.append(cur_input[0])
            val_locs.append(cur_input[1])
            val_preds.append(mdn_pred_out)
            val_predlocs.append(cur_predlocs)

        sess.close()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape((-1,) + in_a.shape[2:])

        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_predlocs = val_reshape(val_predlocs)
        val_locs = val_reshape(val_locs)
        val_means = val_reshape(val_means)
        val_std = val_reshape(val_std)
        val_wts = val_reshape(val_wts)
        tf.reset_default_graph()

        return val_dist, val_ims, val_preds, val_predlocs, val_locs, \
               [val_means, val_std, val_wts]
