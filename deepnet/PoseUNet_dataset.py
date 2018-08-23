from PoseCommon_dataset import PoseCommon, PoseCommonMulti, PoseCommonRNN, PoseCommonTime, conv_relu3, conv_shortcut
import PoseTools
import tensorflow as tf
import os
import sys
import math
import convNetBase as CNB
import numpy as np
import movies
from PoseTools import scale_images
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile
from matplotlib import cm
import movies
import multiResData
from scipy import io as sio
import re
import json
from tensorflow.contrib.layers import batch_norm
import FusionNet

# for tf_unet
#from tf_unet_layers import (weight_variable, weight_variable_devonc, bias_variable,
#                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
#                            cross_entropy)
from collections import OrderedDict


def train_preproc_func(ims, locs, info, conf):
    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, True, conf.rescale)
    hmaps = PoseTools.create_label_images(locs, conf.imsz, conf.rescale, conf.label_blur_rad)
    return ims.astype('float32'), locs.astype('float32'), info.astype('float32'), hmaps.astype('float32')


def val_preproc_func(ims, locs, info, conf):
    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, False, conf.rescale)
    hmaps = PoseTools.create_label_images(locs, conf.imsz, conf.rescale, conf.label_blur_rad)
    return ims.astype('float32'), locs.astype('float32'), info.astype('float32'), hmaps.astype('float32')


class PoseUNet(PoseCommon):

    def __init__(self, conf, name='pose_unet'):

        PoseCommon.__init__(self, conf, name)
        self.down_layers = [] # layers created while down sampling
        self.up_layers = [] # layers created while up sampling
        self.edge_ignore = 10
        self.net_name = 'pose_unet'
        self.n_conv = 2
        self.all_layers = None
        self.for_training = 1 # for prediction.
        self.scale = self.conf.unet_rescale

        def train_pp(ims,locs,info):
            return train_preproc_func(ims,locs,info, conf)
        def val_pp(ims,locs,info):
            return val_preproc_func(ims,locs,info, conf)

        self.train_py_map = lambda ims, locs, info: tuple(tf.py_func( train_pp, [ims, locs, info], [tf.float32, tf.float32, tf.float32, tf.float32]))
        self.val_py_map = lambda ims, locs, info: tuple(tf.py_func( val_pp, [ims, locs, info], [tf.float32, tf.float32, tf.float32, tf.float32]))

    def create_network(self ):
        im, locs, info, hmap = self.inputs
        conf = self.conf
        im.set_shape([conf.batch_size, conf.imsz[0]/conf.rescale,conf.imsz[1]/conf.rescale, conf.imgDim])
        hmap.set_shape([conf.batch_size, conf.imsz[0]/conf.rescale, conf.imsz[1]/conf.rescale,conf.n_classes])
        locs.set_shape([conf.batch_size, conf.n_classes,2])
        info.set_shape([conf.batch_size,3])

        with tf.variable_scope(self.net_name):
            # return self.create_network1()
            # return self.create_network_residual()
            fn = FusionNet.FusionNet(conf.n_classes)
            return fn.inference(self.inputs[0])

    def create_network1(self):

        m_sz = min(self.conf.imsz)/self.conf.unet_rescale
        max_layers = int(math.ceil(math.log(m_sz,2)))-1
        sel_sz = self.conf.sel_sz
        n_layers = int(math.ceil(math.log(sel_sz,2)))+2
        n_layers = min(max_layers,n_layers) - 2

        # n_layers = 6

        n_conv = self.n_conv
        conv = lambda a, b: conv_relu3(
            a,b,self.ph['phase_train'], keep_prob=None,
            use_leaky=self.conf.unet_use_leaky)

        layers = []
        up_layers = []
        layers_sz = []
        X = self.inputs[0]
        n_out = self.conf.n_classes
        all_layers = []

        # downsample
        n_filt = 128
        n_filt_base = 32
        max_filt = 512
        # n_filt_base = 16
        # max_filt = 256

        for ndx in range(n_layers):
            n_filt = min(max_filt, n_filt_base * (2** ndx))

            if ndx == 0:
                n_conv = 2
            elif ndx == 1:
                n_conv = 2
            elif ndx == 2:
                n_conv = 2
            else:
                n_conv = 4

            for cndx in range(n_conv):
                sc_name = 'layerdown_{}_{}'.format(ndx,cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt)
                all_layers.append(X)
            layers.append(X)
            layers_sz.append(X.get_shape().as_list()[1:3])
            # X = tf.nn.max_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
            #                    padding='SAME')
            X = tf.nn.avg_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME')
        self.down_layers = layers

        # few more convolution for the final layers

        top_layers = []
        for cndx in range(n_conv):
            n_filt = min(max_filt, n_filt_base * (2** (n_layers)))
            sc_name = 'layer_{}_{}'.format(n_layers,cndx)
            with tf.variable_scope(sc_name):
                X = conv(X, n_filt)
                top_layers.append(X)
        self.top_layers = top_layers
        all_layers.extend(top_layers)

        # upsample
        for ndx in reversed(range(n_layers)):
            X = CNB.upscale('u_{}'.format(ndx), X, layers_sz[ndx])

            # # upsample using deconv
            # with tf.variable_scope('u_{}'.format(ndx)):
            #     X_sh = X.get_shape().as_list()
            #     w = tf.get_variable('w', [5, 5, X_sh[-1], X_sh[-1]],initializer=tf.contrib.layers.xavier_initializer())
            #     out_shape = [X_sh[0],layers_sz[ndx][0],layers_sz[ndx][1],X_sh[-1]]
            #     X = tf.nn.conv2d_transpose(X, w, output_shape=out_shape, strides=[1, 2, 2, 1], padding="SAME")
            #     biases = tf.get_variable('biases', [out_shape[-1]], initializer=tf.constant_initializer(0))
            #     conv_b = X + biases
            #
            #     bn = batch_norm(conv_b)
            #     X = tf.nn.relu(bn)

            X = tf.concat([X,layers[ndx]], axis=3)
            n_filt = min(2 * max_filt, 2 * n_filt_base* (2** ndx))

            if ndx == 0:
                n_conv = 2
            elif ndx == 1:
                n_conv = 2
            elif ndx == 2:
                n_conv = 2
            else:
                n_conv = 4

            for cndx in range(n_conv):
                sc_name = 'layerup_{}_{}'.format(ndx, cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt)
                all_layers.append(X)
            up_layers.append(X)
        self.all_layers = all_layers
        self.up_layers = up_layers

        # final conv
        weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("out_biases", n_out,
                                 initializer=tf.constant_initializer(0.))
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.add(conv, biases, name = 'unet_pred')
        # X = conv+biases
        return X

    def create_network_residual(self):

        def conv_residual(x_in,n_filt, train_phase):
            in_dim = x_in.get_shape().as_list()[3]
            kernel_shape = [3, 3, in_dim, n_filt]
            weights = tf.get_variable("weights", kernel_shape,
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", kernel_shape[-1],
                                     initializer=tf.constant_initializer(0.))
            conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = batch_norm(conv, decay=0.99, is_training=train_phase)
            return conv

        m_sz = min(self.conf.imsz)/self.conf.unet_rescale
        max_layers = int(math.ceil(math.log(m_sz,2)))-1
        sel_sz = self.conf.sel_sz
        n_layers = int(math.ceil(math.log(sel_sz,2)))+2
        n_layers = min(max_layers,n_layers) - 2

        # n_layers = 6

        n_conv = self.n_conv
        conv = lambda a, b: conv_relu3(
            a,b,self.ph['phase_train'], keep_prob=None,
            use_leaky=self.conf.unet_use_leaky)

        layers = []
        up_layers = []
        layers_sz = []
        X = self.inputs[0]
        n_out = self.conf.n_classes
        all_layers = []

        # downsample
        n_filt = 128
        n_filt_base = 32
        max_filt = 512
        # n_filt_base = 16
        # max_filt = 256

        for ndx in range(n_layers):
            n_filt = min(max_filt, n_filt_base * (2** (ndx)))
            if ndx == 0:
                with tf.variable_scope('layerdown_{}'.format(ndx)):
                    X_sh = conv_residual(X, n_filt, self.ph['phase_train'])
            else:
                X_sh = X

            X_in = X
            with tf.variable_scope('layerdown_{}_0'.format(ndx)):
                X = conv_residual(X, n_filt, self.ph['phase_train'])
                X = tf.nn.leaky_relu(X)
            with tf.variable_scope('layerdown_{}_1'.format(ndx)):
                X = conv_residual(X, n_filt, self.ph['phase_train'])
                X = X + X_sh
                X = tf.nn.leaky_relu(X)

            all_layers.append(X)
            layers.append(X)
            layers_sz.append(X.get_shape().as_list()[1:3])

            in_dim = X.get_shape().as_list()[3]
            n_filt = min(max_filt, n_filt_base * (2** (ndx+1)))
            kernel_shape = [3, 3, in_dim, n_filt]
            with tf.variable_scope('layerdown_{}_2'.format(ndx)):
                weights = tf.get_variable("weights1", kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases1", kernel_shape[-1], initializer=tf.constant_initializer(0.))
                conv = tf.nn.conv2d(X, weights, strides=[1, 2, 2, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99, is_training=self.ph['phase_train'])
                X = conv
#            X = tf.nn.relu(conv + biases)

        self.down_layers = layers

        # few more convolution for the final layers
        top_layers = []
        X_top_in = X
        n_filt = min(max_filt, n_filt_base * (2** (n_layers)))
        with tf.variable_scope('top_layer_{}_0'.format(n_layers)):
            X = conv_residual(X, n_filt, self.ph['phase_train'])
            X = tf.nn.leaky_relu(X)
        with tf.variable_scope('top_layer_{}_1'.format(n_layers)):
            X = conv_residual(X, n_filt, self.ph['phase_train'])
            X += X_top_in
            X = tf.nn.leaky_relu(X)
        top_layers.append(X)
        self.top_layers = top_layers
        all_layers.extend(top_layers)

        # upsample
        for ndx in reversed(range(n_layers)):
            X = CNB.upscale('u_'.format(ndx), X, layers_sz[ndx])
            n_filt = min(max_filt, n_filt_base* (2** ndx))
            with tf.variable_scope('layerup_{}'.format(ndx)):
                X = conv_residual(X, n_filt, self.ph['phase_train'])
                X = X +  layers[ndx]
                X_in = X
            with tf.variable_scope('layerup_{}_0'.format(ndx)):
                X = conv_residual(X, n_filt, self.ph['phase_train'])
                X = tf.nn.leaky_relu(X)
            with tf.variable_scope('layerup_{}_1'.format(ndx)):
                X = conv_residual(X, n_filt, self.ph['phase_train'])
                X += X_in
                X = tf.nn.leaky_relu(X)

            all_layers.append(X)
            up_layers.append(X)
        self.all_layers = all_layers
        self.up_layers = up_layers

        # final conv
        weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("out_biases", n_out,
                                 initializer=tf.constant_initializer(0.))
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.add(conv, biases, name = 'unet_pred')
        # X = conv+biases
        return X


    def restore_net(self, restore=True):
        return  PoseCommon.restore_net_common(self, self.create_network, restore)


    def restore_net_meta(self, train_type=0, model_file=None):

        sess, latest_model_file = PoseCommon.restore_meta_common(self, train_type, model_file)
        graph = tf.get_default_graph()

        # try:
        #     kp = graph.get_tensor_by_name('keep_prob:0')
        # except KeyError:
        #     kp = graph.get_tensor_by_name('Placeholder:0')
        # self.ph['keep_prob'] = kp

        self.ph['x'] = graph.get_tensor_by_name('x:0')
        self.ph['y'] = graph.get_tensor_by_name('y:0')
        self.ph['learning_rate'] = graph.get_tensor_by_name('learning_r:0')
        self.ph['phase_train'] = graph.get_tensor_by_name('phase_train:0')
        self.ph['db_type'] = graph.get_tensor_by_name('db_type:0')

        try:
            pred = graph.get_tensor_by_name('pose_unet/unet_pred:0')
        except KeyError:
            pred = graph.get_tensor_by_name('pose_unet/add:0')
        self.pred = pred
#        self.create_fd()
        return sess, latest_model_file


    def train_unet(self):
        def loss(inputs, pred):
            return tf.nn.l2_loss(pred-inputs[-1])

        PoseCommon.train(self,
            create_network=self.create_network,
            loss=loss,
            learning_rate=0.0001)


    def classify_val(self, model_file=None, onTrain=False):
        if not onTrain:
            val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        else:
            val_file = os.path.join(self.conf.cachedir, self.conf.trainfilename + '.tfrecords')
        print('Classifying data in {}'.format(val_file))

        num_val = 0
        for _ in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        # if at_step < 0:
        #     sess = self.init_net_meta(train_type) #,True)
        # else:

            #self.init_train(train_type)
        self.setup_train()
        self.pred = self.create_network()
        self.create_saver()
        sess = tf.Session()
        model_file = self.restore(sess,model_file)

        val_dist = []
        val_ims = []
        val_preds = []
        val_predlocs = []
        val_locs = []
        val_info = []
        for step in range(num_val/self.conf.batch_size):
            if onTrain:
                self.fd_train()
            else:
                self.fd_val()
            cur_pred, self.cur_inputs = \
                sess.run([self.pred, self.inputs], self.fd)

            cur_predlocs = PoseTools.get_pred_locs(
                cur_pred, self.edge_ignore)
            cur_dist = np.sqrt(np.sum(
                (cur_predlocs-self.cur_inputs[1]) ** 2, 2))
            val_dist.append(cur_dist)
            val_ims.append(self.cur_inputs[0])
            val_locs.append(self.cur_inputs[1])
            val_preds.append(cur_pred)
            val_predlocs.append(cur_predlocs)
            val_info.append(self.cur_inputs[2])

        sess.close()
#        self.close_cursors()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape( (-1,) + in_a.shape[2:])
        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_predlocs = val_reshape(val_predlocs)
        val_locs = val_reshape(val_locs)
        n_records = len(val_info[0][0])
        val_info = np.array(val_info).reshape([-1, n_records ])
        tf.reset_default_graph()

        dstr = PoseTools.get_datestr()

        last_iter = re.findall('\d+$',model_file)[0]
        start_at = int(last_iter)

        f_name = '_'.join([ self.conf.expname, self.name, 'cv_results','{}'.format(start_at-1),dstr])
        out_file = os.path.join(self.conf.cachedir,f_name+'.json')

        json_data = {}
        json_data['val_dist'] = val_dist.tolist()
        json_data['val_predlocs'] = val_predlocs.tolist()
        json_data['val_locs'] = np.array(val_locs).tolist()
        json_data['val_info'] = val_info.tolist()
        json_data['model_file'] = model_file
        json_data['step'] = start_at-1

        with open(out_file,'w') as f:
            json.dump(json_data,f)

        return val_dist, val_ims, val_preds, val_predlocs, val_locs, val_info


    def classify_movie(self, movie_name, sess, end_frame=-1, start_frame=0, flipud=False):
        # maxframes if specificied reads that many frames
        # start at specifies where to start reading.
        conf = self.conf

        cap = movies.Movie(movie_name)
        n_frames = int(cap.get_n_frames())

        # figure out how many frames to read
        if end_frame > 0:
            if end_frame > n_frames:
                print('End frame requested exceeds number of frames in the video. Tracking only till last valid frame')
            else:
                n_frames = end_frame - start_frame
        else:
            n_frames = n_frames - start_frame

        # pre allocate results
        bsize = conf.batch_size
        n_batches = int(math.ceil(float(n_frames)/ bsize))
        pred_locs = np.zeros([n_frames, conf.n_classes, 2])
        pred_max_scores = np.zeros([n_frames, conf.n_classes])
        pred_scores = np.zeros([n_frames,] + self.pred.get_shape().as_list()[1:])
        all_f = np.zeros((bsize,) + conf.imsz + (1,))

        for curl in range(n_batches):
            ndx_start = curl * bsize
            ndx_end = min(n_frames, (curl + 1) * bsize)
            ppe = min(ndx_end - ndx_start, bsize)
            for ii in range(ppe):
                fnum = ndx_start + ii + start_frame
                frame_in = cap.get_frame(fnum)
                if len(frame_in) == 2:
                    frame_in = frame_in[0]
                    if frame_in.ndim == 2:
                        frame_in = frame_in[:, :, np.newaxis]
                frame_in = PoseTools.crop_images(frame_in, conf)
                if flipud:
                    frame_in = np.flipud(frame_in)
                all_f[ii, ...] = frame_in[..., 0:conf.imgDim]

            # converting to uint8 is really really important!!!!!
            xs, _ = PoseTools.preprocess_ims(all_f, in_locs=np.zeros([bsize,self.conf.n_classes, 2]),
                                             conf=self.conf, distort=False, scale=self.conf.unet_rescale)

            self.fd[self.ph['x']] = xs
            self.fd[self.ph['phase_train']] = False
#            self.fd[self.ph['keep_prob']] = 1.
            pred = sess.run(self.pred, self.fd)

            base_locs = PoseTools.get_pred_locs(pred)
            base_locs = base_locs*conf.unet_rescale
            pred_locs[ndx_start:ndx_end, :, :] = base_locs[:ppe, :, :]
            pred_max_scores[ndx_start:ndx_end, :] = pred[:ppe, :, :, :].max(axis=(1,2))
            pred_scores[ndx_start:ndx_end, :, :, :] = pred[:ppe, :, :, :]
            sys.stdout.write('.')
            if curl % 20 == 19:
                sys.stdout.write('\n')

        cap.close()
        return pred_locs, pred_scores, pred_max_scores


    def classify_movie_trx(self, movie_name, trx, sess, end_frame=-1, start_frame=0, flipud=False, return_ims=False):
        # maxframes if specificied reads up to that  frame
        # start at specifies where to start reading.

        conf = self.conf
        cap = movies.Movie(movie_name)
        n_frames = int(cap.get_n_frames())
        T = sio.loadmat(trx)['trx'][0]
        n_trx = len(T)

        end_frames = np.array([x['endframe'][0,0] for x in T])
        first_frames = np.array([x['firstframe'][0,0] for x in T]) - 1 # for converting from 1 indexing to 0 indexing
        if end_frame < 0:
            end_frame = end_frames.max()
        if end_frame > end_frames.max():
            end_frame = end_frames.max()
        if start_frame > end_frame:
            return None
        max_n_frames = end_frame - start_frame
        pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
        pred_locs[:] = np.nan

        if return_ims:
            ims = np.zeros([max_n_frames, n_trx, conf.imsz[0], conf.imsz[1],conf.imgDim])
            pred_ims = np.zeros([max_n_frames, n_trx, conf.imsz[0]/conf.unet_rescale, conf.imsz[1]/conf.unet_rescale,conf.n_classes])

        bsize = conf.batch_size
        hsz_p = conf.imsz[0] / 2  # half size for pred

        for trx_ndx in range(n_trx):
            cur_trx = T[trx_ndx]
            # pre allocate results
            if first_frames[trx_ndx] > start_frame:
                cur_start = first_frames[trx_ndx]
            else:
                cur_start = start_frame

            if end_frames[trx_ndx] < end_frame:
                cur_end = end_frames[trx_ndx]
            else:
                cur_end = end_frame

            n_frames = cur_end - cur_start
            n_batches = int(math.ceil(float(n_frames)/ bsize))
            all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))

            for curl in range(n_batches):
                ndx_start = curl * bsize + cur_start
                ndx_end = min(n_frames, (curl + 1) * bsize) + cur_start
                ppe = min(ndx_end - ndx_start, bsize)
                trx_arr = []
                for ii in range(ppe):
                    fnum = ndx_start + ii
                    frame_in, cur_loc = multiResData.get_patch_trx(
                        cap, cur_trx, fnum, conf, np.zeros([conf.n_classes,2]))
                    if flipud:
                        frame_in = np.flipud(frame_in)

                    trx_fnum = fnum - first_frames[trx_ndx]
                    x = int(round(cur_trx['x'][0,trx_fnum]))-1
                    y = int(round(cur_trx['y'][0,trx_fnum]))-1
                    # -1 for 1-indexing in matlab and 0-indexing in python
                    theta = cur_trx['theta'][0,trx_fnum]
                    assert conf.imsz[0] == conf.imsz[1]
                    tt = -theta - math.pi/2
                    R = [[np.cos(tt), -np.sin(tt)], [np.sin(tt), np.cos(tt)]]
                    trx_arr.append([x,y,theta,R])
                    all_f[ii, ...] = frame_in

                xs, _ = PoseTools.preprocess_ims(all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]),
                                                 conf=self.conf,distort=False, scale=self.conf.unet_rescale)

                self.fd[self.ph['x']] = xs
                self.fd[self.ph['phase_train']] = False
                self.fd[self.ph['keep_prob']] = 1.
                pred = sess.run(self.pred, self.fd)

                base_locs = PoseTools.get_pred_locs(pred)
                base_locs = base_locs * conf.unet_rescale

                base_locs_orig = np.zeros(base_locs.shape)
                for ii in range(ppe):
                    curlocs = np.dot(base_locs[ii, :, :] - [hsz_p, hsz_p], trx_arr[ii][3]) + [trx_arr[ii][0],trx_arr[ii][1]]
                    base_locs_orig[ii,...] = curlocs

                out_start = ndx_start - start_frame
                out_end = ndx_end - start_frame
                if return_ims:
                    ims[out_start:out_end, trx_ndx,:,:,:] = all_f[:ppe,...]
                    pred_ims[out_start:out_end,trx_ndx, ...] = pred[:ppe,...]

                pred_locs[out_start:out_end, trx_ndx, :, :] = base_locs_orig[:ppe,...]
                sys.stdout.write('.')
                if curl % 20 == 19:
                    sys.stdout.write('\n')

            sys.stdout.write('\n')
        cap.close()
        tf.reset_default_graph()
        if return_ims:
            return pred_locs, pred_ims, ims
        else:
            return pred_locs


    def create_pred_movie(self, movie_name, out_movie, max_frames=-1, flipud=False, trace=True):
        conf = self.conf
        sess = self.setup_net(0, True)
        predLocs, pred_scores, pred_max_scores = self.classify_movie(movie_name, sess, end_frame=max_frames, flipud=flipud)
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
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()

            ax1.scatter(predLocs[curl, :, 0],
                        predLocs[curl, :, 1],
                c=color*0.9, linewidths=0,
                edgecolors='face',marker='+',s=45)
            if trace:
                for ndx in range(conf.n_classes):
                    curc = color[ndx,:].copy()
                    curc[3] = 0.5
                    e = np.maximum(0,curl-trace_len)
                    ax1.plot(predLocs[e:curl,ndx,0],
                             predLocs[e:curl, ndx, 1],
                             c = curc,lw=0.8)
            ax1.axis('off')
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
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


    def create_pred_movie_trx(self, movie_name, out_movie, trx, fly_num, max_frames=-1, start_at=0, flipud=False, trace=True):
        conf = self.conf
        sess = self.setup_net(0, True)
        predLocs = self.classify_movie_trx(movie_name, trx, sess, end_frame=max_frames, flipud=flipud, start_frame=start_at)
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

                # curlocs = np.dot(predLocs[curl,fndx,:,:]-[hsz_p,hsz_p],R)
                curlocs = predLocs[curl,fndx,:,:] #-[hsz_p,hsz_p]
                ax1.scatter(curlocs[ :, 0]*sc - c_x + hsz_s,
                            curlocs[ :, 1]*sc - c_y  + hsz_s,
                    c=color*0.9, linewidths=0,
                    edgecolors='face',marker='+',s=30)
                if trace:
                    for ndx in range(conf.n_classes):
                        curc = color[ndx,:].copy()
                        curc[3] = 0.5
                        e = np.maximum(0,curl-trace_len)
                        # zz = np.dot(predLocs[e:(curl+1),fndx,ndx,:]-[hsz_p,hsz_p],R)
                        zz = predLocs[e:(curl+1),fndx,ndx,:]
                        ax1.plot(zz[:,0]*sc - c_x + hsz_s,
                                 zz[:,1]*sc - c_y + hsz_s,
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


class PoseUNetMulti(PoseUNet, PoseCommonMulti):

    def __init__(self, conf, name='pose_unet_multi'):
        PoseUNet.__init__(self, conf, name)


    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess, distort)
        self.fd[self.ph['x']] = self.xs
        n_classes = self.locs.shape[2]
        sz0 = self.conf.imsz[0]
        sz1 = self.conf.imsz[1]
        label_ims = np.zeros([self.conf.batch_size, sz0, sz1, n_classes])
        for ndx in range(self.conf.batch_size):
            for i_ndx in range(self.info[ndx][2][0]):
                cur_l = PoseTools.create_label_images(
                    self.locs[ndx:ndx+1,i_ndx,...], self.conf.imsz, 1, self.conf.label_blur_rad)
                label_ims[ndx,...] = np.maximum(label_ims[ndx,...], cur_l)

        self.fd[self.ph['y']] = label_ims


    def create_cursors(self, sess):
        PoseCommonMulti.create_cursors(self,sess)


class PoseUNetTime(PoseUNet, PoseCommonTime):

    def __init__(self,conf,name='pose_unet_time'):
        PoseUNet.__init__(self, conf, name)
        self.net_name = 'pose_unet_time'

    def read_images(self, db_type, distort, sess, shuffle=None):
        PoseCommonTime.read_images(self,db_type,distort,sess,shuffle)

    def create_ph_fd(self):
        PoseCommon.create_ph_fd(self)
        imsz = self.conf.imsz
        rescale = self.conf.unet_rescale
        b_sz = self.conf.batch_size
        t_sz = self.conf.time_window_size*2 +1
        self.ph['x'] = tf.placeholder(tf.float32,
                           [b_sz*t_sz,imsz[0]/rescale,imsz[1]/rescale, self.conf.imgDim],
                           name='x')
        self.ph['y'] = tf.placeholder(tf.float32,
                           [b_sz,imsz[0]/rescale,imsz[1]/rescale, self.conf.n_classes],
                           name='y')
        self.ph['keep_prob'] = tf.placeholder(tf.float32)

    def create_fd(self):
        b_sz = self.conf.batch_size
        t_sz = self.conf.time_window_size*2 +1
        x_shape = [b_sz*t_sz,] + self.ph['x'].get_shape().as_list()[1:]
        y_shape = [b_sz,] + self.ph['y'].get_shape().as_list()[1:]
        self.fd = {self.ph['x']:np.zeros(x_shape),
                   self.ph['y']:np.zeros(y_shape),
                   self.ph['phase_train']:False,
                   self.ph['learning_rate']:0.
                   }

    def create_network1(self):
        m_sz = min(self.conf.imsz)/self.conf.unet_rescale
        max_layers = int(math.ceil(math.log(m_sz,2)))-1
        sel_sz = self.conf.sel_sz
        n_layers = int(math.ceil(math.log(sel_sz,2)))+2
        # max_layers = int(math.floor(math.log(m_sz)))
        # sel_sz = self.conf.sel_sz
        # n_layers = int(math.ceil(math.log(sel_sz)))+2
        n_layers = min(max_layers,n_layers) - 2
        mix_at = 2

        n_conv = 2
        conv = PoseCommon.conv_relu3
        layers = []
        up_layers = []
        layers_sz = []
        X = self.ph['x']
        n_out = self.conf.n_classes
        debug_layers = []

        # downsample
        for ndx in range(n_layers):
            if ndx is 0:
                n_filt = 64
            elif ndx is 1:
                n_filt = 128
            elif ndx is 2:
                n_filt = 256
            else:
                n_filt = 512

            for cndx in range(n_conv):
                sc_name = 'layerdown_{}_{}'.format(ndx,cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
                debug_layers.append(X)
            layers.append(X)
            layers_sz.append(X.get_shape().as_list()[1:3])
            X = tf.nn.avg_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME')

        self.down_layers = layers
        self.debug_layers = debug_layers
        for cndx in range(n_conv):
            sc_name = 'layer_{}_{}'.format(n_layers,cndx)
            with tf.variable_scope(sc_name):
                X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])

        # upsample
        for ndx in reversed(range(n_layers)):
            if ndx is 0:
                n_filt = 64
            elif ndx is 1:
                n_filt = 128
            elif ndx is 2:
                n_filt = 256
            else:
                n_filt = 512
            X = CNB.upscale('u_'.format(ndx), X, layers_sz[ndx])

            if ndx is mix_at:
            # rotate X along axis-0 and concat to provide context context along previous time steps.
                X_prev = []
                X_next = []
                for t in range(self.conf.time_window_size):
                    if not X_prev:
                        X_prev_cur = tf.concat([X[1:,...],X[0:1,...]],axis=0)
                        X_prev.append(X_prev_cur)
                        X_next_cur = tf.concat([X[-1:,...],X[:-1,...]],axis=0)
                        X_next.append(X_next_cur)
                    else:
                        Z = X_prev[-1]
                        X_prev_cur = tf.concat([Z[1:,...],Z[0:1,...]],axis=0)
                        X_prev.append(X_prev_cur)
                        Z = X_next[-1]
                        X_next_cur = tf.concat([Z[-1:,...],Z[:-1,...]],axis=0)
                        X_next.append(X_next_cur)

                X = tf.concat( X_next + [X, ]+ X_prev, axis = 3)

            X = tf.concat([X,layers[ndx]], axis=3)
            for cndx in range(n_conv):
                sc_name = 'layerup_{}_{}'.format(ndx, cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
            up_layers.append(X)
        self.up_layers = up_layers

        # final conv
        weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("out_biases", n_out,
                                 initializer=tf.constant_initializer(0.))
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
        X = conv + biases

        t_sz = self.conf.time_window_size
        s_sz = 2*t_sz+1
        X = X[t_sz::s_sz,...]
        return X

    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess, distort)
        rescale = self.conf.unet_rescale
        xs = scale_images(self.xs, rescale, self.conf)
        self.fd[self.ph['x']] = PoseTools.normalize_mean(xs, self.conf)
        imsz = [self.conf.imsz[0]/rescale, self.conf.imsz[1]/rescale,]
        label_ims = PoseTools.create_label_images(
            self.locs/rescale, imsz, 1, self.conf.label_blur_rad)
        self.fd[self.ph['y']] = label_ims


class PoseUNetRNN(PoseUNet, PoseCommonRNN):

    def __init__(self, conf, name='pose_unet_rnn', unet_name='pose_unet',joint=True):
        PoseCommon.__init__(self, conf, name)
        self.dep_nets = PoseUNet(conf, unet_name)
        self.net_name = 'pose_unet_rnn'
        self.dep_nets.keep_prob = 1.
        self.net_unet_name = 'pose_unet'
        self.unet_name = unet_name
        self.joint = joint
        self.keep_prob = 0.7
        self.edge_ignore = 1


    def read_images(self, db_type, distort, sess, shuffle=None):
        PoseCommonRNN.read_images(self,db_type,distort,sess,shuffle)


    def create_ph_fd(self):
        PoseCommon.create_ph_fd(self)
        imsz = self.conf.imsz
        rescale = self.conf.unet_rescale
        b_sz = self.conf.batch_size
        t_sz = self.conf.rnn_before + self.conf.rnn_after + 1
        self.ph['x'] = tf.placeholder(tf.float32,
                           [b_sz*t_sz,imsz[0]/rescale,imsz[1]/rescale, self.conf.imgDim],
                           name='x')
        self.ph['y'] = tf.placeholder(tf.float32,
                           [b_sz,imsz[0]/rescale,imsz[1]/rescale, self.conf.n_classes],
                           name='y')
        self.ph['keep_prob'] = tf.placeholder(tf.float32)
        self.ph['rnn_keep_prob'] = tf.placeholder(tf.float32)


    def create_fd(self):
        b_sz = self.conf.batch_size
        t_sz = self.conf.rnn_before + self.conf.rnn_after + 1
        x_shape = [b_sz*t_sz,] + self.ph['x'].get_shape().as_list()[1:]
        y_shape = [b_sz,] + self.ph['y'].get_shape().as_list()[1:]
        self.fd = {self.ph['x']:np.zeros(x_shape),
                   self.ph['y']:np.zeros(y_shape),
                   self.ph['phase_train']:False,
                   self.ph['learning_rate']:0.
                   }


    def create_network(self):
        with tf.variable_scope(self.net_unet_name):
            m_sz = min(self.conf.imsz)/self.conf.unet_rescale
            max_layers = int(math.ceil(math.log(m_sz,2)))-1
            sel_sz = self.conf.sel_sz
            n_layers = int(math.ceil(math.log(sel_sz,2)))+2
            # max_layers = int(math.floor(math.log(m_sz)))
            # sel_sz = self.conf.sel_sz
            # n_layers = int(math.ceil(math.log(sel_sz)))+2
            n_layers = min(max_layers,n_layers) - 2
            mix_at = 2

            n_conv = 2
            conv = PoseCommon.conv_relu3
            layers = []
            up_layers = []
            layers_sz = []
            X = self.ph['x']
            n_out = self.conf.n_classes
            debug_layers = []

            n_filt = 128
            # downsample
            for ndx in range(n_layers):
                # if ndx is 0:
                #     n_filt = 64
                # elif ndx is 1:
                #     n_filt = 128
                # elif ndx is 2:
                #     n_filt = 256
                # else:
                #     n_filt = 512

                for cndx in range(n_conv):
                    sc_name = 'layerdown_{}_{}'.format(ndx,cndx)
                    with tf.variable_scope(sc_name):
                        X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
                    debug_layers.append(X)
                layers.append(X)
                layers_sz.append(X.get_shape().as_list()[1:3])
                X = tf.nn.avg_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
                                   padding='SAME')

            self.down_layers = layers
            self.debug_layers = debug_layers

        if not self.joint:
            X = self.create_top_layer_notjoint(X, conv, n_filt, n_layers)
        else:
            X = self.create_top_layer_joint(X, conv, n_filt, n_layers)

            # upsample
        with tf.variable_scope(self.net_unet_name):
            for ndx in reversed(range(n_layers)):
                # if ndx is 0:
                #     n_filt = 64
                # elif ndx is 1:
                #     n_filt = 128
                # elif ndx is 2:
                #     n_filt = 256
                # else:
                #     n_filt = 512
                X = CNB.upscale('u_'.format(ndx), X, layers_sz[ndx])

                X = tf.concat([X, self.slice_time(layers[ndx])], axis=3)
                for cndx in range(n_conv):
                    sc_name = 'layerup_{}_{}'.format(ndx, cndx)
                    with tf.variable_scope(sc_name):
                        X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
                up_layers.append(X)
            self.up_layers = up_layers

            # final conv
            weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("out_biases", n_out,
                                     initializer=tf.constant_initializer(0.))
            conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
            X = conv + biases

        return X


    def slice_time(self, X):
        bsz = self.conf.batch_size
        tw = (self.conf.rnn_before + self.conf.rnn_after + 1)
        X_shape = X.get_shape().as_list()
        X = tf.reshape(X, [bsz, tw] + X_shape[1:])
        return X[:, self.conf.rnn_before, :, :, :]


    def create_top_layer_joint(self, X, conv, n_filt, n_layers):
        bsz = self.conf.batch_size
        tw = (self.conf.rnn_before + self.conf.rnn_after + 1)

        top_layers = []
        with tf.variable_scope(self.net_unet_name):
            for cndx in range(2):
                sc_name = 'layer_{}_{}'.format(n_layers, cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
                    top_layers.append(X)
            self.top_layers = top_layers

        in_layer = self.top_layers[0]
        in_shape = in_layer.get_shape().as_list()
        in_units = np.prod(in_shape[1:])
        in_layer = tf.reshape(in_layer, [bsz, tw, in_units])

        out_layer = self.top_layers[1]
        out_shape = out_layer.get_shape().as_list()
        n_units = np.prod(out_shape[1:])
        n_rnn_layers = 3

        with tf.variable_scope(self.net_name):
            cells = []
            for _ in range(n_rnn_layers):
                cell = tf.contrib.rnn.GRUCell(n_units)
                cell = tf.contrib.rnn.DropoutWrapper(cell, self.ph['rnn_keep_prob'])
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            rnn_out, _ = tf.nn.dynamic_rnn(cell, in_layer, dtype=tf.float32)
            rnn_out = tf.transpose(rnn_out, [1, 0, 2])
            out = tf.gather(rnn_out, int(rnn_out.get_shape()[0]) - 1)
        self.rnn_out = out
        self.rnn_in = in_layer
        self.rnn_label = None

        return tf.reshape(out,[bsz,] + out_shape[1:])

    def create_top_layer_notjoint(self, X, conv, n_filt, n_layers):
        bsz = self.conf.batch_size
        tw = (self.conf.rnn_before + self.conf.rnn_after + 1)

        top_layers = []
        with tf.variable_scope(self.net_unet_name):
            for cndx in range(2):
                sc_name = 'layer_{}_{}'.format(n_layers, cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
                    top_layers.append(X)
            self.top_layers = top_layers
        X = self.slice_time(X)

        in_layer = tf.stop_gradient(self.top_layers[0])
        in_shape = in_layer.get_shape().as_list()
        in_units = np.prod(in_shape[1:])
        in_layer = tf.reshape(in_layer, [bsz, tw, in_units])

        out_layer = self.top_layers[1]
        out_shape = out_layer.get_shape().as_list()
        n_units = np.prod(out_shape[1:])
        n_rnn_layers = 3

        with tf.variable_scope(self.net_name):
            cells = []
            for _ in range(n_rnn_layers):
                cell = tf.contrib.rnn.GRUCell(n_units)
                cell = tf.contrib.rnn.DropoutWrapper(cell, self.ph['rnn_keep_prob'])
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            rnn_out, _ = tf.nn.dynamic_rnn(cell, in_layer, dtype=tf.float32)
            rnn_out = tf.transpose(rnn_out, [1, 0, 2])
            out = tf.gather(rnn_out, int(rnn_out.get_shape()[0]) - 1)
        self.rnn_out = out
        self.rnn_in = in_layer
        self.rnn_label = tf.stop_gradient(tf.reshape(X,[self.conf.batch_size,-1]))

        return X

    def fd_train(self):
        self.fd[self.ph['phase_train']] = True
        self.fd[self.ph['keep_prob']] = self.keep_prob
        self.fd[self.ph['rnn_keep_prob']] = self.keep_prob

    def fd_val(self):
        self.fd[self.ph['phase_train']] = False
        self.fd[self.ph['keep_prob']] = 1.
        self.fd[self.ph['rnn_keep_prob']] = 1.


    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess, distort)
        rescale = self.conf.unet_rescale
        xs = scale_images(self.xs, rescale, self.conf)
        self.fd[self.ph['x']] = PoseTools.normalize_mean(xs, self.conf)
        imsz = [self.conf.imsz[0]/rescale, self.conf.imsz[1]/rescale,]
        label_ims = PoseTools.create_label_images(
            self.locs/rescale, imsz, 1, self.conf.label_blur_rad)
        self.fd[self.ph['y']] = label_ims


    def loss(self, pred_in, pred_out):
        if not self.joint:
            return tf.nn.l2_loss(self.rnn_out-self.rnn_label)
        else:
            return tf.nn.l2_loss(pred_in-pred_out)


    def train_unet_rnn(self, restore, train_type=0):

        if self.joint:
            training_iters = 20000
        else:
            training_iters = 4000

        PoseCommon.train(self,
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=training_iters,
            loss=self.loss,
            pred_in_key='y',
            learning_rate=0.0001,
            td_fields=('loss','dist'))

