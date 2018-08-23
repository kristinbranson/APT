import PoseCommon
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
# for tf_unet
#from tf_unet_layers import (weight_variable, weight_variable_devonc, bias_variable,
#                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
#                            cross_entropy)
from collections import OrderedDict

class PoseUNet(PoseCommon.PoseCommon):

    def __init__(self, conf, name='pose_unet'):

        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.down_layers = [] # layers created while down sampling
        self.up_layers = [] # layers created while up sampling
        self.edge_ignore = 10
        self.net_name = 'pose_unet'
        self.keep_prob = conf.unet_keep_prob
        self.n_conv = 2
        self.all_layers = None
        self.for_training = 1 # for prediction.
        self.scale = self.conf.unet_rescale


    def create_q_specs(self):
        PoseCommon.PoseCommon.create_q_specs(self)
        self.q_placeholder_spec.append(['label_ims',[self.conf.batch_size,
                                                self.conf.imsz[0]//self.conf.unet_rescale,
                                                self.conf.imsz[1]//self.conf.unet_rescale,
                                                self.conf.n_classes]])


    def create_ph_fd(self):
        # create feed dict and place holders.

        PoseCommon.PoseCommon.create_ph_fd(self)
#        self.ph['keep_prob'] = tf.placeholder(tf.float32,name='keep_prob')
        self.ph['keep_prob'] = None
        self.ph['db_type'] = tf.placeholder(tf.int8,name='db_type')

        batch_out = tf.cond(tf.equal(self.ph['db_type'], 1),
                lambda: self.train_dequeue_op,
                lambda: self.val_dequeue_op)

        names = [k for k,v in self.q_placeholders]
        batch = {}
        for idx, name in enumerate(names):
            batch[name] = batch_out[idx]

        if self.for_training ==0 or self.for_training==2:
            # Add zero so that we can name the operation to access them later
            # when reloading with init_net_meta
            self.ph['x'] = tf.add(batch['images'],0,name='x')
            self.ph['y'] = tf.add(batch['label_ims'],0,name='y')
            img_shape = [v for k,v in self.q_placeholder_spec if k=='images']
            self.ph['x'].set_shape(img_shape[0])
            img_shape = [v for k,v in self.q_placeholder_spec if k=='label_ims']
            self.ph['y'].set_shape(img_shape[0])
            self.locs_op = tf.add(batch['locs'],0,name='locs_op')
            self.info_op = tf.add(batch['info'],0,name='info_op')
            self.orig_xs_op = tf.add(batch['orig_images'],0,name='orig_images_op')
            self.orig_locs_op = tf.add(batch['orig_locs'],0,name='orig_locs_op')
            self.extra_data_op = tf.add(batch['extra_info'],0,name='extra_data_op')

            self.fd = {self.ph['phase_train']:False,
                       self.ph['learning_rate']:0.,
#                       self.ph['keep_prob']:1.,
                       self.ph['db_type']:1
                       }
        else:
            imsz = self.conf.imsz
            rescale = self.conf.unet_rescale
            rimsz = [int(float(imsz[0])/rescale),int(float(imsz[0])/rescale)]
            self.ph['x'] = tf.placeholder(tf.float32,
                               [None,rimsz[0],rimsz[1], self.conf.imgDim],
                               name='x')
            self.ph['y'] = tf.placeholder(tf.float32,
                               [None,rimsz[0],rimsz[1], self.conf.n_classes],
                               name='y')

            x_shape = [self.conf.batch_size, ] + self.ph['x'].get_shape().as_list()[1:]
            y_shape = [self.conf.batch_size, ] + self.ph['y'].get_shape().as_list()[1:]
            self.fd = {  self.ph['x']:np.zeros(x_shape),
                self.ph['y']:np.zeros(y_shape),
                self.ph['phase_train']: False,
                self.ph['learning_rate']: 0.,
#                self.ph['keep_prob']: 1.,
                self.ph['db_type']: 1
            }

    def create_update_fd_fn(self):
        rescale = self.conf.unet_rescale
        imsz = [self.conf.imsz[0]/rescale, self.conf.imsz[1]/rescale,]

        def update_fn(batch):
            label_ims = PoseTools.create_label_images(
                batch['locs'], imsz, 1, self.conf.label_blur_rad)
            batch['label_ims'] = label_ims

        return update_fn


    # def create_fd(self):
    #     x_shape = [self.conf.batch_size,] + self.ph['x'].get_shape().as_list()[1:]
    #     y_shape = [self.conf.batch_size,] + self.ph['y'].get_shape().as_list()[1:]
    #     self.fd = {self.ph['x']:np.zeros(x_shape),
    #                self.ph['y']:np.zeros(y_shape),
    #                self.ph['phase_train']:False,
    #                self.ph['learning_rate']:0.,
    #                self.ph['keep_prob']:1.
    #                }
    #

    def create_network(self, ):
        with tf.variable_scope(self.net_name):
            return self.create_network1()

    # def compute_dist(self, preds, locs):
    #     tt1 = PoseTools.get_pred_locs(preds,self.edge_ignore) - \
    #           locs
    #     tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
    #     return np.nanmean(tt1)
    #

    def create_network1(self):
        m_sz = min(self.conf.imsz)/self.conf.unet_rescale
        max_layers = int(math.ceil(math.log(m_sz,2)))-1
        sel_sz = self.conf.sel_sz
        n_layers = int(math.ceil(math.log(sel_sz,2)))+2
        n_layers = min(max_layers,n_layers) - 2
        n_conv = self.n_conv
        conv = lambda a, b: PoseCommon.conv_relu3(
            a,b,self.ph['phase_train'], keep_prob=None,
            use_leaky=self.conf.unet_use_leaky)

        layers = []
        up_layers = []
        layers_sz = []
        X = self.ph['x']
        n_out = self.conf.n_classes
        all_layers = []

        # downsample
        n_filt = 128
        n_filt_base = 32
        max_filt = 512
        for ndx in range(n_layers):
            n_filt = min(max_filt, n_filt_base * (2** ndx))
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
            X = CNB.upscale('u_'.format(ndx), X, layers_sz[ndx])
            X = tf.concat([X,layers[ndx]], axis=3)
            n_filt = min(2 * max_filt, 2 * n_filt_base* (2** ndx))
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

    # def create_network_tf_unet(self):
    #     # implementation of unet from https://github.com/jakeret/tf_unet/blob/master/tf_unet/unet.py
    #     x = self.ph['x']
    #     nx = tf.shape(x)[1]
    #     ny = tf.shape(x)[2]
    #     # x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
    #     x_image = x
    #     in_node = x_image
    #     batch_size = tf.shape(x_image)[0]
    #
    #     weights = []
    #     biases = []
    #     convs = []
    #     pools = OrderedDict()
    #     deconv = OrderedDict()
    #     dw_h_convs = OrderedDict()
    #     up_h_convs = OrderedDict()
    #
    #     layers = 5
    #     features_root = 16
    #     filter_size = 3
    #     pool_size = 2
    #     channels = self.conf.imgDim
    #     keep_prob = 1.
    #     n_class = self.conf.n_classes
    #
    #     in_size = 1000
    #     size = in_size
    #     # down layers
    #     for layer in range(0, layers):
    #         features = 2 ** layer * features_root
    #         stddev = np.sqrt(2 / (filter_size ** 2 * features))
    #         if layer == 0:
    #             w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
    #         else:
    #             w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev)
    #
    #         w2 = weight_variable([filter_size, filter_size, features, features], stddev)
    #         b1 = bias_variable([features])
    #         b2 = bias_variable([features])
    #
    #         conv1 = conv2d(in_node, w1, keep_prob)
    #         tmp_h_conv = tf.nn.relu(conv1 + b1)
    #         conv2 = conv2d(tmp_h_conv, w2, keep_prob)
    #         dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
    #
    #         weights.append((w1, w2))
    #         biases.append((b1, b2))
    #         convs.append((conv1, conv2))
    #
    #         size -= 4
    #         if layer < layers - 1:
    #             pools[layer] = max_pool(dw_h_convs[layer], pool_size)
    #             in_node = pools[layer]
    #             size /= 2
    #
    #     in_node = dw_h_convs[layers - 1]
    #
    #     # up layers
    #     for layer in range(layers - 2, -1, -1):
    #         features = 2 ** (layer + 1) * features_root
    #         stddev = np.sqrt(2 / (filter_size ** 2 * features))
    #
    #         wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev)
    #         bd = bias_variable([features // 2])
    #         h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
    #         h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
    #         deconv[layer] = h_deconv_concat
    #
    #         w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev)
    #         w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
    #         b1 = bias_variable([features // 2])
    #         b2 = bias_variable([features // 2])
    #
    #         conv1 = conv2d(h_deconv_concat, w1, keep_prob)
    #         h_conv = tf.nn.relu(conv1 + b1)
    #         conv2 = conv2d(h_conv, w2, keep_prob)
    #         in_node = tf.nn.relu(conv2 + b2)
    #         up_h_convs[layer] = in_node
    #
    #         weights.append((w1, w2))
    #         biases.append((b1, b2))
    #         convs.append((conv1, conv2))
    #
    #         size *= 2
    #         size -= 4
    #
    #     # Output Map
    #     weight = weight_variable([1, 1, features_root, n_class], stddev)
    #     bias = bias_variable([n_class])
    #     conv = conv2d(in_node, weight, tf.constant(1.0))
    #     output_map = tf.nn.relu(conv + bias)
    #     up_h_convs["out"] = output_map
    #
    #     variables = []
    #     for w1, w2 in weights:
    #         variables.append(w1)
    #         variables.append(w2)
    #
    #     for b1, b2 in biases:
    #         variables.append(b1)
    #         variables.append(b2)
    #
    #     return output_map #, variables, int(in_size - size)

    def fd_train(self):
        self.fd[self.ph['phase_train']] = True
        # self.fd[self.ph['keep_prob']] = self.keep_prob
        self.fd[self.ph['db_type']] = 1

    def fd_val(self):
        self.fd[self.ph['phase_train']] = False
        # self.fd[self.ph['keep_prob']] = 1.
        self.fd[self.ph['db_type']] = 0

    def update_fd_nothread(self, db_type, sess, distort):
        bb = self.dequeue_thread_op(sess, db_type)

        self.read_images(db_type, distort, sess, distort, self.conf.unet_rescale)
        self.fd[self.ph['x']] = self.xs

        rescale = self.conf.unet_rescale
        imsz = self.conf.imsz
        rimsz = [int(float(imsz[0])/rescale), int(float(imsz[1])/rescale),]
        label_ims = PoseTools.create_label_images(
            self.locs, rimsz, 1, self.conf.label_blur_rad)
        self.fd[self.ph['y']] = label_ims


    def update_fd(self, db_type, sess, distort):
        pass
        # batch = self.dequeue_thread_op(sess, db_type)
        # self.fd[self.ph['y']] = batch['label_ims']
        # self.fd[self.ph['x']] = batch['images']

    def init_net(self, train_type=0, restore=True):
        return  self.init_net_common(self.create_network,
                                       train_type, restore)


    def init_net_meta(self, train_type=0, model_file=None):

        sess, latest_model_file = PoseCommon.PoseCommon.init_net_meta(self, train_type, model_file)
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


    def train_unet(self, restore, train_type=0):
        self.for_training = 0

        def loss(pred_in, pred_out):
            return tf.nn.l2_loss(pred_in-pred_out)

        PoseCommon.PoseCommon.train(self,
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=self.conf.unet_steps,
            loss=loss,
            pred_in_key='y',
            learning_rate=0.0001,
            td_fields=('loss','dist'))


    def classify_val(self, train_type=0, at_step=-1, onTrain=False):
        self.for_training = 2
        if train_type is 0:
            if not onTrain:
                val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
            else:
                val_file = os.path.join(self.conf.cachedir, self.conf.trainfilename + '.tfrecords')
        else:
            val_file = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename + '.tfrecords')
        print('Classifying data in {}'.format(val_file))

        num_val = 0
        for _ in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        if at_step < 0:
            sess = self.init_net_meta(train_type) #,True)
        else:

            self.init_train(train_type)
            self.pred = self.create_network()
            self.create_saver()
            sess = tf.Session()
            start_at = self.init_and_restore(sess,True,['loss','dist'],distort=False, shuffle=False,
                                  at_step=at_step)

        val_dist = []
        val_ims = []
        val_preds = []
        val_predlocs = []
        val_locs = []
        val_info = []
        for step in range(num_val/self.conf.batch_size):
            if onTrain:
                self.setup_train(sess, distort=False,treat_as_val=True)
            else:
                self.setup_val(sess)
            cur_pred, self.locs, self.info, self.xs = sess.run(
                [self.pred, self.locs_op, self.info_op, self.ph['x']], self.fd)
            cur_predlocs = PoseTools.get_pred_locs(
                cur_pred, self.edge_ignore)
            cur_dist = np.sqrt(np.sum(
                (cur_predlocs-self.locs) ** 2, 2))
            val_dist.append(cur_dist)
            val_ims.append(self.xs)
            val_locs.append(self.locs)
            val_preds.append(cur_pred)
            val_predlocs.append(cur_predlocs)
            val_info.append(self.info)

        sess.close()
        self.close_cursors()

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
        if at_step < 0:
            model_file = self.get_latest_model_file()
            start_at = int(re.findall('\d+$',model_file)[0]) + 1
        else:
            model_file = self.get_latest_model_file()
            last_iter = re.findall('\d+$',model_file)[0]
            model_file.replace(last_iter,'{}'.format(start_at-1))


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
        sess = self.init_net(0,True)
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
        sess = self.init_net(0,True)
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


class PoseUNetMulti(PoseUNet, PoseCommon.PoseCommonMulti):

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
        PoseCommon.PoseCommonMulti.create_cursors(self,sess)


class PoseUNetTime(PoseUNet, PoseCommon.PoseCommonTime):

    def __init__(self,conf,name='pose_unet_time'):
        PoseUNet.__init__(self, conf, name)
        self.net_name = 'pose_unet_time'

    def read_images(self, db_type, distort, sess, shuffle=None):
        PoseCommon.PoseCommonTime.read_images(self,db_type,distort,sess,shuffle)

    def create_ph_fd(self):
        PoseCommon.PoseCommon.create_ph_fd(self)
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


class PoseUNetRNN(PoseUNet, PoseCommon.PoseCommonRNN):

    def __init__(self, conf, name='pose_unet_rnn', unet_name='pose_unet',joint=True):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.dep_nets = PoseUNet(conf, unet_name)
        self.net_name = 'pose_unet_rnn'
        self.dep_nets.keep_prob = 1.
        self.net_unet_name = 'pose_unet'
        self.unet_name = unet_name
        self.joint = joint
        self.keep_prob = 0.7
        self.edge_ignore = 1


    def read_images(self, db_type, distort, sess, shuffle=None):
        PoseCommon.PoseCommonRNN.read_images(self,db_type,distort,sess,shuffle)


    def create_ph_fd(self):
        PoseCommon.PoseCommon.create_ph_fd(self)
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

        PoseCommon.PoseCommon.train(self,
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=training_iters,
            loss=self.loss,
            pred_in_key='y',
            learning_rate=0.0001,
            td_fields=('loss','dist'))

