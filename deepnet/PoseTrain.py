from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import tensorflow as tf
import os, sys, shutil
import tempfile, copy, re
from enum import Enum
import localSetup
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math, cv2, scipy, time, pickle
import numpy as np

import PoseTools, myutils, multiResData

import convNetBase as CNB
from batch_norm import batch_norm


class PoseTrain(object):
    Nets = Enum('Nets', 'Base Joint Fine')
    TrainingType = Enum('TrainingType', 'Base MRF Fine All')

    # DBType = Enum('DBType','Train Val')
    class DBType(Enum):
        Train = 1
        Val = 2

    def __init__(self, conf):
        self.conf = conf
        self.feed_dict = {}
        self.coord = None
        self.threads = None
        self.xs = None
        self.basePred = None
        self.trainType = None
        self.locs = None
        self.info = None
        self.train_data = None
        self.val_data = None
        self.train_queue = None
        self.val_queue = None
        self.ph = {}
        self.baseLayers = None
        self.basesaver = None
        self.basestartat = None
        self.basetrainData = None
        self.fine_labels = None
        self.finePred = None
        self.finesaver = None
        self.finestartat = None
        self.finetrainData = None
        self.mrfPred = None
        self.mrftrainData = None
        self.mrfsaver = None
        self.mrfstartat = None
        self.read_time = 0.
        self.jointsaver = None
        self.jointstartat = None
        self.jointtrainData = None
        self.pred_fine_in = None
        self.opt_time = 0.
        self.doBatchNorm = None
        self.cost = None
        self.opt = None

    # ---------------- DATABASE ---------------------

    def openDBs(self):
        #         lmdbfilename =os.path.join(self.conf.cachedir,self.conf.trainfilename)
        #         vallmdbfilename =os.path.join(self.conf.cachedir,self.conf.valfilename)
        #         self.env = lmdb.open(lmdbfilename, readonly = True)
        #         self.valenv = lmdb.open(vallmdbfilename, readonly = True)
        if self.trainType == 0:
            train_filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + '.tfrecords'
            val_filename = os.path.join(self.conf.cachedir, self.conf.valfilename) + '.tfrecords'
            self.train_queue = tf.train.string_input_producer([train_filename])
            self.val_queue = tf.train.string_input_producer([val_filename])
        else:
            train_filename = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename) + '.tfrecords'
            val_filename = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename) + '.tfrecords'
            self.train_queue = tf.train.string_input_producer([train_filename])
            self.val_queue = tf.train.string_input_producer([val_filename])

    # def openHoldoutDBs(self):
    #     lmdbfilename =os.path.join(self.conf.cachedir,self.conf.holdouttrain)
    #     vallmdbfilename =os.path.join(self.conf.cachedir,self.conf.holdouttest)
    #     self.env = lmdb.open(lmdbfilename, readonly = True)
    #     self.valenv = lmdb.open(vallmdbfilename, readonly = True)

    def createCursors(self, sess):
        #         if txn is None:
        #             txn = self.env.begin()
        #             valtxn = self.valenv.begin()
        #         self.train_cursor = txn.cursor();
        #         self.val_cursor = valtxn.cursor()
        train_ims, train_locs, train_info = multiResData.read_and_decode(self.train_queue, self.conf)
        val_ims, val_locs, val_info = multiResData.read_and_decode(self.val_queue, self.conf)
        self.train_data = [train_ims, train_locs, train_info]
        self.val_data = [val_ims, val_locs, val_info]
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def closeCursors(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

    def readImages(self, dbType, distort, sess):
        conf = self.conf
        #         curcursor = self.val_cursor if (dbType == self.DBType.Val) \
        #                     else self.train_cursor
        #         xs, locs = PoseTools.readLMDB(curcursor,
        #                          conf.batch_size,conf.imsz,multiResData)
        cur_data = self.val_data if (dbType == self.DBType.Val)                 else self.train_data
        xs = []
        locs = []
        info = []
        for _ in range(conf.batch_size):
            [curxs, curlocs, curinfo] = sess.run(cur_data)
            if np.ndim(curxs) < 3:
                xs.append(curxs[np.newaxis, :, :])
            else:
                curxs = np.transpose(curxs, [2, 0, 1])
                xs.append(curxs)
            locs.append(curlocs)
            info.append(curinfo)
        xs = np.array(xs)
        locs = np.array(locs)
        locs = multiResData.sanitize_locs(locs)
        if distort:
            if conf.horzFlip:
                xs, locs = PoseTools.randomly_flip_lr(xs, locs)
            if conf.vertFlip:
                xs, locs = PoseTools.randomly_flip_ud(xs, locs)
            xs, locs = PoseTools.randomly_rotate(xs, locs, conf)
            xs, locs = PoseTools.randomly_translate(xs, locs, conf)
            xs = PoseTools.randomly_adjust(xs, conf)

        self.xs = xs
        self.locs = locs
        self.info = info

    def createPH(self):
        x0, x1, x2, y, keep_prob = CNB.createPlaceHolders(self.conf.imsz,
                                                          self.conf.rescale, self.conf.scale, self.conf.pool_scale,
                                                          self.conf.n_classes, self.conf.imgDim)
        fine_pred_in = tf.placeholder(tf.float32, y.shape, name='fine_pred_in')
        fine_pred_locs_in = tf.placeholder(tf.float32, [self.conf.batch_size,
                                                        self.conf.n_classes, 2], name='fine_pred_locs_in')
        locs_ph = tf.placeholder(tf.float32, [self.conf.batch_size,
                                              self.conf.n_classes, 2])
        learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_r')
        phase_train_base = tf.placeholder(tf.bool, name='phase_train_base')
        phase_train_fine = tf.placeholder(tf.bool, name='phase_train_fine')
        self.ph = {'x0': x0, 'x1': x1, 'x2': x2,
                   'y': y, 'keep_prob': keep_prob, 'locs': locs_ph,
                   'fine_pred_in': fine_pred_in, 'fine_pred_locs_in': fine_pred_locs_in,
                   'phase_train_base': phase_train_base,
                   'phase_train_fine': phase_train_fine,
                   'learning_rate': learning_rate_ph}

    def createFeedDict(self):
        predsz = self.ph['y'].shape.as_list()
        fine_pred_dummy = np.zeros([self.conf.batch_size, ] + predsz[1:])
        fine_pred_locs_dummy = np.ones([self.conf.batch_size, self.conf.n_classes, 2]) * predsz[1] / 2
        self.feed_dict = {self.ph['x0']: [],
                          self.ph['x1']: [],
                          self.ph['x2']: [],
                          self.ph['y']: [],
                          self.ph['fine_pred_in']: fine_pred_dummy,
                          self.ph['fine_pred_locs_in']: fine_pred_locs_dummy,
                          self.ph['keep_prob']: 1.,
                          self.ph['learning_rate']: 1,
                          self.ph['phase_train_base']: False,
                          self.ph['phase_train_fine']: False,
                          self.ph['locs']: []}

    def updateFeedDict(self, dbType, distort, sess):
        conf = self.conf
        self.readImages(dbType, distort, sess)
        x0, x1, x2 = PoseTools.multi_scale_images(self.xs.transpose([0, 2, 3, 1]),
                                                  conf.rescale, conf.scale, conf)

        labelims = PoseTools.create_label_images(self.locs,
                                                 self.conf.imsz,
                                                 self.conf.pool_scale * self.conf.rescale,
                                                 self.conf.label_blur_rad)
        self.feed_dict[self.ph['x0']] = x0
        self.feed_dict[self.ph['x1']] = x1
        self.feed_dict[self.ph['x2']] = x2
        self.feed_dict[self.ph['y']] = labelims
        self.feed_dict[self.ph['locs']] = self.locs

        # For fine stuff.
        predsz = self.ph['y'].shape.as_list()
        fine_pred_dummy = np.zeros([self.conf.batch_size, ] + predsz[1:])
        fine_pred_locs_dummy = np.ones([self.conf.batch_size, self.conf.n_classes, 2]) * predsz[1] / 2

        self.feed_dict[self.ph['fine_pred_in']] = fine_pred_dummy
        self.feed_dict[self.ph['fine_pred_locs_in']] = fine_pred_locs_dummy

    def updateFeedDictFine(self, dbType, distort, sess):
        self.updateFeedDict(dbType, distort, sess)
        pred = sess.run(self.pred_fine_in, feed_dict=self.feed_dict)
        self.feed_dict[self.ph['fine_pred_in']] = pred
        baseLocs = PoseTools.get_base_pred_locs(pred, self.conf)
        j_sz = 2 * self.conf.rescale * self.conf.pool_scale
        baseLocs += np.random.randint(-j_sz, j_sz, baseLocs.shape)
        self.feed_dict[self.ph['fine_pred_locs_in']] = baseLocs

    # ---------------- NETWORK ---------------------

    def createBaseNetwork(self, doBatch):
        pred, layers = CNB.net_multi_conv(self.ph['x0'], self.ph['x1'],
                                          self.ph['x2'], self.ph['keep_prob'],
                                          self.conf, doBatch,
                                          self.ph['phase_train_base']
                                          )
        self.basePred = pred
        self.baseLayers = layers

    def createBaseNetwork_vgg(self):

        return None

    def createMRFNetwork(self, jointTraining=False):

        n_classes = self.conf.n_classes
        base_pred = self.basePred if jointTraining else tf.stop_gradient(self.basePred)
        mrf_weights = PoseTools.init_mrf_weights(self.conf).astype('float32')
        mrf_weights = old_div(mrf_weights, n_classes)

        base_shape = tf.Tensor.get_shape(base_pred).as_list()[1:3]
        mrf_sz = mrf_weights.shape[0:2]

        base_pred = tf.nn.relu((base_pred + 1) / 2)
        slice_end = [0, 0]
        pad = False
        if mrf_sz[0] > base_shape[0]:
            dd1 = int(math.ceil(old_div(float(mrf_sz[0] - base_shape[0]), 2)))
            slice_end[0] = mrf_sz[0] - dd1
            pad = True
        else:
            dd1 = 0
            slice_end[0] = base_shape[0]

        if mrf_sz[1] > base_shape[1]:
            dd2 = int(math.ceil(old_div(float(mrf_sz[1] - base_shape[1]), 2)))
            slice_end[1] = mrf_sz[1] - dd2
            pad = True
        else:
            dd2 = 0
            slice_end[1] = base_shape[1]

        if pad:
            print('Padding base prediction by %d,%d. Filter shape:%d,%d Base shape:%d,%d' % (
            dd1, dd2, mrf_sz[0], mrf_sz[1], base_shape[0], base_shape[1]))
            base_pred = tf.pad(base_pred, [[0, 0], [dd1, dd1], [dd2, dd2], [0, 0]])

        base_shape_pad = tf.Tensor.get_shape(base_pred).as_list()[1:3]

        if hasattr(self.conf, 'add_loc_info') and self.conf.add_loc_info:
            # generate a constant image to represent distance from center
            x_mesh, y_mesh = np.meshgrid(np.arange(float(base_shape_pad[1])),
                                         np.arange(float(base_shape_pad[0])))
            x_mesh -= float(base_shape_pad[1]) / 2
            y_mesh -= float(base_shape_pad[0]) / 2
            loc_image = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
            loc_image = np.tile(loc_image[np.newaxis, ..., np.newaxis], [self.conf.batch_size, 1, 1, 1])
            self.ph['mrf_loc'] = tf.placeholder(tf.float32,
                                                [None, base_shape_pad[0], base_shape_pad[1], 1],
                                                name='x1')
            self.feed_dict[self.ph['mrf_loc']] = loc_image

        with tf.variable_scope('mrf'):
            # conv_std = old_div((old_div(1.,n_classes)),ksz)
            weights = tf.get_variable("weights", initializer=tf.constant(mrf_weights))
            biases = tf.get_variable("biases", n_classes,
                                     initializer=tf.constant_initializer(0))
            conv = tf.nn.conv2d(base_pred, weights,
                                strides=[1, 1, 1, 1], padding='SAME')
            mrf_out = conv + biases

            if hasattr(self.conf, 'add_loc_info') and self.conf.add_loc_info:
                weights = tf.get_variable("loc_weights", [4, 4, 1, self.conf.n_classes],
                                          initializer=tf.random_normal_initializer(stddev=0.001))
                conv = tf.nn.conv2d(self.ph['mrf_loc'], weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable("biases_loc", n_classes,
                                         initializer=tf.constant_initializer(0))
                t_mrf_out = tf.nn.relu(tf.concat([mrf_out, conv + biases], axis=3))
                weights = tf.get_variable("loc_weights_1", [1, 1, 2 * self.conf.n_classes, self.conf.n_classes],
                                          initializer=tf.random_normal_initializer(stddev=1.))
                conv = tf.nn.conv2d(t_mrf_out, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable("biases_loc_1", n_classes,
                                         initializer=tf.constant_initializer(0))
                mrf_out = conv + biases

        self.mrfPred = mrf_out[:, dd1:slice_end[0], dd2:slice_end[1], :]

    #   BELOW is from Chris bregler's paper where they try to do MRF style inference.
    #   This didn't work so well and the performance was always worse than base.
    #         mrf_conv = 0
    #         conv_out = []
    #         all_wts = []
    #         for cls in range(n_classes):
    #             with tf.variable_scope('mrf_%d'%cls):
    #                 curwt = 10*mrf_weights[:,:,cls:cls+1,:]-3
    #                 #the scaling is so that zero values are close to zero after softplus
    #                 weights = tf.get_variable("weights",dtype = tf.float32,
    #                               initializer=tf.constant(curwt))
    #                 biases = tf.get_variable("biases", mrf_weights.shape[-1],dtype = tf.float32,
    #                               initializer=tf.constant_initializer(-1))

    #             sweights = tf.nn.softplus(weights)
    #             sbiases = tf.nn.softplus(biases)

    #             if doBatch:
    #                 curBasePred = batch_norm(bpred[:,:,:,cls:cls+1],trainPhase)
    #                 curBasePred = tf.maximum(curBasePred,0.0001)
    #             else:
    #                 curBasePred = tf.maximum(bpred[:,:,:,cls:cls+1],0.0001)

    #             curconv = tf.nn.conv2d(curBasePred,sweights,strides=[1, 1, 1, 1],
    #                                    padding='SAME')+sbiases
    #             conv_out.append(tf.log(curconv))
    #             mrf_conv += tf.log(curconv)
    #             all_wts.append(sweights)
    #         mrfout = tf.exp(mrf_conv)
    #         self.mrfPred = mrfout[:,dd:sliceEnd,dd:sliceEnd,:]
    #    Return the value below when we used MRF kind of stuff
    #         return conv_out,all_wts
    def extractPatches(self, layer, out, locs, conf, scale, out_scale):
        # extract patch from a layer for finer resolution
        hsz = conf.fine_sz // scale // 2
        padsz = tf.constant([[0, 0], [hsz, hsz], [hsz, hsz], [0, 0]])
        patchsz = tf.to_int32([conf.fine_sz // scale, conf.fine_sz // scale, -1])

        patches = []
        # maxloc = PoseTools.argmax2d(out) * outscale
        maxloc = tf.to_int32(tf.transpose(locs // conf.pool_scale // conf.rescale * out_scale, [2, 0, 1]))
        maxloc = tf.stack([maxloc[1, ...], maxloc[0, ...]], axis=0)  # stupid x, y thing
        padlayer = tf.pad(layer, padsz)  # pad layer so that we don't worry about boundaries
        for inum in range(conf.batch_size):
            curpatches = []
            for ndx in range(conf.n_classes):
                curloc = tf.concat([tf.squeeze(maxloc[:, inum, ndx]), [0]], 0)
                curpatches.append(tf.slice(padlayer[inum, :, :, :], curloc, patchsz))
            patches.append(tf.stack(curpatches))
        return tf.stack(patches)

    def createFineNetwork(self, doBatch, jointTraining=False):
        if self.conf.useMRF:
            if not jointTraining:
                self.pred_fine_in = tf.stop_gradient(self.mrfPred)
            else:
                self.pred_fine_in = self.mrfPred
        else:
            if not jointTraining:
                self.pred_fine_in = tf.stop_gradient(self.basePred)
            else:
                self.pred_fine_in = self.basePred

        pred = self.ph['fine_pred_in']
        locs = self.ph['fine_pred_locs_in']

        # Construct fine model
        labelT = PoseTools.create_fine_label_tensor(self.conf)
        layers = self.baseLayers

        if not jointTraining:
            layer1_1 = tf.stop_gradient(layers['base_dict_0']['conv1'])
            layer1_2 = tf.stop_gradient(layers['base_dict_0']['conv2'])
            layer2_1 = tf.stop_gradient(layers['base_dict_1']['conv1'])
            layer2_2 = tf.stop_gradient(layers['base_dict_1']['conv2'])
        else:
            layer1_1 = layers['base_dict_0']['conv1']
            layer1_2 = layers['base_dict_0']['conv2']
            layer2_1 = layers['base_dict_1']['conv1']
            layer2_2 = layers['base_dict_1']['conv2']

        curfine1_1 = self.extractPatches(layer1_1, pred, locs, self.conf, 1, 4)
        curfine1_2 = self.extractPatches(layer1_2, pred, locs, self.conf, 2, 2)
        curfine2_1 = self.extractPatches(layer2_1, pred, locs, self.conf, 2, 2)
        curfine2_2 = self.extractPatches(layer2_2, pred, locs, self.conf, 4, 1)
        curfine1_1u = tf.unstack(tf.transpose(curfine1_1, [1, 0, 2, 3, 4]))
        curfine1_2u = tf.unstack(tf.transpose(curfine1_2, [1, 0, 2, 3, 4]))
        curfine2_1u = tf.unstack(tf.transpose(curfine2_1, [1, 0, 2, 3, 4]))
        curfine2_2u = tf.unstack(tf.transpose(curfine2_2, [1, 0, 2, 3, 4]))
        finepred = CNB.fineOut(curfine1_1u, curfine1_2u, curfine2_1u, curfine2_2u,
                               self.conf, doBatch,
                               self.ph['phase_train_fine'])
        limgs = PoseTools.create_fine_label_images(self.ph['locs'],
                                                   locs, self.conf, labelT)
        self.finePred = finepred
        self.fine_labels = limgs

    # ---------------- SAVING/RESTORING ---------------------

    def createBaseSaver(self):
        self.basesaver = tf.train.Saver(var_list=PoseTools.get_vars('base'),
                                        max_to_keep=self.conf.maxckpt)

    def createMRFSaver(self):
        self.mrfsaver = tf.train.Saver(var_list=PoseTools.get_vars('mrf'),
                                       max_to_keep=self.conf.maxckpt)

    def createFineSaver(self):
        self.finesaver = tf.train.Saver(var_list=PoseTools.get_vars('fine'),
                                        max_to_keep=self.conf.maxckpt)

    def createJointSaver(self):
        vlist = PoseTools.get_vars('fine') + PoseTools.get_vars('mrf') + PoseTools.get_vars('base')
        self.jointsaver = tf.train.Saver(var_list=vlist,
                                         max_to_keep=self.conf.maxckpt)

    def loadBase(self, sess, iterNum):
        outfilename = os.path.join(self.conf.cachedir, self.conf.baseoutname)
        ckptfilename = '%s-%d' % (outfilename, iterNum)
        print('Loading base from %s' % ckptfilename)
        self.basesaver.restore(sess, ckptfilename)

    def restoreBase(self, sess, restore):
        outfilename = os.path.join(self.conf.cachedir, self.conf.baseoutname)
        outfilename = outfilename.replace('\\', '/')
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.basedataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                                    latest_filename=self.conf.baseckptname)
        if not latest_ckpt or not restore:
            self.basestartat = 0
            self.basetrainData = {'train_err': [], 'val_err': [], 'step_no': [],
                                  'train_dist': [], 'val_dist': []}
            sess.run(tf.variables_initializer(PoseTools.get_vars('base')), feed_dict=self.feed_dict)
            print("Not loading base variables. Initializing them")
            return False
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
            return True

    def restoreMRF(self, sess, restore):
        outfilename = os.path.join(self.conf.cachedir, self.conf.mrfoutname)
        outfilename = outfilename.replace('\\', '/')
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.mrfdataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                                    latest_filename=self.conf.mrfckptname)
        if not latest_ckpt or not restore:
            self.mrfstartat = 0
            self.mrftrainData = {'train_err': [], 'val_err': [], 'step_no': [],
                                 'train_base_err': [], 'val_base_err': [],
                                 'train_dist': [], 'val_dist': [],
                                 'train_base_dist': [], 'val_base_dist': []}
            sess.run(tf.variables_initializer(PoseTools.get_vars('mrf')))
            print("Not loading mrf variables. Initializing them")
            return False
        else:
            self.mrfsaver.restore(sess, latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)', latest_ckpt.model_checkpoint_path)
            self.mrfstartat = int(matchObj.group(1)) + 1
            with open(traindatafilename, 'rb') as tdfile:
                if sys.version_info.major == 3:
                    inData = pickle.load(tdfile, encoding='latin1')
                else:
                    inData = pickle.load(tdfile)
                if not isinstance(inData, dict):
                    self.mrftrainData, loadconf = inData
                    print('Parameters that dont match for mrf:')
                    PoseTools.compare_conf(self.conf, loadconf)
                else:
                    print("No config was stored for mrf. Not comparing conf")
                    self.mrftrainData = inData
            print("Loading mrf variables from %s" % latest_ckpt.model_checkpoint_path)
            return True

    def restoreFine(self, sess, restore):
        outfilename = os.path.join(self.conf.cachedir, self.conf.fineoutname)
        outfilename = outfilename.replace('\\', '/')
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.finedataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                                    latest_filename=self.conf.fineckptname)
        if not latest_ckpt or not restore:
            self.finestartat = 0
            self.finetrainData = {'train_err': [], 'val_err': [], 'step_no': [],
                                  'train_mrf_err': [], 'val_mrf_err': [],
                                  'train_base_err': [], 'val_base_err': [],
                                  'train_dist': [], 'val_dist': [],
                                  'train_mrf_dist': [], 'val_mrf_dist': [],
                                  'train_base_dist': [], 'val_base_dist': []}
            sess.run(tf.variables_initializer(PoseTools.get_vars('fine')))
            print("Not loading fine variables. Initializing them")
            return False
        else:
            self.finesaver.restore(sess, latest_ckpt.model_checkpoint_path)
            mcp = latest_ckpt.model_checkpoint_path
            mcp = mcp.replace('\\', '/')

            matchObj = re.match(outfilename + '-(\d*)', mcp)
            self.finestartat = int(matchObj.group(1)) + 1
            with open(traindatafilename, 'rb') as tdfile:
                try:
                    if sys.version_info.major == 3:
                        inData = pickle.load(tdfile, encoding='latin1')
                    else:
                        inData = pickle.load(tdfile)
                    if not isinstance(inData, dict):
                        self.finetrainData, loadconf = inData
                        print('Parameters that dont match for fine:')
                        PoseTools.compare_conf(self.conf, loadconf)
                    else:
                        print("No conf was stored for fine. Not comparing conf")
                        self.finetrainData = inData
                except ValueError:
                    self.finetrainData = {'train_err': [], 'val_err': [], 'step_no': [],
                                          'train_mrf_err': [], 'val_mrf_err': [],
                                          'train_base_err': [], 'val_base_err': [],
                                          'train_dist': [], 'val_dist': [],
                                          'train_mrf_dist': [], 'val_mrf_dist': [],
                                          'train_base_dist': [], 'val_base_dist': []}

            print("Loading fine variables from %s" % latest_ckpt.model_checkpoint_path)
            return True

    def restoreJoint(self, sess, restore):
        outfilename = os.path.join(self.conf.cachedir, self.conf.jointoutname)
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.jointdataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                                    latest_filename=self.conf.jointckptname)
        if not latest_ckpt or not restore:
            self.finestartat = 0
            self.jointtrainData = {'train_err': [], 'val_err': [], 'step_no': [],
                                   'train_fine_err': [], 'val_fine_err': [],
                                   'train_mrf_err': [], 'val_mrf_err': [],
                                   'train_base_err': [], 'val_base_err': [],
                                   'train_dist': [], 'val_dist': [],
                                   'train_fine_dist': [], 'val_fine_dist': [],
                                   'train_mrf_dist': [], 'val_mrf_dist': [],
                                   'train_base_dist': [], 'val_base_dist': []}
            print("Not loading joint variables. Initializing them")
            return False
        else:
            self.jointsaver.restore(sess, latest_ckpt.model_checkpoint_path)
            print("Loading joint variables from %s" % latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)', latest_ckpt.model_checkpoint_path)
            self.jointstartat = int(matchObj.group(1)) + 1
            with open(traindatafilename, 'rb') as tdfile:
                inData = pickle.load(tdfile)
                if not isinstance(inData, dict):
                    self.jointtrainData, loadconf = inData
                    print('Parameters that dont match for joint:')
                    PoseTools.compare_conf(self.conf, loadconf)
                else:
                    print("No conf was stored for joint. Not comparing conf")
                    self.jointtrainData = inData
            return True

    def saveBase(self, sess, step):
        outfilename = os.path.join(self.conf.cachedir, self.conf.baseoutname)
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.basedataname)
        self.basesaver.save(sess, outfilename, global_step=step,
                            latest_filename=self.conf.baseckptname)
        print('Saved state to %s-%d' % (outfilename, step))
        with open(traindatafilename, 'wb') as tdfile:
            pickle.dump([self.basetrainData, self.conf], tdfile, protocol=2)

    def saveMRF(self, sess, step):
        outfilename = os.path.join(self.conf.cachedir, self.conf.mrfoutname)
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.mrfdataname)
        self.mrfsaver.save(sess, outfilename, global_step=step,
                           latest_filename=self.conf.mrfckptname)
        print('Saved state to %s-%d' % (outfilename, step))
        with open(traindatafilename, 'wb') as tdfile:
            pickle.dump([self.mrftrainData, self.conf], tdfile, protocol=2)

    def saveFine(self, sess, step):
        outfilename = os.path.join(self.conf.cachedir, self.conf.fineoutname)
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.finedataname)
        self.finesaver.save(sess, outfilename, global_step=step,
                            latest_filename=self.conf.fineckptname)
        print('Saved state to %s-%d' % (outfilename, step))
        with open(traindatafilename, 'wb') as tdfile:
            pickle.dump([self.finetrainData, self.conf], tdfile, protocol=2)

    def saveJoint(self, sess, step):
        outfilename = os.path.join(self.conf.cachedir, self.conf.jointoutname)
        traindatafilename = os.path.join(self.conf.cachedir, self.conf.jointdataname)
        self.finesaver.save(sess, outfilename, global_step=step,
                            latest_filename=self.conf.jointckptname)
        print('Saved state to %s-%d' % (outfilename, step))
        with open(traindatafilename, 'wb') as tdfile:
            pickle.dump([self.jointtrainData, self.conf], tdfile, protocol=2)

    def initializeRemainingVars(self, sess):
        # var_list = tf.report_uninitialized_variables()
        # sess.run(tf.variables_initializer(var_list))
        # at some stage do faster initialization.
        varlist = tf.global_variables()
        for var in varlist:
            try:
                sess.run(tf.assert_variables_initialized([var]))
            except tf.errors.FailedPreconditionError:
                sess.run(tf.variables_initializer([var]))
                print('Initializing variable:%s' % var.name)

    # ---------------- OPTIMIZATION/LOSS ---------------------



    def createOptimizer(self):
        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.ph['learning_rate']).minimize(self.cost)
        self.read_time = 0.
        self.opt_time = 0.

    def doOpt(self, sess):
        # self.feed_dict[self.ph['keep_prob']] = self.conf.dropout
        r_start = time.clock()
        self.updateFeedDict(self.DBType.Train, distort=True, sess=sess)
        r_end = time.clock()
        sess.run(self.opt, self.feed_dict)
        o_end = time.clock()

        self.read_time += r_end - r_start
        self.opt_time += o_end - r_end

    def doOptFine(self, sess):
        r_start = time.clock()
        self.updateFeedDictFine(self.DBType.Train, distort=True, sess=sess)
        r_end = time.clock()
        sess.run(self.opt, self.feed_dict)
        o_end = time.clock()

        self.read_time += r_end - r_start
        self.opt_time += o_end - r_end

    def computeLoss(self, sess, costfcns):
        self.feed_dict[self.ph['keep_prob']] = 1.
        loss = sess.run(costfcns, self.feed_dict)
        loss = [old_div(x, self.conf.batch_size) for x in loss]
        return loss

    def computePredDist(self, sess, predfcn):
        self.feed_dict[self.ph['keep_prob']] = 1.
        pred = sess.run(predfcn, self.feed_dict)
        bee = PoseTools.get_base_error(self.locs, pred, self.conf)
        dee = np.sqrt(np.sum(np.square(bee), 2))
        return dee

    def computeFinePredDist(self, sess, predfcn):
        self.feed_dict[self.ph['keep_prob']] = 1.
        pred = sess.run(predfcn, self.feed_dict)
        base_ee, fine_ee = PoseTools.get_fine_error(self.locs, pred[0], pred[1], self.conf)
        fine_dist = np.sqrt(np.sum(np.square(fine_ee), 2))
        return fine_dist

    def updateBaseLoss(self, step, train_loss, val_loss, trainDist, valDist):
        print("Iter " + str(step) + ", Train = " + "{:.3f},{:.1f}".format(train_loss[0], trainDist[
            0]) + ", Val = " + "{:.3f},{:.1f}".format(val_loss[0], valDist[0]))
        #         nstep = step-self.basestartat
        #         print "  Read Time:" + "{:.2f}, ".format(self.read_time/(nstep+1)) + \
        #               "Opt Time:" + "{:.2f}".format(self.opt_time/(nstep+1))
        self.basetrainData['train_err'].append(train_loss[0])
        self.basetrainData['val_err'].append(val_loss[0])
        self.basetrainData['step_no'].append(step)
        self.basetrainData['train_dist'].append(trainDist[0])
        self.basetrainData['val_dist'].append(valDist[0])

    def updateMRFLoss(self, step, train_loss, val_loss, trainDist, valDist):
        print("Iter " + str(step) + ", Train = " + "{:.3f},{:.1f}".format(train_loss[0], trainDist[
            0]) + ", Val = " + "{:.3f},{:.1f}".format(val_loss[0],
                                                      valDist[0]) + " ({:.1f},{:.1f}),({:.1f},{:.1f})".format(
            train_loss[1], val_loss[1],
            trainDist[1], valDist[1]))
        self.mrftrainData['train_err'].append(train_loss[0])
        self.mrftrainData['val_err'].append(val_loss[0])
        self.mrftrainData['train_base_err'].append(train_loss[1])
        self.mrftrainData['val_base_err'].append(val_loss[1])
        self.mrftrainData['train_dist'].append(trainDist[0])
        self.mrftrainData['val_dist'].append(valDist[0])
        self.mrftrainData['train_base_dist'].append(trainDist[1])
        self.mrftrainData['val_base_dist'].append(valDist[1])
        self.mrftrainData['step_no'].append(step)

    def updateFineLoss(self, step, train_loss, val_loss, trainDist, valDist):
        print("Iter " + str(step) + \
              ", Train = " + "{:.3f},{:.1f}".format(train_loss[0], trainDist[0]) + \
              ", Val = " + "{:.3f},{:.1f}".format(val_loss[0], valDist[0]) + \
              " (MRF:{:.1f},{:.1f},{:.1f},{:.1f})".format(train_loss[1], val_loss[1], trainDist[1], valDist[1]) + \
              " (Base:{:.1f},{:.1f},{:.1f},{:.1f})".format(train_loss[2], val_loss[2], trainDist[2], valDist[2]))
        self.finetrainData['train_err'].append(train_loss[0])
        self.finetrainData['val_err'].append(val_loss[0])
        self.finetrainData['train_mrf_err'].append(train_loss[1])
        self.finetrainData['val_mrf_err'].append(val_loss[1])
        self.finetrainData['train_base_err'].append(train_loss[2])
        self.finetrainData['val_base_err'].append(val_loss[2])
        self.finetrainData['step_no'].append(step)
        self.finetrainData['train_dist'].append(trainDist[0])
        self.finetrainData['val_dist'].append(valDist[0])
        self.finetrainData['train_mrf_dist'].append(trainDist[1])
        self.finetrainData['val_mrf_dist'].append(valDist[1])
        self.finetrainData['train_base_dist'].append(trainDist[2])
        self.finetrainData['val_base_dist'].append(valDist[2])

    def updateJointLoss(self, step, train_loss, val_loss):
        print("Iter " + str(step) + ", Train = " + "{:.3f}".format(train_loss[0]) + ", Val = " + "{:.3f}".format(
            val_loss[0]) + " (Fine:{:.1f},{:.1f})".format(train_loss[1], val_loss[1]) + " (MRF:{:.1f},{:.1f})".format(
            train_loss[2], val_loss[2]) + " (Base:{:.1f},{:.1f})".format(train_loss[3], val_loss[3]))
        self.jointtrainData['train_err'].append(train_loss)
        self.jointftrainData['val_err'].append(val_loss[0])
        self.jointtrainData['train_fine_err'].append(train_loss[1])
        self.jointtrainData['val_fine_err'].append(val_loss[1])
        self.jointtrainData['train_mrf_err'].append(train_loss[2])
        self.jointtrainData['val_mrf_err'].append(val_loss[2])
        self.jointtrainData['train_base_err'].append(train_loss[3])
        self.jointtrainData['val_base_err'].append(val_loss[3])
        self.jointtrainData['step_no'].append(step)



        # ---------------- TRAINING ---------------------

    def baseTrain(self, restore=True, trainPhase=True, trainType=0):
        base_dropout = 1.

        self.createPH()
        self.createFeedDict()
        self.feed_dict[self.ph['phase_train_base']] = trainPhase
        self.feed_dict[self.ph['keep_prob']] = base_dropout
        self.trainType = trainType
        self.doBatchNorm = self.conf.doBatchNorm

        with tf.variable_scope('base'):
            self.createBaseNetwork(self.doBatchNorm)
        self.cost = tf.nn.l2_loss(self.basePred - self.ph['y'])
        self.openDBs()
        self.createOptimizer()
        self.createBaseSaver()

        #         with self.env.begin() as txn,\
        #                  self.valenv.begin() as valtxn,\
        with tf.Session() as sess:

            self.createCursors(sess)
            self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
            self.restoreBase(sess, restore)
            self.initializeRemainingVars(sess)

            if self.basestartat < self.conf.base_training_iters:

                for step in range(self.basestartat, self.conf.base_training_iters + 1):
                    ex_count = step * self.conf.batch_size
                    cur_lr = self.conf.base_learning_rate * \
                             self.conf.gamma ** math.floor(old_div(ex_count, self.conf.step_size))
                    self.feed_dict[self.ph['learning_rate']] = cur_lr
                    self.feed_dict[self.ph['keep_prob']] = base_dropout
                    self.doOpt(sess)
                    if step % self.conf.display_step == 0:
                        self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
                        self.feed_dict[self.ph['keep_prob']] = 1.
                        train_loss = self.computeLoss(sess, [self.cost])
                        tt1 = self.computePredDist(sess, self.basePred)
                        nantt1 = np.invert(np.isnan(tt1.flatten()))
                        nantt1_mean = tt1.flatten()[nantt1].mean()
                        trainDist = [nantt1_mean]
                        numrep = int(old_div(self.conf.numTest, self.conf.batch_size)) + 1
                        val_loss = np.zeros([2, ])
                        valDist = [0.]
                        for _ in range(numrep):
                            self.updateFeedDict(self.DBType.Val, distort=False, sess=sess)
                            val_loss += np.array(self.computeLoss(sess, [self.cost]))
                            tt1 = self.computePredDist(sess, self.basePred)
                            nantt1 = np.invert(np.isnan(tt1.flatten()))
                            nantt1_mean = tt1.flatten()[nantt1].mean()
                            valDist += nantt1_mean
                        val_loss = old_div(val_loss, numrep)
                        valDist = [old_div(valDist[0], numrep)]
                        self.updateBaseLoss(step, train_loss, val_loss, trainDist, valDist)
                    if step % self.conf.save_step == 0:
                        self.saveBase(sess, step)
                print("Optimization Finished!")
                self.saveBase(sess, self.conf.base_training_iters)
            self.closeCursors()

    def mrfTrain(self, restore=True, trainType=0):
        mrf_dropout = 0.1
        self.createPH()
        self.createFeedDict()
        doBatchNorm = self.conf.doBatchNorm
        self.feed_dict[self.ph['keep_prob']] = mrf_dropout
        self.feed_dict[self.ph['phase_train_base']] = False
        self.trainType = trainType

        with tf.variable_scope('base'):
            self.createBaseNetwork(doBatchNorm)

        with tf.variable_scope('mrf'):
            self.createMRFNetwork(doBatchNorm)

        self.createBaseSaver()
        self.createMRFSaver()

        # mod_labels = tf.maximum((self.ph['y'] + 1.) / 2, 0.01)
        # the labels shouldn't be zero because the prediction is an output of
        # exp. And it seems a lot of effort is wasted to make the prediction goto
        # zero rather than match the location.

        #         self.cost = tf.nn.l2_loss(self.mrfPred-mod_labels)
        self.cost = tf.nn.l2_loss(2 * (self.mrfPred - 0.5) - self.ph['y'])
        basecost = tf.nn.l2_loss(self.basePred - self.ph['y'])

        if self.conf.useHoldout:
            self.openHoldoutDBs()
        else:
            self.openDBs()

        self.createOptimizer()

        #         self.env.begin() as txn,self.valenv.begin() as valtxn,
        with tf.Session() as sess:

            self.createCursors(sess)
            self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
            self.restoreBase(sess, restore=True)
            self.restoreMRF(sess, restore)
            self.initializeRemainingVars(sess)

            for step in range(self.mrfstartat, self.conf.mrf_training_iters + 1):
                ex_count = step * self.conf.batch_size
                cur_lr = self.conf.mrf_learning_rate * \
                         self.conf.gamma ** math.floor(old_div(ex_count, self.conf.step_size))
                self.feed_dict[self.ph['learning_rate']] = cur_lr
                self.feed_dict[self.ph['keep_prob']] = mrf_dropout
                self.doOpt(sess)
                if step % self.conf.display_step == 0:
                    self.feed_dict[self.ph['keep_prob']] = 1.0
                    self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
                    train_loss = self.computeLoss(sess, [self.cost, basecost])
                    tt1 = self.computePredDist(sess, self.mrfPred)
                    tt2 = self.computePredDist(sess, self.basePred)
                    trainDist = [tt1.mean(), tt2.mean()]

                    numrep = int(old_div(self.conf.numTest, self.conf.batch_size)) + 1
                    val_loss = np.zeros([2, ])
                    valDist = [0., 0.]
                    for _ in range(numrep):
                        self.updateFeedDict(self.DBType.Val, sess=sess, distort=False)
                        val_loss += np.array(self.computeLoss(sess, [self.cost, basecost]))
                        tt1 = self.computePredDist(sess, self.mrfPred)
                        tt2 = self.computePredDist(sess, self.basePred)
                        valDist = [valDist[0] + tt1.mean(), valDist[1] + tt2.mean()]

                    val_loss = old_div(val_loss, numrep)
                    valDist = [old_div(valDist[0], numrep), old_div(valDist[1], numrep)]
                    self.updateMRFLoss(step, train_loss, val_loss, trainDist, valDist)
                if step % self.conf.save_step == 0:
                    self.saveMRF(sess, step)
            print("Optimization Finished!")
            self.saveMRF(sess, self.conf.mrf_training_iters)
            self.closeCursors()

    def fineTrain(self, restore=True, trainPhase=True, trainType=0):
        self.createPH()
        self.createFeedDict()
        self.trainType = trainType
        self.openDBs()
        fine_dropout = 1.
        self.feed_dict[self.ph['keep_prob']] = fine_dropout
        self.feed_dict[self.ph['phase_train_fine']] = trainPhase
        self.feed_dict[self.ph['phase_train_base']] = False
        doBatchNorm = self.conf.doBatchNorm
        self.trainType = trainType

        with tf.variable_scope('base'):
            self.createBaseNetwork(doBatchNorm)
        if self.conf.useMRF:
            with tf.variable_scope('mrf'):
                self.createMRFNetwork(doBatchNorm)
        with tf.variable_scope('fine'):
            self.createFineNetwork(doBatchNorm)

        self.createBaseSaver()
        if self.conf.useMRF:
            self.createMRFSaver()
        self.createFineSaver()

        mod_labels = tf.maximum(old_div((self.ph['y'] + 1.), 2), 0.01)
        # the labels shouldn't be zero because the prediction is an output of
        # exp. And it seems a lot of effort is wasted to make the prediction goto
        # zero rather than match the location.

        self.cost = tf.nn.l2_loss(self.finePred - tf.to_float(self.fine_labels))
        basecost = tf.nn.l2_loss(self.basePred - self.ph['y'])
        if self.conf.useMRF:
            mrfcost = tf.nn.l2_loss(self.mrfPred - mod_labels)
        else:
            mrfcost = basecost
        self.createOptimizer()
        if self.conf.useMRF:
            predPair = [self.mrfPred, self.finePred]
        else:
            predPair = [self.basePred, self.finePred]

        with tf.Session() as sess:

            self.restoreBase(sess, True)
            if self.conf.useMRF:
                self.restoreMRF(sess, True)
            self.restoreFine(sess, restore)
            self.initializeRemainingVars(sess)
            self.createCursors(sess)

            for step in range(self.finestartat, self.conf.fine_training_iters + 1):
                ex_count = step * self.conf.batch_size
                cur_lr = self.conf.fine_learning_rate * \
                         self.conf.gamma ** math.floor(old_div(ex_count, self.conf.step_size))
                self.feed_dict[self.ph['learning_rate']] = cur_lr
                self.feed_dict[self.ph['keep_prob']] = fine_dropout
                self.doOptFine(sess)
                if step % self.conf.display_step == 0:
                    self.feed_dict[self.ph['keep_prob']] = 1.
                    self.updateFeedDictFine(self.DBType.Train, distort=True, sess=sess)
                    train_loss = self.computeLoss(sess, [self.cost, mrfcost, basecost])
                    tt1 = self.computePredDist(sess, self.basePred)
                    if self.conf.useMRF:
                        tt2 = self.computePredDist(sess, self.mrfPred)
                    else:
                        tt2 = tt1
                    tt3 = self.computeFinePredDist(sess, predPair)

                    trainDist = [tt3.mean(), tt2.mean(), tt1.mean()]

                    numrep = int(old_div(self.conf.numTest, self.conf.batch_size)) + 1
                    val_loss = np.zeros([3, ])
                    valDist = [0., 0., 0.]
                    for _ in range(numrep):
                        self.updateFeedDictFine(self.DBType.Val, distort=False, sess=sess)
                        val_loss += np.array(self.computeLoss(sess, [self.cost, mrfcost, basecost]))
                        tt1 = self.computePredDist(sess, self.basePred)
                        if self.conf.useMRF:
                            tt2 = self.computePredDist(sess, self.mrfPred)
                        else:
                            tt2 = tt1
                        tt3 = self.computeFinePredDist(sess, predPair)
                        valDist = [valDist[0] + tt3.mean(),
                                   valDist[1] + tt2.mean(),
                                   valDist[2] + tt1.mean()]

                    val_loss = old_div(val_loss, numrep)
                    valDist = [old_div(valDist[0], numrep), old_div(valDist[1], numrep), old_div(valDist[2], numrep)]
                    self.updateFineLoss(step, train_loss, val_loss, trainDist, valDist)
                if step % self.conf.save_step == 0:
                    self.saveFine(sess, step)
            print("Optimization Finished!")
            self.saveFine(sess, self.conf.fine_training_iters)
            self.closeCursors()

    def jointTrain(self, restore=True):
        self.createPH()
        self.createFeedDict()
        self.openDBs()
        self.feed_dict[self.ph['phase_train_base']] = True
        self.feed_dict[self.ph['phase_train_fine']] = True
        doBatchNorm = self.conf.doBatchNorm

        with tf.variable_scope('base'):
            self.createBaseNetwork(doBatchNorm)
        with tf.variable_scope('mrf'):
            self.createMRFNetwork(doBatchNorm)
        with tf.variable_scope('fine'):
            self.createFineNetwork(doBatchNorm)

        self.createBaseSaver()
        self.createMRFSaver()
        self.createACSaver()
        self.createFineSaver()
        self.createJointSaver()

        mod_labels = tf.maximum(old_div((self.ph['y'] + 1.), 2), 0.01)
        # the labels shouldn't be zero because the prediction is an output of
        # exp. And it seems a lot of effort is wasted to make the prediction goto
        # zero rather than match the location.

        finecost = tf.nn.l2_loss(self.finePred - self.fine_labels)
        mrfcost = tf.nn.l2_loss(self.mrfPred - mod_labels)
        basecost = tf.nn.l2_loss(self.basePred - self.ph['y'])
        self.cost = finecost + self.conf.joint_MRFweight * mrfcost

        self.createOptimizer()

        with tf.Session() as sess:

            self.restoreBase(sess, True)
            self.restoreMRF(sess, True)
            self.restoreFine(sess, True)
            self.restoreJoint(sess, restore)
            self.initializeRemainingVars(sess)
            self.createCursors(sess)

            for step in range(self.jointstartat, self.conf.joint_training_iters + 1):
                self.doOpt(sess)
                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train,distort=True,sess= sess)
                    train_loss = self.computeLoss(sess, [self.cost, finecost, mrfcost, basecost])
                    numrep = int(old_div(self.conf.numTest, self.conf.batch_size)) + 1
                    val_loss = np.zeros([2, ])
                    for _ in range(numrep):
                        self.updateFeedDict(self.DBType.Val,distort=False, sess=sess)
                        val_loss += np.array(self.computeLoss(sess, [self.cost, finecost, mrfcost, basecost]))
                    val_loss = old_div(val_loss, numrep)
                    self.updateJointLoss(step, train_loss, val_loss)
                if step % self.conf.save_step == 0:
                    self.saveJoint(sess, step)
            print("Optimization Finished!")
            self.saveJoint(sess, self.conf.joint_training_iters )

    def classify_val_base(self):
        val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        num_val = 0
        for record in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        self.createPH()
        self.createFeedDict()
        self.feed_dict[self.ph['phase_train_base']] = False
        self.feed_dict[self.ph['keep_prob']] = 1.
        self.trainType = 0
        self.doBatchNorm = self.conf.doBatchNorm
        self.openDBs()

        with tf.variable_scope('base'):
            self.createBaseNetwork(self.doBatchNorm)
        self.createBaseSaver()

        val_dist = []
        val_ims = []
        val_preds = []
        val_predlocs = []
        val_locs = []
        with tf.Session() as sess:
            self.createCursors(sess)
            self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
            self.restoreBase(sess, True)
            self.initializeRemainingVars(sess)
            for step in range(num_val/self.conf.batch_size):
                self.updateFeedDict(self.DBType.Val, distort=False, sess=sess)
                cur_pred = sess.run(self.basePred, self.feed_dict)
                cur_predlocs = PoseTools.get_pred_locs(cur_pred)*self.conf.pool_scale*self.conf.rescale
                cur_dist = np.sqrt(np.sum(
                    (cur_predlocs-self.locs) ** 2, 2))
                val_dist.append(cur_dist)
                val_ims.append(self.xs)
                val_locs.append(self.locs)
                val_preds.append(cur_pred)
                val_predlocs.append(cur_predlocs)
            self.closeCursors()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape( (-1,) + in_a.shape[2:])
        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_predlocs = val_reshape(val_predlocs)
        val_locs = val_reshape(val_locs)

        return val_dist, val_ims, val_preds, val_predlocs, val_locs

