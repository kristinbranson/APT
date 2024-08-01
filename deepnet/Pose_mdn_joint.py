import PoseUNet_resnet
import PoseTools as pt

# import tensorflow
# vv = [int(v) for v in tensorflow.__version__.split('.')]
# if (vv[0]==1 and vv[1]>12) or vv[0]==2:
#     tf = tensorflow.compat.v1
# else:
#     tf = tensorflow
# # from batch_norm import batch_norm_mine_old as batch_norm
# if vv[0]==1:
#     from tensorflow.contrib.layers import batch_norm
#     import tensorflow.contrib.slim as slim
#     from tensorflow.contrib.slim.nets import resnet_v1
#     from tensorflow.contrib.layers import xavier_initializer
# else:
#     from tensorflow.compat.v1.layers import batch_normalization as batch_norm_temp
#     def batch_norm(inp,decay,is_training,renorm=False,data_format=None):
#         return batch_norm_temp(inp,momentum=decay,training=is_training)
#     import tf_slim as slim
#     from tf_slim.nets import resnet_v1
#     from tensorflow.keras.initializers import GlorotUniform as  xavier_initializer

import tensorflow
# Assume TensorFlow 2.x.x
tf = tensorflow.compat.v1
# from batch_norm import batch_norm_mine_old as batch_norm
#from tensorflow.compat.v1.layers import BatchNormalization as batch_norm_temp
batch_norm_temp = tensorflow.compat.v1.layers.BatchNormalization
def batch_norm(inp,decay,is_training,renorm=False,data_format=None):
    return batch_norm_temp(inp,momentum=decay,training=is_training)
import tf_slim as slim
from tf_slim.nets import resnet_v1
#from tensorflow.keras.initializers import GlorotUniform as  xavier_initializer
xavier_initializer = tensorflow.keras.initializers.GlorotUniform

from PoseCommon_dataset import conv_relu3, conv_relu
import resnet_official
import numpy as np
import contextlib
from upsamp import upsample_init_value


class Pose_mdn_joint_tf(PoseUNet_resnet.PoseUMDN_resnet):

    def set_shape(self):
        im, locs, info, hmap = self.inputs
        conf = self.conf
        in_sz = [int(sz//conf.rescale) for sz in conf.imsz]
        im.set_shape([conf.batch_size,
                      in_sz[0] + self.pad_y,
                      in_sz[1] + self.pad_x,
                      conf.img_dim])
        hmap.set_shape([conf.batch_size, in_sz[0], in_sz[1],conf.n_classes])
        locs.set_shape([conf.batch_size, conf.n_classes,2])
        info.set_shape([conf.batch_size,3])

    def create_network(self):
        self.set_shape()
        im, locs, info, hmap = self.inputs
        conf = self.conf
        in_sz = [int(sz//conf.rescale) for sz in conf.imsz]

        if conf.img_dim == 1:
            im = tf.tile(im,[1,1,1,3])

        if self.conf.get('pretrain_freeze_bnorm', True):
            pretrain_update_bnorm = False
        else:
            pretrain_update_bnorm = self.ph['phase_train']

        if self.resnet_source == 'slim':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                output_stride =  self.conf.get('mdn_slim_output_stride',None)
                net, end_points = resnet_v1.resnet_v1_50(im,global_pool=False, is_training=pretrain_update_bnorm,output_stride=output_stride)

        elif self.resnet_source == 'official_tf':
            mm = resnet_official.Model(resnet_size=50, bottleneck=True, num_classes=17, num_filters=64, kernel_size=7,
                                       conv_stride=2, first_pool_size=3, first_pool_stride=2, block_sizes=[3, 4, 6, 3],
                                       block_strides=[1, 2, 2, 2], final_size=2048, resnet_version=2,
                                       data_format='channels_last', dtype=tf.float32)
            resnet_out = mm(im, pretrain_update_bnorm)
            down_layers = mm.layers
            # down_layers.pop(2) # remove one of the layers of size imsz/4, imsz/4 at index 2
            net = down_layers[-1]


        X_joint = net
        if self.conf.get('mdn_joint_use_wasp',False) is True:
            n_wasp = 256
            n_dil = self.conf.get('mdn_joint_wasp_dilation',2)

            w_layers = []
            w_in = X_joint
            for wn in range(4):
                curX = tf.layers.Conv2D(n_wasp,3,padding='same',dilation_rate=n_dil*(wn+1))(w_in)
                curX = batch_norm(curX,decay=0.99, is_training=self.ph['phase_train'])
                curX = tf.nn.relu(curX)
                curX = tf.layers.conv2d(curX,n_wasp,1,padding='same')
                curX = batch_norm(curX,decay=0.99, is_training=self.ph['phase_train'])
                curX = tf.nn.relu(curX)
                curX = tf.layers.conv2d(curX,n_wasp,1,padding='same')
                curX = batch_norm(curX,decay=0.99, is_training=self.ph['phase_train'])
                curX = tf.nn.relu(curX)
                w_layers.append(curX)
                w_in = curX

            if self.conf.get('mdn_joint_wasp_skip',False) is True:
                # Do 1x1 conv on resnets output to get the channels down to 256
                curX = tf.layers.conv2d(X_joint, n_wasp, 1, padding='same')
                curX = batch_norm(curX, decay=0.99, is_training=self.ph['phase_train'])
                curX = tf.nn.relu(curX)
                curX = tf.layers.conv2d(curX, n_wasp, 1, padding='same')
                curX = batch_norm(curX, decay=0.99, is_training=self.ph['phase_train'])
                curX = tf.nn.relu(curX)
                w_layers.append(curX)

                X_joint = tf.concat(w_layers,3)
                curX = tf.layers.conv2d(X_joint, 512, 1, padding='same')
                curX = batch_norm(curX, decay=0.99, is_training=self.ph['phase_train'])
                curX = tf.nn.relu(curX)
                curX = tf.layers.conv2d(curX, 512, 1, padding='same')
                curX = batch_norm(curX, decay=0.99, is_training=self.ph['phase_train'])
                X_joint = tf.nn.relu(curX)
            else:
                X_wasp = tf.concat(w_layers, 3)
                n_joint_in = X_joint.get_shape().as_list()[3]
                curX = tf.layers.conv2d(X_wasp, n_joint_in, 3, padding='same')
                curX = batch_norm(curX, decay=0.99, is_training=self.ph['phase_train'])
                X_joint = X_joint+curX
                X_joint = tf.nn.relu(X_joint)

        if self.conf.get('mdn_joint_use_fpn',False) is False:
            X_ref = net
            self.ref_scale = 1
        else:
            fpn_layers = down_layers[-4:]
            fpn_channels = fpn_layers[0].get_shape().as_list()[3]
            cur_ch = fpn_layers[-1].get_shape().as_list()[3]
            prev_in = conv_relu(fpn_layers[-1],[3,3,cur_ch,fpn_channels],self.ph['phase_train'])
            for ndx in reversed(range(len(fpn_layers)-1)):
                with tf.variable_scope('argh_scopes_{}'.format(ndx)):
                    cur_ch = fpn_layers[ndx].get_shape().as_list()[3]
                    lateral = conv_relu(fpn_layers[ndx],[3,3,cur_ch,fpn_channels],self.ph['phase_train'])
                    layers_sz = lateral.get_shape().as_list()[1:]

                with tf.variable_scope('argh_scopes_1_{}'.format(ndx)):
                    X_sh = prev_in.get_shape().as_list()
                    w_sh = [4, 4, X_sh[-1], X_sh[-1]]
                    w_init = upsample_init_value(w_sh, alg='bl', dtype=np.float32)
                    w = tf.get_variable('w', w_sh, initializer=tf.constant_initializer(w_init))
                    out_shape = [X_sh[0], layers_sz[0], layers_sz[1], X_sh[-1]]
                    prev_in = tf.nn.conv2d_transpose(prev_in, w, output_shape=out_shape, strides=[1, 2, 2, 1], padding="SAME")
                    biases = tf.get_variable('biases', [out_shape[-1]], initializer=tf.constant_initializer(0))
                    conv_b = prev_in + biases
                    bn = batch_norm(conv_b,0.99,is_training=self.ph['phase_train'])
                    prev_in = tf.nn.relu(bn) + lateral
            X_ref = prev_in
            self.ref_scale = 2**(len(fpn_layers)-1)

        k_joint = 1
        locs_offset = 1.
        n_out = self.conf.n_classes
        self.resnet_out = net
        k_ref = 5

        with tf.variable_scope('mdn_joint'):
            if self.conf.get('mdn_regularize_wt',False) is True:
                wt_scale = self.conf.get('mdn_regularize_wt_scale',0.1)
                wt_reg = tensorflow.contrib.layers.l2_regularizer(scale=wt_scale)
            else:
                wt_reg = None

            n_filt_in = X_joint.get_shape().as_list()[3]
            n_filt = 512
            k_sz = 3

            with tf.variable_scope('locs_joint'):
                with tf.variable_scope('layer_locs'):
                    kernel_shape = [k_sz, k_sz, n_filt_in, 3*n_filt]
                    mdn_l = conv_relu(X_joint,kernel_shape,self.ph['phase_train'])
                with tf.variable_scope('layer_locs_1_1'):
                    in_filt = mdn_l.get_shape().as_list()[3]
                    kernel_shape = [k_sz, k_sz, in_filt, n_filt]
                    mdn_l = conv_relu(mdn_l,kernel_shape, self.ph['phase_train'])

                loc_shape = mdn_l.get_shape().as_list()
                n_x = loc_shape[2]
                n_y = loc_shape[1]
                self.n_x_j = n_x
                self.n_y_j = n_y
                x_off, y_off = np.meshgrid(np.arange(n_x), np.arange(n_y))
                # no need for tiling because of broadcasting
                x_off = x_off[np.newaxis,:,:,np.newaxis]
                y_off = y_off[np.newaxis,:,:,np.newaxis]

                in_filt = loc_shape[-1]

                weights_locs = tf.get_variable("weights_locs", [1, 1, in_filt, 2 * n_out], initializer=xavier_initializer(),regularizer=wt_reg)
                biases_locs = tf.get_variable("biases_locs", 2 *  n_out, initializer=tf.constant_initializer(0))
                o_locs = tf.nn.conv2d(mdn_l, weights_locs, [1, 1, 1, 1], padding='SAME') + biases_locs
                o_locs = tf.reshape(o_locs,[-1,n_y,n_x,n_out,2])
                self.i_locs = o_locs
                x_locs = o_locs[:, :, :, :, 0] + x_off
                y_locs = o_locs[:, :, :, :, 1] + y_off
                o_locs = tf.stack([x_locs, y_locs], axis=-1)
                locs_joint = o_locs


            with tf.variable_scope('locs'):
                n_filt_in = X_ref.get_shape().as_list()[3]
                n_filt = 512
                k_sz = 3

                with tf.variable_scope('layer_locs'):
                    kernel_shape = [k_sz, k_sz, n_filt_in, 3*n_filt]
                    mdn_l = conv_relu(X_ref,kernel_shape,self.ph['phase_train'])
                with tf.variable_scope('layer_locs_1_1'):
                    in_filt = mdn_l.get_shape().as_list()[3]
                    kernel_shape = [k_sz, k_sz, in_filt, n_filt]
                    mdn_l = conv_relu(mdn_l,kernel_shape, self.ph['phase_train'])

                loc_shape = mdn_l.get_shape().as_list()
                n_x = loc_shape[2]
                n_y = loc_shape[1]
                self.n_x_r = n_x
                self.n_y_r = n_y

                in_filt = loc_shape[-1]

                weights_locs = tf.get_variable("weights_locs", [1, 1, in_filt, 2 * k_ref * n_out],                                              initializer=xavier_initializer(),regularizer=wt_reg)
                biases_locs = tf.get_variable("biases_locs", 2 * k_ref* n_out, initializer=tf.constant_initializer(0))
                o_locs = tf.nn.conv2d(mdn_l, weights_locs, [1, 1, 1, 1], padding='SAME') + biases_locs

                o_locs = tf.reshape(o_locs,[-1,n_y,n_x,k_ref, n_out,2])
                self.i_locs = o_locs
                o_locs *= float(locs_offset)
                o_locs += float(locs_offset)/2

                x_off, y_off = np.meshgrid(np.arange(n_x), np.arange(n_y))
                # no need for tiling because of broadcasting
                x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
                x_locs = o_locs[:, :, :, :, :, 0] + x_off*locs_offset
                y_locs = o_locs[:, :, :, :, :, 1] + y_off*locs_offset
                o_locs = tf.stack([x_locs, y_locs], axis=-1)
                locs = o_locs


            with tf.variable_scope('logits_joint'):
                n_filt_in = X_joint.get_shape().as_list()[3]
                n_filt = 512
                k_sz = 3

                with tf.variable_scope('layer_logits'):
                    kernel_shape = [1, 1, n_filt_in, n_filt]
                    weights = tf.get_variable("weights", kernel_shape, initializer=xavier_initializer(),regularizer=wt_reg)
                    biases = tf.get_variable("biases", kernel_shape[-1], initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(X_joint, weights,strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99, is_training=self.ph['phase_train'])
                mdn_l = tf.nn.relu(conv + biases)

                loc_shape = mdn_l.get_shape().as_list()
                in_filt = loc_shape[-1]
                weights_logits = tf.get_variable("weights_logits", [1, 1, in_filt,1], initializer=xavier_initializer())
                biases_logits = tf.get_variable("biases_logits", k_joint * 1, initializer=tf.constant_initializer(0))
                logits_joint = tf.nn.conv2d(mdn_l, weights_logits, [1, 1, 1, 1], padding='SAME') + biases_logits

            with tf.variable_scope('logits'):
                n_filt_in = X_ref.get_shape().as_list()[3]
                n_filt = 512
                k_sz = 3

                with tf.variable_scope('layer_logits'):
                    kernel_shape = [1, 1, n_filt_in, n_filt]
                    weights = tf.get_variable("weights", kernel_shape, initializer=xavier_initializer(),regularizer=wt_reg)
                    biases = tf.get_variable("biases", kernel_shape[-1], initializer=tf.constant_initializer(0))
                    conv = tf.nn.conv2d(X_ref, weights,strides=[1, 1, 1, 1], padding='SAME')
                    conv = batch_norm(conv, decay=0.99, is_training=self.ph['phase_train'])
                mdn_l = tf.nn.relu(conv + biases)

                loc_shape = mdn_l.get_shape().as_list()
                in_filt = loc_shape[-1]
                weights_logits = tf.get_variable("weights_logits", [1, 1, in_filt, k_ref*n_out], initializer=xavier_initializer())
                biases_logits = tf.get_variable("biases_logits", k_ref * n_out, initializer=tf.constant_initializer(0))
                logits = tf.nn.conv2d(mdn_l, weights_logits,
                                      [1, 1, 1, 1], padding='SAME') + biases_logits

                logits = tf.reshape(logits, [-1, n_y, n_x, k_ref, n_out])


            if self.conf.get('predict_occluded',False):
            # predicting occlusion
                X_occ = tf.layers.Conv2D(n_filt,1,padding='same')(net)
                X_occ = tf.layers.batch_normalization(X_occ,training=self.ph['phase_train'])
                X_occ = tf.nn.relu(X_occ)
                X_occ = tf.layers.Conv2D(k_joint*n_out,1)(X_occ)
                occ_pred = tf.reshape(X_occ,[-1,n_x*n_y,k_joint,n_out])
                occ_pred = tf.reshape(occ_pred,[-1,n_x*n_y*k_joint,n_out])
                self.occ_pred = occ_pred

        self.k_joint = k_joint
        self.k_ref = k_ref
        return [locs_joint, locs, logits_joint, logits]


    def l2_loss(self,X,y):

        assert self.k_joint==1, 'This only works for k_joint ==1'
        locs_offset = self.offset
        n_x = self.n_x_j
        n_y = self.n_y_j
        n_classes = self.conf.n_classes

        mdn_locs_joint, mdn_locs, mdn_logits_joint, mdn_logits = X
        logits_all = tf.reshape(mdn_logits_joint,[-1,n_x*n_y])
        ll_joint = tf.nn.softmax(logits_all, axis=1)
        self.softmax_logits = ll_joint
        # All predictions have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        ll_joint = tf.cond(self.ph['phase_train'], lambda: ll_joint + logit_eps, lambda: tf.identity(ll_joint))
        ll_joint = ll_joint / tf.reduce_sum(ll_joint, axis=1, keepdims=True)

        mdn_locs_joint = tf.reshape(mdn_locs_joint,[-1,n_x*n_y,n_classes, 2])
        n_preds_joint = n_x*n_y
        cur_loss = 0
        all_pp = []
        for cls in range(self.conf.n_classes):
            pp = tf.round(y[:, cls:cls + 1, :] /locs_offset)
            occ_pts = tf.is_finite(pp)
            pp = tf.where(occ_pts, pp, tf.zeros_like(pp))
            occ_pts_pred = tf.tile(occ_pts, [1, n_preds_joint, 1])
            qq = mdn_locs_joint[:, :, cls, :]
            qq = tf.where(occ_pts_pred, qq, tf.zeros_like(qq))
            kk = tf.sqrt(tf.reduce_sum(tf.square(pp - qq), axis=2))
            pp = ll_joint * kk
            cur_loss += tf.reduce_sum(pp)
            all_pp.append(pp)
        joint_loss = cur_loss #*self.offset
        self.joint_loss_points = all_pp

        ## refinement
        n_x = self.n_x_r
        n_y = self.n_y_r
        ll_img = tf.reshape(mdn_logits, [-1, n_x * n_y, self.k_ref, n_classes])
        cur_loss = 0
        all_pp = []
        locs_noise = self.conf.get('mdn_joint_ref_noise',1.)
        if locs_noise > 0.001:
            mdn_locs_noise = mdn_locs_joint + tf.random.uniform(mdn_locs_joint.get_shape(),minval=-locs_noise,maxval=locs_noise)
        else:

            mdn_locs_noise = mdn_locs_joint
        mdn_locs_noise = mdn_locs_noise*self.ref_scale
        for b in range(self.conf.batch_size):
            cur_pp = []
            selex = tf.argmax(ll_joint[b, :])
            for cls in range(self.conf.n_classes):
                idx = tf.cast(tf.round(mdn_locs_noise[b,selex,cls,:]),tf.int64)
                # ids are predicted as x,y to match input locs.
                idx_y = tf.clip_by_value(idx[1],0,n_y-1)
                idx_x = tf.clip_by_value(idx[0],0,n_x-1)
                pp = y[b, cls:cls + 1, :] / locs_offset*self.ref_scale
                occ_pts = tf.is_finite(pp)
                pp = tf.where(occ_pts, pp, tf.zeros_like(pp))
                occ_pts_pred = tf.tile(occ_pts, [self.k_ref, 1])
                qq = mdn_locs[b, idx_y,idx_x, :, cls, :]
                qq = tf.where(occ_pts_pred, qq, tf.zeros_like(qq))
                kk = tf.sqrt(tf.reduce_sum(tf.square(pp - qq), axis=1))
                # kk is the distance between all predictions at location selex for point cls
                ll = mdn_logits[b, idx_y, idx_x, :, cls]
                ll = tf.nn.softmax(ll)
                pp = ll * kk
                cur_loss += tf.reduce_sum(pp)
                cur_pp.append(pp)
            all_pp.append(cur_pp)

        self.ref_loss_pp = all_pp
        ref_loss = cur_loss

        tot_loss = joint_loss + ref_loss
        return tot_loss / self.conf.n_classes


    def get_joint_pred(self, preds, occ_out=None):
        locs_joint, locs_ref, logits_joint, logits_ref = preds
        bsz = locs_joint.shape[0]
        n_classes = locs_joint.shape[-2]
        n_x_j = self.n_x_j; n_y_j = self.n_y_j
        n_x_r = self.n_x_r; n_y_r = self.n_y_r
        locs_offset = self.offset
        k_ref = self.k_ref
        ll_joint_img = np.reshape(logits_joint,[-1,n_x_j*n_y_j])
        ll_img = np.reshape(logits_ref,[-1,n_x_r*n_y_r,k_ref,n_classes])
        locs_ref = locs_ref * locs_offset / self.ref_scale

        preds_ref = np.zeros([bsz,n_classes,2])
        preds_joint = np.zeros([bsz,n_classes,2])
        preds_occ = np.ones([bsz,n_classes])*np.nan
        for ndx in range(bsz):
            sel_ex = np.argmax(ll_joint_img[ndx, :])
            idx = np.unravel_index(sel_ex, [n_y_j, n_x_j])
            preds_joint[ndx,...] = locs_joint[ndx,idx[0],idx[1],...]* locs_offset
            for cls in range(n_classes):
                mm = np.round(locs_joint[ndx,idx[0],idx[1],cls,:]).astype('int')*self.ref_scale
                mm_y = np.clip(mm[1],0,n_y_r-1)
                mm_x = np.clip(mm[0],0,n_x_r-1)
                pt_selex = np.argmax(logits_ref[ndx, mm_y, mm_x, :, cls])
                cur_pred = locs_ref[ndx,mm_y,mm_x,pt_selex,cls,:]
                preds_ref[ndx,cls,:] = cur_pred
            if self.conf.predict_occluded and (occ_out is not None):
                preds_occ[ndx,:] = occ_out[ndx, idx[0], idx[1], ...]
        return preds_ref, preds_joint,preds_occ

    def compute_dist(self, preds, locs):
        locs = locs.copy()
        return np.linalg.norm(self.get_joint_pred(preds)[0] - locs, axis=-1).mean()
               

    def compute_train_data(self, sess, db_type):
        self.fd_train() if db_type is self.DBType.Train \
            else self.fd_val()
        cur_loss, cur_inputs , cur_pred= sess.run(
            [self.cost,self.inputs, self.pred], self.fd)

        cur_dist = self.compute_dist( cur_pred, cur_inputs[1])

        cur_dict = {'cur_loss': cur_loss, 'cur_dist': cur_dist}
        return cur_dict

    def get_pred_fn(self, model_file=None,distort=False,tmr_pred=None):

        sess, latest_model_file = self.restore_net(model_file)
        self.sess = sess
        if tmr_pred is None:
            tmr_pred = contextlib.suppress()
        conf = self.conf
        pred_occ = self.conf.predict_occluded

        def pred_fn(all_f):
            # this is the function that is used for classification.
            # this should take in an array B x H x W x C of images, and
            # output an array of predicted locations.
            # predicted locations should be B x N x 2
            # pt.get_pred_locs can be used to convert heatmaps into locations.

            bsize = conf.batch_size
            xs, _ = pt.preprocess_ims(
                all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf, distort=distort, scale=self.conf.rescale)

            self.fd[self.inputs[0]] = xs
            self.fd[self.ph['phase_train']] = False
            self.fd[self.ph['learning_rate']] = 0
            out_list = [self.pred, self.inputs]

            if pred_occ:
                out_list.append(self.occ_pred)

            with tmr_pred:
                out = sess.run(out_list, self.fd)

            pred = out[0]
            cur_input = out[1]

            if pred_occ:
                occ_out = out[-1]
            else:
                occ_out = None

            pred_locs = self.get_joint_pred(pred)
            ret_dict = {}
            ret_dict['locs'] = pred_locs[0]*self.conf.rescale
            ret_dict['locs_joint'] = pred_locs[1]*self.conf.rescale
            ret_dict['conf'] = np.ones([bsize,self.conf.n_classes])
            ret_dict['occ'] = pred_locs[2]
            return ret_dict

        def close_fn():
            sess.close()

        return pred_fn, close_fn, latest_model_file


    def train_wrapper(self,restore=False,model_file=None):
        self.train_umdn(restore,model_file=model_file)


import PoseCommon_pytorch
import Pose_multi_mdn_joint_torch
import torch
import PoseTools
import numpy as np
from poseConfig import conf

class Pose_mdn_joint_torch(Pose_multi_mdn_joint_torch.Pose_multi_mdn_joint_torch):
    def __init__(self,conf,**kwargs):
        conf.max_n_animals = 1
        conf.min_n_animals = 1
        conf.multi_loss_mask = False
        super(Pose_mdn_joint_torch, self).__init__(conf, **kwargs)

    def create_targets(self, inputs):
        target_dict = {'locs':inputs['locs'][:,None,...]}
        if 'mask' in inputs.keys():
            target_dict['mask'] = inputs['mask']
        target_dict['occ'] = inputs['occ'][:,None,...]
        return target_dict


    def get_pred_fn(self, model_file=None,max_n=None,imsz=None):
        assert not self.conf.is_multi, 'This is for single animal'
        if imsz is not None:
            self.conf.imsz = imsz


        if model_file is None:
            latest_model_file = self.get_latest_model_file()
        else:
            latest_model_file = model_file

        self.set_version(latest_model_file)

        model = self.create_model()
        model = torch.nn.DataParallel(model)

        self.restore(latest_model_file,model)
        model.to(self.device)
        model.eval()
        self.model = model
        conf = self.conf
        match_dist_factor = conf.get('multi_match_dist_factor',0.2)

        def pred_fn(ims, retrawpred=False):
            locs_sz = (conf.batch_size, conf.n_classes, 2)
            locs_dummy = np.zeros(locs_sz)

            ims, _ = PoseTools.preprocess_ims(ims,locs_dummy,conf,False,conf.rescale)
            with torch.no_grad():
                preds = model({'images':torch.tensor(ims).permute([0,3,1,2])/255.})

            locs = self.get_joint_pred(preds)
            ret_dict = {}
            ret_dict['locs'] = locs['ref'][:,0,...] * conf.rescale

            if ('conf_dist' not in locs) or (locs['conf_dist'] is None):
                cur_joint_conf = locs['conf_ref'][:,0,...]
                ret_dict['conf'] = 1/(1+np.exp(-cur_joint_conf))
            else:
                cur_joint_conf = locs['conf_dist'][:,0,...]
                ret_dict['conf'] = 1/np.clip(cur_joint_conf,0,25)

            ret_dict['conf_joint'] = locs['conf_joint'][...,0]
            if self.conf.predict_occluded:
                ret_dict['occ'] = locs['pred_occ'][:,0,...]
            else:
                ret_dict['occ'] = np.ones_like(locs['pred_occ'][:,0])*np.nan
            if retrawpred:
                ret_dict['preds'] = preds
                ret_dict['raw_locs'] = locs
            return ret_dict

        def close_fn():
            del self.model
            torch.cuda.empty_cache()

        return pred_fn, close_fn, latest_model_file

class Pose_mdn_joint(Pose_mdn_joint_torch):
    def __init__(self,conf,**kwargs):
        super(Pose_mdn_joint,self).__init__(conf,**kwargs)
