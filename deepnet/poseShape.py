# coding: utf-8

# In[2]:

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from past.utils import old_div
import tensorflow as tf
import os
import sys
import numpy as np
import scipy
import scipy.spatial
import math
import cv2
import tempfile
import copy
import re
import h5py

from batch_norm import *
import myutils
import PoseTools
import localSetup
import operator
import copy
import convNetBase as CNB
import mpiiData


# In[3]:

def conv_relu(X, kernel_shape, conv_std, bias_val, do_batch_norm, train_phase, add_summary=True):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
    # tf.random_normal_initializer(stddev=conv_std))
    biases = tf.get_variable("biases", kernel_shape[-1], initializer=tf.constant_initializer(bias_val))
    if add_summary:
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
            #     PoseTools.variable_summaries(biases)
    conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
    if do_batch_norm:
        conv = batch_norm(conv, train_phase)
    with tf.variable_scope('conv'):
        PoseTools.variable_summaries(conv)
    return tf.nn.relu(conv - biases)


def conv_relu_norm_init(X, kernel_shape, conv_std, bias_val, do_batch_norm, train_phase, add_summary=True):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(stddev=conv_std))
    biases = tf.get_variable("biases", kernel_shape[-1], initializer=tf.constant_initializer(bias_val))
    if add_summary:
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
            #     PoseTools.variable_summaries(biases)
    conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
    if do_batch_norm:
        conv = batch_norm(conv, train_phase)
    with tf.variable_scope('conv'):
        PoseTools.variable_summaries(conv)
    return tf.nn.relu(conv - biases)


def fc_2d(S, n_filt, train_phase, add_summary=True):
    in_dim = S.get_shape()[1]
    weights = tf.get_variable("weights", [in_dim, n_filt], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", n_filt, initializer=tf.constant_initializer(0))
    if add_summary:
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(biases)

    fc_out = tf.nn.relu(batch_norm_2D(tf.matmul(S, weights), train_phase) - biases)
    with tf.variable_scope('fc'):
        PoseTools.variable_summaries(fc_out)
    return fc_out


def fc_2d_norm_init(S, n_filt, train_phase, conv_std, add_summary=True):
    in_dim = S.get_shape()[1]
    weights = tf.get_variable("weights", [in_dim, n_filt], initializer=tf.random_normal_initializer(stddev=conv_std))
    biases = tf.get_variable("biases", n_filt, initializer=tf.constant_initializer(0))
    if add_summary:
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(biases)

    fc_out = tf.nn.relu(batch_norm_2D(tf.matmul(S, weights), train_phase) - biases)
    with tf.variable_scope('fc'):
        PoseTools.variable_summaries(fc_out)
    return fc_out


def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)


def create_place_holders(conf):
    psz = conf.shape_psz
    n_classes = conf.n_classes
    img_dim = conf.imgDim+1

    nex = conf.batch_size
    x0 = []
    x1 = []
    x2 = []

    for ndx in range(len(conf.shape_selpt1)):
        x0.append(tf.placeholder(tf.float32, [nex, psz, psz, img_dim], name='x0_{}'.format(ndx)))
        x1.append(tf.placeholder(tf.float32, [nex, psz, psz, img_dim], name='x1_{}'.format(ndx)))
        x2.append(tf.placeholder(tf.float32, [nex, psz, psz, img_dim], name='x2_{}'.format(ndx)))

    n_out = 0
    for selpt2 in conf.shape_selpt2:
        n_out += len(selpt2)
    n_bins = len(conf.shape_r_bins)-1
    # y = tf.placeholder(tf.float32, [nex, conf.shape_n_orts*n_out*n_bins], 'out')
    y = tf.placeholder(tf.float32, [nex, psz,psz,n_out], 'out')

    X = [x0, x1, x2]

    phase_train = tf.placeholder(tf.bool, name='phase_train')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    ph = {'X': X, 'y': y, 'phase_train': phase_train, 'learning_rate': learning_rate}
    return ph


def create_feed_dict(ph, conf):
    feed_dict = {}
    for ndx in range(len(conf.shape_selpt1)):
        feed_dict[ph['X'][0][ndx]] = []
        feed_dict[ph['X'][1][ndx]] = []
        feed_dict[ph['X'][2][ndx]] = []
    feed_dict[ph['y']] = []
    feed_dict[ph['learning_rate']] = 1
    feed_dict[ph['phase_train']] = False

    return feed_dict


def net_multi_base_named(X, n_filt, do_batch_norm, train_phase, add_summary=True):
    in_dim_x = X.get_shape()[3]
    nex = X.get_shape()[0].value

    with tf.variable_scope('layer1_X'):
        conv1 = conv_relu_norm_init(X, [5, 5, in_dim_x, 48], 0.3, 0, do_batch_norm, train_phase, add_summary)
        #         pool1 = max_pool('pool1',conv1,k=3,s=2)
        pool1 = conv1

    with tf.variable_scope('layer2'):
        conv2 = conv_relu(pool1, [3, 3, 48, n_filt], 0.01, 0, do_batch_norm, train_phase, add_summary)
        #         pool2 = max_pool('pool2',conv2,k=3,s=2)
        pool2 = conv2

    with tf.variable_scope('layer3'):
        conv3 = conv_relu(pool2, [3, 3, n_filt, n_filt], 0.01, 0, do_batch_norm, train_phase, add_summary)
        # pool3 = max_pool('pool3', conv3, k=3, s=2)
        pool3 = conv3

    with tf.variable_scope('layer4'):
        conv4 = conv_relu(pool3, [3, 3, n_filt, n_filt], 0.01, 0, do_batch_norm, train_phase, add_summary)
        # pool4 = max_pool('pool4', conv4, k=3, s=2)
        pool4 = conv4
    # conv4_reshape = tf.reshape(pool4, [nex, -1])

    # with tf.variable_scope('layer5'):
    #     conv5 = fc_2d_norm_init(conv4_reshape, 128, train_phase, 0.01, add_summary)
    with tf.variable_scope('layer5'):
        conv5 = conv_relu(conv4,[3,3,n_filt,n_filt],0.01,1,do_batch_norm,train_phase)

    out_dict = {'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4, 'conv5': conv5}
    return conv5, out_dict

def upscale(name,l_input,sz):
    l_out = tf.image.resize_nearest_neighbor(l_input,sz,name=name)
    return l_out

def net_multi_conv(ph, conf):
    X = ph['X']
    X0, X1, X2 = X
    imsz = conf.imsz; rescale = conf.rescale
    pool_scale = conf.pool_scale
    nfilt = conf.nfilt

    # out_size = ph['y'].get_shape()[1]
    train_phase = ph['phase_train']

    n_filter = conf.nfilt
    do_batch_norm = conf.doBatchNorm

    base_dict_array = []
    out_array = []
    for ndx,selpt in enumerate(conf.shape_selpt1):
        n_bins = len(conf.shape_r_bins)-1
        n_out = conf.shape_n_orts * len(conf.shape_selpt2[ndx])*n_bins

        with tf.variable_scope('scale0_{}'.format(ndx)):
            conv5_0, base_dict_0 = net_multi_base_named(X0[ndx], n_filter, do_batch_norm, train_phase, True)
        with tf.variable_scope('scale1_{}'.format(ndx)):
            conv5_1, base_dict_1 = net_multi_base_named(X1[ndx], n_filter, do_batch_norm, train_phase, False)
        with tf.variable_scope('scale2_{}'.format(ndx)):
            conv5_2, base_dict_2 = net_multi_base_named(X2[ndx], n_filter, do_batch_norm, train_phase, False)
        conv5_cat = tf.concat([conv5_0, conv5_1, conv5_2],1)

        sz0 = conv5_0.shape.as_list()[1]
        sz1 = conv5_0.shape.as_list()[2]
        conv5_1_up = upscale('5_1', conv5_1, [sz0, sz1])
        conv5_2_up = upscale('5_2', conv5_2, [sz0, sz1])

        # crop lower res layers to match higher res size
        conv5_0_sz = tf.Tensor.get_shape(conv5_0).as_list()
        conv5_1_sz = tf.Tensor.get_shape(conv5_1_up).as_list()
        crop_0 = int(old_div((sz0 - conv5_0_sz[1]), 2))
        crop_1 = int(old_div((sz1 - conv5_0_sz[2]), 2))

        curloc = [0, crop_0, crop_1, 0]
        patchsz = tf.to_int32([-1, conv5_0_sz[1], conv5_0_sz[2], -1])
        conv5_1_up = tf.slice(conv5_1_up, curloc, patchsz)
        conv5_2_up = tf.slice(conv5_2_up, curloc, patchsz)

        conv5_cat = tf.concat([conv5_0, conv5_1_up, conv5_2_up], 3)

        with tf.variable_scope('layer6'):
            conv6 = conv_relu(conv5_cat,
                              [conf.psz, conf.psz, conf.numscale * nfilt, conf.nfcfilt],
                              0.005, 1, True, train_phase)

        with tf.variable_scope('layer7'):
            conv7 = conv_relu(conv6, [1, 1, conf.nfcfilt, conf.nfcfilt],
                              0.005, 1, True, train_phase)

        with tf.variable_scope('layer8'):
            l8_weights = tf.get_variable("weights", [1, 1, conf.nfcfilt, len(conf.shape_selpt2[ndx])],
                                         initializer=tf.random_normal_initializer(stddev=0.01))
            l8_biases = tf.get_variable("biases", len(conf.shape_selpt2[ndx]),
                                        initializer=tf.constant_initializer(0))
            out = tf.nn.conv2d(conv7, l8_weights,
                               strides=[1, 1, 1, 1], padding='SAME') + l8_biases
            out_array.append(out)
    out = tf.concat(out_array,3)
    return out,{}

            # with shape context kind of output
    #     with tf.variable_scope('L6_{}'.format(ndx)):
    #         l6 = fc_2d(conv5_cat, 256, train_phase)
    #     with tf.variable_scope('L7_{}'.format(ndx)):
    #         l7 = fc_2d(l6, 256, train_phase)
    #     with tf.variable_scope('L8_{}'.format(ndx)):
    #         l8 = fc_2d(l7, 256, train_phase)
    #
    #     with tf.variable_scope('out_{}'.format(ndx)):
    #         weights = tf.get_variable("weights", [l8.get_shape()[1].value, n_out],
    #                                   initializer=tf.random_normal_initializer(stddev=0.2))
    #         biases = tf.get_variable("biases", n_out, initializer=tf.constant_initializer(0))
    #         with tf.variable_scope('weights'):
    #             PoseTools.variable_summaries(weights)
    #         with tf.variable_scope('biases'):
    #             PoseTools.variable_summaries(biases)
    #
    #     base_dict_array.append([base_dict_0, base_dict_2, base_dict_2, l6, l7,l8])
    #     out_array.append(tf.matmul(l8, weights) - biases)
    #
    # out = tf.concat(out_array,1)
    # out_dict = {'base_dict_array': base_dict_array}
    #
    # return out, out_dict


def open_dbs(conf, train_type=0):
    if train_type == 0:
        train_filename = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
        val_filename = os.path.join(conf.cachedir, conf.valfilename) + '.tfrecords'
        train_queue = tf.train.string_input_producer([train_filename])
        val_queue = tf.train.string_input_producer([val_filename])
    else:
        train_filename = os.path.join(conf.cachedir, conf.fulltrainfilename) + '.tfrecords'
        val_filename = os.path.join(conf.cachedir, conf.fulltrainfilename) + '.tfrecords'
        train_queue = tf.train.string_input_producer([train_filename])
        val_queue = tf.train.string_input_producer([val_filename])
    return [train_queue, val_queue]


def create_cursors(sess, queue, conf):
    train_queue, val_queue = queue
    train_ims, train_locs, train_exp_data = mpiiData.read_and_decode(train_queue)
    val_ims, val_locs, val_exp_data = mpiiData.read_and_decode(val_queue)
    train_data = [train_ims, train_locs,train_exp_data]
    val_data = [val_ims, val_locs,val_exp_data]
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return [train_data, val_data], coord, threads


def read_images(conf, db_type, distort, sess, data):
    train_data, val_data = data
    cur_data = val_data if (db_type == 'val') else train_data
    xs = []
    locs = []
    exp_data = []

    count = 0
    while count < conf.batch_size:
        [cur_xs, cur_locs, cur_exp_data] = sess.run(cur_data)

        # kk = cur_locs[conf.shape_selpt2, :] - cur_locs[conf.shape_selpt1, :]
        # dd = np.sqrt(kk[0] ** 2 + kk[1] ** 2)
        # if dd>150:
        #   continue

        if np.ndim(cur_xs) < 3:
            xs.append(cur_xs[np.newaxis, :, :])
        else:
            cur_xs = np.transpose(cur_xs,[2,0,1])
            xs.append(cur_xs)
        locs.append(cur_locs)
        exp_data.append(cur_exp_data)
        count += 1


    xs = np.array(xs)
    locs = np.array(locs)
    if distort:
        if conf.horzFlip:
            xs, locs = PoseTools.randomly_flip_lr(xs, locs)
        if conf.vertFlip:
            xs, locs = PoseTools.randomly_flip_ud(xs, locs)
        xs, locs = PoseTools.randomly_rotate(xs, locs, conf)
        # xs = PoseTools.randomlyAdjust(xs, conf)

    return xs, locs, exp_data


# In[2]:

def update_feed_dict(conf, db_type, distort, sess, data, feed_dict, ph):
    xs, locs, exp_data = read_images(conf, db_type, distort, sess, data)

    shape_perturb_rad = conf.shape_perturb_rad

    sel_pt1 = conf.shape_selpt1
    sel_pt2 = conf.shape_selpt2

    assert len(sel_pt1)==1, "current implementation only works for 1 pt"
    assert len(sel_pt2[0])==1, "current implementation only works for 1 pt"

    psz = conf.shape_psz
    # perturb the locs a bit
    sel_locs = []
    all_label_locs = []
    all_label_ims = []
    for ndx,count in enumerate(sel_pt1):
        cur_locs = copy.deepcopy(locs)
        cur_locs[:,count,0] += np.random.randn()*shape_perturb_rad
        cur_locs[:,count,1] += np.random.randn()*shape_perturb_rad
        cur_locs[cur_locs<0] = np.nan
        cur_locs[cur_locs[:,:,0]>conf.imsz[1],0] = np.nan
        cur_locs[cur_locs[:,:,1]>conf.imsz[0],1] = np.nan
        sel_locs.append(cur_locs)
        label_locs = copy.deepcopy(cur_locs)
        label_locs -= (label_locs[:,count:count+1,:]-psz/2)
        label_locs[label_locs<0] = np.nan
        label_locs[label_locs>=psz] = np.nan
        label_ims = PoseTools.create_label_images(label_locs, [psz, psz], 1, conf.label_blur_rad)
        label_ims = label_ims[...,sel_pt2[ndx]]
        all_label_locs.append(label_locs[:,sel_pt2[ndx],:])
        all_label_ims.append(label_ims)
        # ind_labels = shape_from_locs(cur_locs,conf)
        # labels = []
        # curlabels = ind_labels[:, sel_pt1, sel_pt2[ndx], ...]
        # labels.append(curlabels.reshape([curlabels.shape[0],-1]))

    all_label_locs = np.concatenate(all_label_locs,axis=1)
    # labels = np.concatenate(labels,1)
    feed_dict[ph['y']] = np.concatenate(all_label_ims,axis=3)

    x0, x1, x2 = PoseTools.multi_scale_images(xs.transpose([0, 2, 3, 1]),
                                              conf.rescale, conf.scale, conf.l1_cropsz, conf)

    for ndx, count in enumerate(sel_pt1):
        cur_locs = sel_locs[ndx]
        feed_dict[ph['X'][0][ndx]] = extract_patches(x0, cur_locs[:, count, :], psz)
        feed_dict[ph['X'][1][ndx]] = extract_patches(x1, old_div((cur_locs[:, count, :]), conf.scale), psz)
        feed_dict[ph['X'][2][ndx]] = extract_patches(x2, old_div((cur_locs[:, count, :]), (conf.scale ** 2)), psz)

    return locs, xs, all_label_locs, exp_data


def angle_from_locs(locs):
    n_orts = 8
    n_pts = locs.shape[1]
    bsz = locs.shape[0]
    yy = np.zeros([bsz, n_pts, n_pts, n_orts])
    for ndx in range(bsz):
        curl = locs[ndx, ...]
        rloc = np.tile(curl, [n_pts, 1, 1])
        kk = rloc - curl[:, np.newaxis, :]
        aa = np.arctan2(kk[:, :, 1], kk[:, :, 0] + 1e-5) * 180 / np.pi + 180
        for i in range(n_orts):
            pndx = np.abs(np.mod(aa - 360 / n_orts * i + 45, 360) - 45 - 22.5) < 32.5
            zz = np.zeros([n_pts, n_pts])
            zz[pndx] = 1
            yy[ndx, ..., i] = zz
    return yy


def dist_from_locs(locs):
    n_pts = locs.shape[1]
    bsz = locs.shape[0]
    yy = np.zeros([bsz, n_pts, n_pts])
    for ndx in range(bsz):
        curl = locs[ndx, ...]
        dd = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(curl))
        yy[ndx, ...] = dd
    return yy


def shape_from_locs(locs,conf):
    # shape context kinda labels
    n_angle = conf.shape_n_orts
    r_bins = np.array(conf.shape_r_bins)
    n_radius = len(r_bins) - 1
    n_pts = locs.shape[1]
    bsz = locs.shape[0]
    yy = np.zeros([bsz, n_pts, n_pts, n_angle, n_radius])
    for ndx in range(bsz):
        curl = locs[ndx, ...]
        dd = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(curl))
        dd_bins = np.digitize(dd, r_bins)
        r_loc = np.tile(curl, [n_pts, 1, 1])
        kk = r_loc - curl[:, np.newaxis, :]
        aa = np.arctan2(kk[:, :, 1], kk[:, :, 0] + 1e-5) * 180 / np.pi + 180

        for i in range(n_angle):
            for d_bin in range(n_radius):
                pndx = np.abs(np.mod(aa - 360 / n_angle * i + 45, 360) - 45 - 22.5) <= 22.5
                zndx = pndx & (dd_bins == d_bin + 1)
                zz = np.zeros([n_pts, n_pts])
                zz[zndx] = 1
                yy[ndx, ..., i, d_bin] = zz
    return yy



def extract_patches(img, locs, psz):
    zz = np.zeros([img.shape[0],psz,psz,img.shape[-1]+1])
    pad_arg = [(psz, psz), (psz, psz), (0, 0)]
    int_locs = np.round(locs).astype('int')

    x_mesh, y_mesh = np.meshgrid(np.arange(float(psz)),
                                 np.arange(float(psz)))
    x_mesh -= float(psz)/2
    y_mesh -= float(psz) / 2
    loc_image = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
    loc_image -= psz/2

    for ndx in range(img.shape[0]):
        if np.isnan(locs[ndx,0]) or np.isnan(locs[ndx,1]):
            continue
        p_img = np.pad(img[ndx, ...], pad_arg, 'constant')
        zz[ndx,...,:-1]= p_img[(int_locs[ndx, 1] + psz - old_div(psz, 2)):(int_locs[ndx, 1] + psz + old_div(psz, 2)),
                  (int_locs[ndx, 0] + psz - old_div(psz, 2)):(int_locs[ndx, 0] + psz + old_div(psz, 2)), :]
        zz[ndx,...,-1] = loc_image

    return np.array(zz)


# In[ ]:

def init_shape_prior(conf):
    #     global shape_prior
    labels = h5py.File(conf.labelfile, 'r')

    if 'pts' in labels:
        pts = np.array(labels['pts'])
    else:
        pp = np.array(labels['labeledpos'])
        n_movie = pp.shape[1]
        pts = np.zeros([0, conf.n_classes, 2])
        for ndx in range(n_movie):
            cur_pts = np.array(labels[pp[0, ndx]])
            frames = np.where(np.invert(np.all(np.isnan(cur_pts), axis=(1, 2))))[0]
            n_pts_per_view = np.array(labels['cfg']['NumLabelPoints'])[0, 0]
            pts_st = int(conf.view * n_pts_per_view)
            sel_pts = pts_st + conf.selpts
            cur_locs = cur_pts[:, :, sel_pts]
            cur_locs = cur_locs[frames, :, :]
            cur_locs = cur_locs.transpose([0, 2, 1])
            pts = np.append(pts, cur_locs[:, :, :], axis=0)
    shape_prior = np.mean(shape_from_locs(pts,conf) > 0, axis=0)
    return shape_prior


# In[ ]:

def restore_shape(sess, shape_saver, restore, conf, feed_dict):
    out_filename = os.path.join(conf.cachedir, conf.shapeoutname)
    latest_ckpt = tf.train.get_checkpoint_state(conf.cachedir, latest_filename=conf.shapeckptname)
    sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
    if not latest_ckpt or not restore:
        shape_start_at = 0
        print("Not loading Shape variables. Initializing them")
    else:
        shape_saver.restore(sess, latest_ckpt.model_checkpoint_path)
        match_obj = re.match(out_filename + '-(\d*)', latest_ckpt.model_checkpoint_path)
        shape_start_at = int(match_obj.group(1)) + 1
        print("Loading shape variables from %s" % latest_ckpt.model_checkpoint_path)
    return shape_start_at


def save_shape(sess, shape_saver, step, conf):
    out_filename = os.path.join(conf.cachedir, conf.shapeoutname)
    shape_saver.save(sess, out_filename, global_step=step, latest_filename=conf.shapeckptname)
    print('Saved state to %s-%d' % (out_filename, step))


def create_shape_saver(conf):
    shape_saver = tf.train.Saver(var_list=PoseTools.get_vars('shape'), max_to_keep=conf.maxckpt)
    return shape_saver


def print_gradients(sess, feed_dict, loss):
    vv = tf.global_variables()
    aa = [v for v in vv if
          not re.search('Adam|batch_norm|beta|scale[1-2]|scale0_[1-9][0-9]*|fc_[1-9][0-9]*|L[6-7]_[1-9][0-9]*|biases',
                        v.name)]

    grads = sess.run(tf.gradients(loss, aa), feed_dict=feed_dict)

    wts = sess.run(aa, feed_dict=feed_dict)

    grads_std = [g.std() for g in grads]
    wts_std = [w.std() for w in wts]

    grads_by_wts = [old_div(s, w) for s, w in zip(grads_std, wts_std)]

    bb = [[r, n.name] for r, n in zip(grads_by_wts, aa)]
    for b, k, g in zip(bb, grads_std, wts_std):
        print(b, k, g)


# In[ ]:

def pose_shape_net_init(conf):
    ph = create_place_holders(conf)
    feed_dict = create_feed_dict(ph, conf)
    # init_shape_prior(conf)
    with tf.variable_scope('shape'):
        out, out_dict = net_multi_conv(ph, conf)
    # change 3 22022017
    #         out,out_dict = net_multi_conv(ph,conf)
    train_type = 0
    queue = open_dbs(conf, train_type=train_type)
    if train_type == 1:
        print("Training with all the data!")
        print("Validation data is same as training data!!!! ")
    return ph, feed_dict, out, queue, out_dict

def print_shape_accuracy(correct_pred,conf):
    acc = []
    ptsDone = 0
    for ndx in range(len(conf.shape_selpt1)):
        n = len(conf.shape_selpt2[ndx])
        n_o = conf.shape_n_orts
        n_r = len(conf.shape_r_bins)-1
        start = ptsDone * n_o * n_r
        ptsDone += n
        stop = ptsDone * n_o * n_r
        cur_acc = correct_pred[:,start:stop].reshape(correct_pred.shape[0],n,n_o,n_r)

        print('{}:'.format(conf.shape_selpt1[ndx]))
        print(cur_acc.mean(axis=0).squeeze())


def getPredError(locs,pred):
    locerr = np.zeros(locs.shape)
    for ndx in range(pred.shape[0]):
        for cls in range(pred.shape[-1]):
            maxndx = np.argmax(pred[ndx,:,:,cls])
            predloc = np.array(np.unravel_index(maxndx,pred.shape[1:3]))
            locerr[ndx][cls][0]= float(predloc[1])-locs[ndx][cls][0]
            locerr[ndx][cls][1]= float(predloc[0])-locs[ndx][cls][1]
    return np.sqrt(np.sum((locerr**2),2))


def pose_shape_train(conf, restore=True):
    ph, feed_dict, out, queue, _ = pose_shape_net_init(conf)
    feed_dict[ph['phase_train']] = True
    shape_saver = create_shape_saver(conf)

    np.set_printoptions(precision=3,suppress=True)
    # for weighted..
    # y_re = tf.reshape(ph['y'], [conf.batch_size, 1,1,conf.shape_n_orts,len(conf.shape_r_bins)-1])
    # sel_pt1 = conf.shape_selpt1[0]
    # sel_pt2 = conf.shape_selpt2[0][0]
    # shape_prior = init_shape_prior(conf)
    # wt_den = shape_prior[sel_pt1:sel_pt1+1, sel_pt2:sel_pt2+1, ...]
    # wt = tf.reduce_max(old_div(y_re, (wt_den + 0.1)), axis=(1, 2,3,4))
    loss = tf.nn.l2_loss(out - ph['y'])

    # loss = tf.nn.l2_loss(out-ph['y'])
    # correct_pred = tf.cast(tf.equal(out > 0.5, ph['y'] > 0.5), tf.float32)
    # accuracy = tf.reduce_mean(correct_pred)

    #     tf.summary.scalar('cross_entropy',loss)
    #     tf.summary.scalar('accuracy',accuracy)

    opt = tf.train.AdamOptimizer(learning_rate=ph['learning_rate']).minimize(loss)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        #         train_writer = tf.summary.FileWriter(conf.cachedir + '/shape_train_summary',sess.graph)
        #         test_writer = tf.summary.FileWriter(conf.cachedir + '/shape_test_summary',sess.graph)
        data, coord, threads = create_cursors(sess, queue, conf)
        update_feed_dict(conf, 'train', distort=True, sess=sess,
                         data=data, feed_dict=feed_dict, ph=ph)
        shape_start_at = restore_shape(sess, shape_saver, restore, conf, feed_dict)
        for step in range(shape_start_at, conf.shape_training_iters + 1):
            ex_count = step * conf.batch_size
            cur_lr = conf.shape_learning_rate * \
                     conf.gamma**math.floor(old_div(ex_count, conf.step_size))
            feed_dict[ph['learning_rate']] = cur_lr
            feed_dict[ph['phase_train']] = True
            update_feed_dict(conf, 'train', distort=True, sess=sess,
                             data=data, feed_dict=feed_dict, ph=ph)
            sess.run(opt, feed_dict=feed_dict)
            # train_writer.add_summary(train_summary,step)

            if step % conf.display_step == 0:
                xs,locs,label_locs,exp_data = update_feed_dict(conf, 'train', sess=sess,
                                                      distort=True, data=data,
                                                      feed_dict=feed_dict, ph=ph)
                feed_dict[ph['phase_train']] = False
                train_loss, train_pred = sess.run([loss,out], feed_dict=feed_dict)
                train_dist = np.nanmean(getPredError(label_locs, train_pred))
                train_loss /= (conf.shape_psz**2)*train_pred.shape[-1]*conf.batch_size
                train_loss = np.sqrt(train_loss)

                num_rep = int(old_div(conf.numTest, conf.batch_size)) + 1
                val_loss = 0.
                val_dist = 0.
                for rep in range(num_rep):
                    xs, locs, label_locs,exp_data = update_feed_dict(conf, 'val', distort=False,
                                                            sess=sess, data=data,
                                                            feed_dict=feed_dict, ph=ph)
                    vloss,vpred = sess.run([loss,out], feed_dict=feed_dict)
                    v_dist = np.nanmean(getPredError(label_locs, vpred))
                    vloss /= (conf.shape_psz**2)*train_pred.shape[-1] * conf.batch_size
                    vloss = np.sqrt(vloss)
                    val_loss += vloss
                    val_dist += v_dist
                # val_acc_wt += vacc_wt
                val_loss /= num_rep
                val_dist /= num_rep
                #                 val_acc_wt /= num_rep
                # test_summary, _ = sess.run([merged, loss], feed_dict=feed_dict)
                #                 test_writer.add_summary(test_summary,step)
                print('Val -- Dist:{:.2f} Loss:{:.4f} Train Dist:{:.4f} Loss:{:.4f} Iter:{}'\
                      .format(val_dist,val_loss,train_dist, train_loss,step))

            # print_gradients(sess,feed_dict,loss)
            if step % conf.save_step == 0:
                save_shape(sess, shape_saver, step, conf)
        print("Optimization Done!")
        save_shape(sess, shape_saver, step, conf)
        # train_writer.close()
        # test_writer.close()
        coord.request_stop()
        coord.join(threads)


# In[ ]:

def gen_labels(r_locs, locs, conf):
    d2locs = np.sqrt(((r_locs - locs[..., np.newaxis]) ** 2).sum(-2))
    ll = np.arange(1, conf.n_classes + 1)
    labels = np.tile(ll[:, np.newaxis], [d2locs.shape[0], 1, d2locs.shape[2]])
    labels[d2locs > conf.poseshapeNegDist] = -1.
    labels[d2locs < conf.poseshapeNegDist] = 1.
    labels = np.concatenate([labels[:, np.newaxis], 1 - labels[:, np.newaxis]], -1)


# In[ ]:

def gen_random_neg_samples(bout, l7out, locs, conf, n_samples=10):
    sz = (np.array(l7out.shape[1:3]) - 1) * conf.rescale * conf.pool_scale
    b_size = conf.batch_size
    r_locs = np.zeros(locs.shape + (n_samples,))
    r_locs[:, :, 0, :] = np.random.randint(sz[1], size=locs.shape[0:2] + (n_samples,))
    r_locs[:, :, 1, :] = np.random.randint(sz[0], size=locs.shape[0:2] + (n_samples,))
    return r_locs


# In[ ]:

def gen_gaussian_pos_samples(bout, l7out, locs, conf, nsamples=10, max_len=4):
    scale = conf.rescale * conf.pool_scale
    sigma = float(max_len) * 0.5 * scale
    sz = (np.array(l7out.shape[1:3]) - 1) * scale
    b_size = conf.batch_size
    r_locs = np.round(np.random.normal(size=locs.shape + (15 * nsamples,)) * sigma)
    # remove r_locs that are far away.
    d_locs = np.all(np.sqrt((r_locs ** 2).sum(2)) < (max_len * scale), 1)
    c_locs = np.zeros(locs.shape + (nsamples,))
    for ii in range(d_locs.shape[0]):
        ndx = np.where(d_locs[ii, :])[0][:nsamples]
        c_locs[ii, :, :, :] = r_locs[ii, :, :, ndx].transpose([1, 2, 0])

    r_locs = locs[..., np.newaxis] + c_locs

    # sanitize the locs
    r_locs[r_locs < 0] = 0
    xlocs = r_locs[:, :, 0, :]
    xlocs[xlocs >= sz[1]] = sz[1] - 1
    r_locs[:, :, 0, :] = xlocs
    ylocs = r_locs[:, :, 1, :]
    ylocs[ylocs >= sz[0]] = sz[0] - 1
    r_locs[:, :, 1, :] = ylocs
    return r_locs


# In[ ]:

def gen_gaussian_neg_samples(bout, locs, conf, nsamples=10, minlen=8):
    sigma = minlen
    #     sz = (np.array(bout.shape[1:3])-1)*scale
    sz = np.array(bout.shape[1:3]) - 1
    bsize = conf.batch_size
    rlocs = np.round(np.random.normal(size=locs.shape + (5 * nsamples,)) * sigma)
    # remove rlocs that are small.
    dlocs = np.sqrt((rlocs ** 2).sum(2)).sum(1)
    clocs = np.zeros(locs.shape + (nsamples,))
    for ii in range(dlocs.shape[0]):
        ndx = np.where(dlocs[ii, :] > (minlen * conf.n_classes))[0][:nsamples]
        clocs[ii, :, :, :] = rlocs[ii, :, :, ndx].transpose([1, 2, 0])

    rlocs = locs[..., np.newaxis] + clocs

    # sanitize the locs
    rlocs[rlocs < 0] = 0
    xlocs = rlocs[:, :, 0, :]
    xlocs[xlocs >= sz[1]] = sz[1] - 1
    rlocs[:, :, 0, :] = xlocs
    ylocs = rlocs[:, :, 1, :]
    ylocs[ylocs >= sz[0]] = sz[0] - 1
    rlocs[:, :, 1, :] = ylocs
    return rlocs


# In[ ]:

def gen_moved_neg_samples(bout, locs, conf, nsamples=10, min_len=8):
    # Add same x and y to locs

    min_len = old_div(float(min_len), 2)
    max_len = 2 * min_len
    r_locs = np.zeros(locs.shape + (nsamples,))
    #     sz = (np.array(bout.shape[1:3])-1)*conf.rescale*conf.pool_scale
    sz = np.array(bout.shape[1:3]) - 1

    for curi in range(locs.shape[0]):
        rx = np.round(np.random.rand(nsamples) * (max_len - min_len) + min_len) * np.sign(np.random.rand(nsamples) - 0.5)
        ry = np.round(np.random.rand(nsamples) * (max_len - min_len) + min_len) * np.sign(np.random.rand(nsamples) - 0.5)

        r_locs[curi, :, 0, :] = locs[curi, :, 0, np.newaxis] + rx
        r_locs[curi, :, 1, :] = locs[curi, :, 1, np.newaxis] + ry

    # sanitize the locs
    r_locs[r_locs < 0] = 0
    x_locs = r_locs[:, :, 0, :]
    x_locs[x_locs >= sz[1]] = sz[1] - 1
    r_locs[:, :, 0, :] = x_locs
    y_locs = r_locs[:, :, 1, :]
    y_locs[y_locs >= sz[0]] = sz[0] - 1
    r_locs[:, :, 1, :] = y_locs
    return r_locs


# In[ ]:

def gen_n_moved_neg_samples(locs, conf, min_len=8):
    # Move a random number of points.
    min_len = float(min_len)
    max_len = 2 * min_len
    min_len = 0

    r_locs = copy.deepcopy(locs)
    sz = conf.imsz

    for cur_i in range(locs.shape[0]):
        cur_n = np.random.randint(conf.n_classes)
        for rand_point in np.random.choice(conf.n_classes, size=[cur_n, ], replace=False):
            rx = np.round(np.random.rand() * (max_len - min_len) + min_len) * np.sign(np.random.rand() - 0.5)
            ry = np.round(np.random.rand() * (max_len - min_len) + min_len) * np.sign(np.random.rand() - 0.5)

            r_locs[cur_i, rand_point, 0] = locs[cur_i, rand_point, 0] + rx
            r_locs[cur_i, rand_point, 1] = locs[cur_i, rand_point, 1] + ry

    # sanitize the locs
    r_locs[r_locs < 0] = 0
    x_locs = r_locs[:, :, 0]
    x_locs[x_locs >= sz[1]] = sz[1] - 1
    r_locs[:, :, 0] = x_locs
    y_locs = r_locs[:, :, 1]
    y_locs[y_locs >= sz[0]] = sz[0] - 1
    r_locs[:, :, 1] = y_locs
    return r_locs


# In[ ]:

def gen_neg_samples(locs, conf, minlen=8):
    return gen_n_moved_neg_samples(locs, conf, minlen)
