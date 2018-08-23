
# coding: utf-8

# In[1]:

'''
Mayank Jan 12 2016
Paw detector modified from:
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
import tensorflow as tf
import os,sys
# import caffe
# import lmdb
# import caffe.proto.caffe_pb2
# from caffe.io import datum_to_array
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cv2
import tempfile
import copy

from batch_norm import batch_norm
import myutils
import PoseTools
import localSetup

# def conv2d(name, l_input, w, b):
#     return tf.nn.relu(
#         tf.nn.bias_add(
#             tf.nn.conv2d(
#                 l_input, w, strides=[1, 1, 1, 1], padding='SAME')
#             ,b), 
#         name=name)

def max_pool(name, l_input, k,s):
    return tf.nn.max_pool(
        l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], 
        padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(
        l_input, lsize, bias=1.0, alpha=0.0001 , beta=0.75, 
        name=name)

def upscale(name,l_input,sz):
    l_out = tf.image.resize_nearest_neighbor(l_input,sz,name=name)
    return l_out

# def initNetConvWeights(conf):
#     # Store layers weight & bias
#     nfilt = conf.nfilt
#     nfcfilt = conf.nfcfilt
#     n_classes = conf.n_classes
#     rescale = conf.rescale
#     pool_scale = conf.pool_scale
    
# #     sz5 = int(math.ceil(conf.psz/rescale/pool_scale))
#     sz5 = conf.psz
#     weights = {
#         'base0': initBaseWeights(nfilt),
#         'base1': initBaseWeights(nfilt),
#         'base2': initBaseWeights(nfilt),       
#         'wd1': tf.Variable(tf.random_normal([sz5,sz5,conf.numscale*nfilt,nfcfilt],stddev=0.005)),
#         'wd2': tf.Variable(tf.random_normal([1,1,nfcfilt, nfcfilt],stddev=0.005)),
#         'wd3': tf.Variable(tf.random_normal([1,1,nfcfilt, n_classes],stddev=0.01)),
#         'bd1': tf.Variable(tf.ones([nfcfilt])),
#         'bd2': tf.Variable(tf.ones([nfcfilt])),
#         'bd3': tf.Variable(tf.zeros([n_classes]))
#     }
#     return weights

# def initBaseWeights(nfilt):
    
#     weights = {
#     'wc1': tf.Variable(tf.random_normal([5, 5, 1, 48],stddev=0.01)),
#     'wc2': tf.Variable(tf.random_normal([3, 3, 48, nfilt],stddev=0.01)),
#     'wc3': tf.Variable(tf.random_normal([3, 3, nfilt, nfilt],stddev=0.01)),
#     'wc4': tf.Variable(tf.random_normal([3, 3, nfilt, nfilt],stddev=0.01)),
#     'wc5': tf.Variable(tf.random_normal([3, 3, nfilt, nfilt],stddev=0.01)),
#     'bc1': tf.Variable(tf.zeros([48])),
#     'bc2': tf.Variable(tf.ones([nfilt])),
#     'bc3': tf.Variable(tf.ones([nfilt])),
#     'bc4': tf.Variable(tf.ones([nfilt])),
#     'bc5': tf.Variable(tf.ones([nfilt]))
#     }
#     return weights

# def net_multi_base(X,_weights):
    
#     conv1 = conv2d('conv1', X, _weights['wc1'], _weights['bc1'])
#     pool1 = max_pool('pool1', conv1, k=3,s=2)
#     norm1 = norm('norm1', pool1, lsize=2)
#     conv2 = conv2d('conv2', norm1, _weights['wc2'], _weights['bc2'])
#     pool2 = max_pool('pool2', conv2, k=3,s=2)
#     norm2 = norm('norm2', pool2, lsize=4)
#     conv3 = conv2d('conv3', norm2, _weights['wc3'], _weights['bc3'])
#     conv4 = conv2d('conv4', conv3, _weights['wc4'], _weights['bc4'])
#     conv5 = conv2d('conv5', conv4, _weights['wc5'], _weights['bc5'])
#     out_dict = {'conv1':conv1,'conv2':conv2,'conv3':conv3,
#                 'conv4':conv4,'conv5':conv5,'pool1':pool1,
#                 'pool2':pool2,'norm1':norm1,'norm2':norm2,
#                }
#     return conv5, out_dict

def net_multi_base_named(X,nfilt,doBatchNorm,trainPhase,pool_stride,pool_size,conf):
    inDim = X.get_shape()[3]
    with tf.variable_scope('layer1'):
        conv1 = conv_relu(X,[5, 5, inDim, 48],0.01,0,doBatchNorm,trainPhase)
        if hasattr(conf,'num_pools'):
            if conf.num_pools > 0:
                pool1 = max_pool('pool1', conv1, k=pool_size,s=pool_stride)
            else:
                pool1 = conv1
        else:
            pool1 = max_pool('pool1', conv1, k=pool_size, s=pool_stride)
        norm1 = norm('norm1', pool1, lsize=2)
    
    with tf.variable_scope('layer2'):
        conv2 = conv_relu(norm1,[3,3,48,nfilt],0.01,1,doBatchNorm,trainPhase)
        if hasattr(conf,'num_pools'):
            if conf.num_pools > 1:
                pool2 = max_pool('pool2', conv2, k=pool_size, s=pool_stride)
            else:
                pool2 = conv2
        else:
            pool2 = max_pool('pool2', conv2, k=pool_size, s=pool_stride)
        norm2 = norm('norm2', pool2, lsize=4)

    with tf.variable_scope('layer3'):
        conv3 = conv_relu(norm2,[3,3,nfilt,nfilt],0.01,1,doBatchNorm,trainPhase)
    with tf.variable_scope('layer4'):
        conv4 = conv_relu(conv3,[3,3,nfilt,nfilt],0.01,1,doBatchNorm,trainPhase)
    with tf.variable_scope('layer5'):
        conv5 = conv_relu(conv4,[3,3,nfilt,nfilt],0.01,1,doBatchNorm,trainPhase)
        
    out_dict = {'conv1':conv1,'conv2':conv2,'conv3':conv3,
                'conv4':conv4,'conv5':conv5,'pool1':pool1,
                'pool2':pool2,'norm1':norm1,'norm2':norm2,
               }
    return conv5,out_dict


def net_multi_base_named_dilated(X, nfilt, doBatchNorm, trainPhase, pool_stride, pool_size, conf):
    inDim = X.get_shape()[3]
    with tf.variable_scope('layer1'):
        conv1 = conv_relu(X, [5, 5, inDim, 48], 0.01, 0, doBatchNorm, trainPhase)
        norm1 = norm('norm1', conv1, lsize=2)

    with tf.variable_scope('layer2'):
        weights = tf.get_variable("weights", [3,3,48,nfilt],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", nfilt,
                                 initializer=tf.constant_initializer(1))
        conv2 = tf.nn.convolution(norm1, weights,
                            strides=[1, 1], padding='SAME',dilation_rate=[4,4])
        if doBatchNorm:
            conv2 = batch_norm(conv2, trainPhase)
        conv2 = tf.nn.relu(conv2 + biases)
        norm2 = norm('norm2',conv2 , lsize=4)

    with tf.variable_scope('layer3'):
        weights = tf.get_variable("weights", [3,3,nfilt,nfilt],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases",nfilt,
                                 initializer=tf.constant_initializer(1))
        conv3 = tf.nn.convolution(norm2, weights,
                                  strides=[1, 1], padding='SAME', dilation_rate=[2, 2])
        if doBatchNorm:
            conv3 = batch_norm(conv3, trainPhase)
        conv3 = tf.nn.relu(conv3 + biases)

    with tf.variable_scope('layer4'):
        conv4 = conv_relu(conv3, [3, 3, nfilt, nfilt], 0.01, 1, doBatchNorm, trainPhase)
    with tf.variable_scope('layer5'):
        conv5 = conv_relu(conv4, [3, 3, nfilt, nfilt], 0.01, 1, doBatchNorm, trainPhase)

    out_dict = {'conv1': conv1, 'conv2': conv2, 'conv3': conv3,
                'conv4': conv4, 'conv5': conv5, 'norm1': norm1, 'norm2': norm2,
                }
    return conv5, out_dict


def net_multi_conv(X0,X1,X2,_dropout,conf,doBatchNorm,trainPhase):
    imsz = conf.imsz; rescale = conf.rescale
    pool_scale = conf.pool_scale
    nfilt = conf.nfilt
    pool_stride = conf.pool_stride
    pool_size = conf.pool_size

    #     conv5_0,base_dict_0 = net_multi_base(X0,_weights['base0'])
    #     conv5_1,base_dict_1 = net_multi_base(X1,_weights['base1'])
    #     conv5_2,base_dict_2 = net_multi_base(X2,_weights['base2'])
    if conf.dilation_rate is 4:
        net_to_use = net_multi_base_named_dilated
    else:
        net_to_use = net_multi_base_named

    with tf.variable_scope('scale0'):
        conv5_0,base_dict_0 = net_to_use(X0,nfilt,doBatchNorm,trainPhase,
                                                   pool_stride,pool_size,conf)
    with tf.variable_scope('scale1'):
        conv5_1,base_dict_1 = net_to_use(X1,nfilt,doBatchNorm,trainPhase,
                                                   pool_stride,pool_size,conf)
    with tf.variable_scope('scale2'):
        conv5_2,base_dict_2 = net_to_use(X2,nfilt,doBatchNorm,trainPhase,
                                                   pool_stride,pool_size,conf)

    sz0 = int(math.ceil(float(imsz[0])/pool_scale/rescale))
    sz1 = int(math.ceil(float(imsz[1])/pool_scale/rescale))
    conv5_1_up = upscale('5_1',conv5_1,[sz0,sz1])
    conv5_2_up = upscale('5_2',conv5_2,[sz0,sz1])

    # crop lower res layers to match higher res size
    conv5_0_sz = tf.Tensor.get_shape(conv5_0).as_list()
    conv5_1_sz = tf.Tensor.get_shape(conv5_1_up).as_list()
    crop_0 = int(old_div((sz0-conv5_0_sz[1]),2))
    crop_1 = int(old_div((sz1-conv5_0_sz[2]),2))

    curloc = [0,crop_0,crop_1,0]
    patchsz = tf.to_int32([-1,conv5_0_sz[1],conv5_0_sz[2],-1])
    conv5_1_up = tf.slice(conv5_1_up,curloc,patchsz)
    conv5_2_up = tf.slice(conv5_2_up,curloc,patchsz)
    conv5_1_final_sz = tf.Tensor.get_shape(conv5_1_up).as_list()
#     print("Initial lower res layer size %s"%(', '.join(map(str,conv5_1_sz))))
#     print("Initial higher res layer size %s"%(', '.join(map(str,conv5_0_sz))))
#     print("Crop start lower res layer at %s"%(', '.join(map(str,curloc))))
#     print("Final size of lower res layer %s"%(', '.join(map(str,conv5_1_final_sz))))


    conv5_cat = tf.concat([conv5_0,conv5_1_up,conv5_2_up],3)
    
    # Reshape conv5 output to fit dense layer input
#     conv6 = conv2d('conv6',conv5_cat,_weights['wd1'],_weights['bd1']) 
#     conv6 = tf.nn.dropout(conv6,_dropout)
#     conv7 = conv2d('conv7',conv6,_weights['wd2'],_weights['bd2']) 
#     conv7 = tf.nn.dropout(conv7,_dropout)

    with tf.variable_scope('layer6'):
        if hasattr(conf, 'dilation_rate'):
            dilation_rate = [conf.dilation_rate, conf.dilation_rate]
        else:
            dilation_rate = [1, 1]
        weights = tf.get_variable("weights", [conf.psz,conf.psz,conf.numscale*nfilt,conf.nfcfilt],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", conf.nfcfilt,
                                 initializer=tf.constant_initializer(1))
        conv6 = tf.nn.convolution(conv5_cat, weights,
                            strides=[1, 1], padding='SAME',dilation_rate=dilation_rate)
        if doBatchNorm:
            conv6 = batch_norm(conv6, trainPhase)
        conv6 = tf.nn.relu(conv6 + biases)
        conv6 = tf.nn.dropout(conv6,_dropout,
                          [conf.batch_size,1,1,conf.nfcfilt])

    with tf.variable_scope('layer7'):
        conv7 = conv_relu(conv6,[1,1,conf.nfcfilt,conf.nfcfilt],
                          0.005,1,doBatchNorm,trainPhase) 
        # if not doBatchNorm:
        conv7 = tf.nn.dropout(conv7,_dropout,
                              [conf.batch_size,1,1,conf.nfcfilt])

# Output, class prediction
#     out = tf.nn.bias_add(tf.nn.conv2d(
#             conv7, _weights['wd3'], 
#             strides=[1, 1, 1, 1], padding='SAME'),_weights['bd3'])

    with tf.variable_scope('layer8'):
        l8_weights = tf.get_variable("weights", [1,1,conf.nfcfilt,conf.n_classes],
            initializer=tf.random_normal_initializer(stddev=0.01))
        l8_biases = tf.get_variable("biases", conf.n_classes,
            initializer=tf.constant_initializer(0))
        out = tf.nn.conv2d(conv7, l8_weights,
            strides=[1, 1, 1, 1], padding='SAME') + l8_biases
#   No batch norm for the output layer.

    out_dict = {'base_dict_0':base_dict_0,
                'base_dict_1':base_dict_1,
                'base_dict_2':base_dict_2,
                'conv6':conv6,
                'conv7':conv7,
               }
    
    return out,out_dict


def net_multi_conv_vgg(X0, X1, X2, _dropout, conf, doBatchNorm, trainPhase):
    imsz = conf.imsz
    rescale = conf.rescale
    pool_scale = conf.pool_scale
    nfilt = conf.nfilt
    pool_stride = conf.pool_stride
    pool_size = conf.pool_size

    with tf.variable_scope('scale0'):
        conv5_0, base_dict_0 = net_multi_base_named(X0, nfilt, doBatchNorm, trainPhase, pool_stride, pool_size)
    with tf.variable_scope('scale1'):
        conv5_1, base_dict_1 = net_multi_base_named(X1, nfilt, doBatchNorm, trainPhase, pool_stride, pool_size)
    with tf.variable_scope('scale2'):
        conv5_2, base_dict_2 = net_multi_base_named(X2, nfilt, doBatchNorm, trainPhase, pool_stride, pool_size)

    sz0 = int(math.ceil(float(imsz[0]) / pool_scale / rescale))
    sz1 = int(math.ceil(float(imsz[1]) / pool_scale / rescale))
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
    conv5_1_final_sz = tf.Tensor.get_shape(conv5_1_up).as_list()

    conv5_cat = tf.concat([conv5_0, conv5_1_up, conv5_2_up], 3)

    with tf.variable_scope('layer6'):
        conv6 = conv_relu(conv5_cat,
                          [conf.psz, conf.psz, conf.numscale * nfilt, conf.nfcfilt],
                          0.005, 1, doBatchNorm, trainPhase)
        # if not doBatchNorm:
        conv6 = tf.nn.dropout(conv6, _dropout,
                              [conf.batch_size, 1, 1, conf.nfcfilt])

    with tf.variable_scope('layer7'):
        conv7 = conv_relu(conv6, [1, 1, conf.nfcfilt, conf.nfcfilt],
                          0.005, 1, doBatchNorm, trainPhase)
        # if not doBatchNorm:
        conv7 = tf.nn.dropout(conv7, _dropout,
                              [conf.batch_size, 1, 1, conf.nfcfilt])

    # Output, class prediction
    #     out = tf.nn.bias_add(tf.nn.conv2d(
    #             conv7, _weights['wd3'],
    #             strides=[1, 1, 1, 1], padding='SAME'),_weights['bd3'])

    with tf.variable_scope('layer8'):
        l8_weights = tf.get_variable("weights", [1, 1, conf.nfcfilt, conf.n_classes],
                                     initializer=tf.random_normal_initializer(stddev=0.01))
        l8_biases = tf.get_variable("biases", conf.n_classes,
                                    initializer=tf.constant_initializer(0))
        out = tf.nn.conv2d(conv7, l8_weights,
                           strides=[1, 1, 1, 1], padding='SAME') + l8_biases
    # No batch norm for the output layer.

    out_dict = {'base_dict_0': base_dict_0,
                'base_dict_1': base_dict_1,
                'base_dict_2': base_dict_2,
                'conv6': conv6,
                'conv7': conv7,
                }

    return out, out_dict


def net_multi_conv_reg(X0,X1,X2,_dropout,conf,doBatchNorm,trainPhase):
    imsz = conf.imsz; rescale = conf.rescale
    pool_scale = conf.pool_scale
    nfilt = conf.nfilt
    
    #     conv5_0,base_dict_0 = net_multi_base(X0,_weights['base0'])
    #     conv5_1,base_dict_1 = net_multi_base(X1,_weights['base1'])
    #     conv5_2,base_dict_2 = net_multi_base(X2,_weights['base2'])
    with tf.variable_scope('scale0'):
        conv5_0,base_dict_0 = net_multi_base_named(X0,nfilt,doBatchNorm,trainPhase)
    with tf.variable_scope('scale1'):
        conv5_1,base_dict_1 = net_multi_base_named(X1,nfilt,doBatchNorm,trainPhase)
    with tf.variable_scope('scale2'):
        conv5_2,base_dict_2 = net_multi_base_named(X2,nfilt,doBatchNorm,trainPhase)

    sz0 = int(math.ceil(float(imsz[0])/pool_scale/rescale))
    sz1 = int(math.ceil(float(imsz[1])/pool_scale/rescale))
    conv5_1_up = upscale('5_1',conv5_1,[sz0,sz1])
    conv5_2_up = upscale('5_2',conv5_2,[sz0,sz1])

    # crop lower res layers to match higher res size
    conv5_0_sz = tf.Tensor.get_shape(conv5_0).as_list()
    conv5_1_sz = tf.Tensor.get_shape(conv5_1_up).as_list()
    crop_0 = int(old_div((sz0-conv5_0_sz[1]),2))
    crop_1 = int(old_div((sz1-conv5_0_sz[2]),2))

    curloc = [0,crop_0,crop_1,0]
    patchsz = tf.to_int32([-1,conv5_0_sz[1],conv5_0_sz[2],-1])
    conv5_1_up = tf.slice(conv5_1_up,curloc,patchsz)
    conv5_2_up = tf.slice(conv5_2_up,curloc,patchsz)
    conv5_1_final_sz = tf.Tensor.get_shape(conv5_1_up).as_list()
#     print("Initial lower res layer size %s"%(', '.join(map(str,conv5_1_sz))))
#     print("Initial higher res layer size %s"%(', '.join(map(str,conv5_0_sz))))
#     print("Crop start lower res layer at %s"%(', '.join(map(str,curloc))))
#     print("Final size of lower res layer %s"%(', '.join(map(str,conv5_1_final_sz))))


    conv5_cat = tf.concat([conv5_0,conv5_1_up,conv5_2_up],3)
    
    # Reshape conv5 output to fit dense layer input
#     conv6 = conv2d('conv6',conv5_cat,_weights['wd1'],_weights['bd1']) 
#     conv6 = tf.nn.dropout(conv6,_dropout)
#     conv7 = conv2d('conv7',conv6,_weights['wd2'],_weights['bd2']) 
#     conv7 = tf.nn.dropout(conv7,_dropout)

    with tf.variable_scope('layer6'):
        conv6 = conv_relu(conv5_cat,
                         [conf.psz,conf.psz,conf.numscale*nfilt,conf.nfcfilt],
                          0.005,1,doBatchNorm,trainPhase) 
        if not doBatchNorm:
            conv6 = tf.nn.dropout(conv6,_dropout,
                              [conf.batch_size,1,1,conf.nfcfilt])

    with tf.variable_scope('layer7'):
        conv7 = conv_relu(conv6,[1,1,conf.nfcfilt,conf.nfcfilt],
                          0.005,1,doBatchNorm,trainPhase) 
        if not doBatchNorm:
            conv7 = tf.nn.dropout(conv7,_dropout,
                                  [conf.batch_size,1,1,conf.nfcfilt])

# Output, class prediction
#     out = tf.nn.bias_add(tf.nn.conv2d(
#             conv7, _weights['wd3'], 
#             strides=[1, 1, 1, 1], padding='SAME'),_weights['bd3'])

    with tf.variable_scope('layer8'):
        l8_weights = tf.get_variable("weights", [1,1,conf.nfcfilt,conf.n_classes],
            initializer=tf.random_normal_initializer(stddev=0.01))
        l8_biases = tf.get_variable("biases", conf.n_classes,
            initializer=tf.constant_initializer(0))
        out = tf.nn.conv2d(conv7, l8_weights,
            strides=[1, 1, 1, 1], padding='SAME') + l8_biases

    with tf.variable_scope('layer8_x'):
        l8x_weights = tf.get_variable("weights", [1,1,conf.nfcfilt,conf.n_classes],
            initializer=tf.random_normal_initializer(stddev=0.01))
        l8x_biases = tf.get_variable("biases", conf.n_classes,
            initializer=tf.constant_initializer(0))
        out_x = tf.nn.conv2d(conv7, l8x_weights,
            strides=[1, 1, 1, 1], padding='SAME') + l8x_biases

    with tf.variable_scope('layer8_y'):
        l8y_weights = tf.get_variable("weights", [1,1,conf.nfcfilt,conf.n_classes],
            initializer=tf.random_normal_initializer(stddev=0.01))
        l8y_biases = tf.get_variable("biases", conf.n_classes,
            initializer=tf.constant_initializer(0))
        out_y = tf.nn.conv2d(conv7, l8y_weights,
            strides=[1, 1, 1, 1], padding='SAME') + l8y_biases
        
    out_dict = {'base_dict_0':base_dict_0,
                'base_dict_1':base_dict_1,
                'base_dict_2':base_dict_2,
                'conv6':conv6,
                'conv7':conv7,
               }
    
    return out, out_x, out_y, out_dict



def createPlaceHolders(imsz,rescale,scale,pool_scale,n_classes,inDim=1):
#     imsz = conf.imsz
    # tf Graph input
    keep_prob = tf.placeholder(tf.float32,name='dropout') # dropout(keep probability)
    x0 = tf.placeholder(tf.float32, [None, 
                                     old_div(imsz[0],rescale),
                                     old_div(imsz[1],rescale),inDim],name='x0')
    x1 = tf.placeholder(tf.float32, [None, 
                                     imsz[0]//scale//rescale,
                                     imsz[1]//scale//rescale,inDim],name='x1')
    x2 = tf.placeholder(tf.float32, [None, 
                                     imsz[0]//scale//scale//rescale,
                                     imsz[1]//scale//scale//rescale,inDim],name='x2')

    lsz0,lsz1 = findPredSize(imsz,rescale,pool_scale)
    y = tf.placeholder(tf.float32, [None, lsz0,lsz1,n_classes],'limg')
    return x0,x1,x2,y,keep_prob

def findPredSize(imsz,rescale,pool_scale):
    lsz0 = int(math.ceil(float(imsz[0])/pool_scale/rescale))
    lsz1 = int(math.ceil(float(imsz[1])/pool_scale/rescale))
    return lsz0, lsz1

def conv_relu(X, kernel_shape, conv_std,bias_val,doBatchNorm,trainPhase):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer(stddev=conv_std))
    biases = tf.get_variable("biases", kernel_shape[-1],
        initializer=tf.constant_initializer(bias_val))
    conv = tf.nn.conv2d(X, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    if doBatchNorm:
        conv = batch_norm(conv,trainPhase)
    return tf.nn.relu(conv + biases)


def fine_base(X,conf,insize,doBatchNorm,trainPhase):
    fsz = conf.fine_flt_sz
    fine_nfilt = conf.fine_nfilt
    with tf.variable_scope("fine_1"):
        conv1 = conv_relu(X, [fsz, fsz, insize, fine_nfilt],
                          0.05,1,doBatchNorm,trainPhase)
    with tf.variable_scope("fine_2"):
        conv2 = conv_relu(conv1, [fsz, fsz, fine_nfilt, fine_nfilt],
                          0.05,1,doBatchNorm,trainPhase)
    with tf.variable_scope("fine_3"):
        conv3 = conv_relu(conv2, [fsz, fsz, fine_nfilt, old_div(fine_nfilt,2)],
                          0.05,1,doBatchNorm,trainPhase)
    return conv3

    

def fineNetwork(fineIn1_1,fineIn1_2,fineIn2_1,fineIn2_2,
                conf,doBatchNorm,trainPhase):

#     fsz = conf.fine_sz
#     fineIn1_2_up = upscale('fine1_2',fineIn1_2,[fsz,fsz])
#     fineIn2_1_up = upscale('fine2_1',fineIn2_1,[fsz,fsz])
#     fineIn2_2_up = upscale('fine2_2',fineIn2_2,[fsz,fsz])
#     fineIn7_up   = upscale('fine7',fineIn7,[fsz,fsz])
    with tf.variable_scope('1_1'):
        fine1_1 = fine_base(fineIn1_1,conf,48,doBatchNorm,trainPhase)
    with tf.variable_scope('1_2'):
        fine1_2 = fine_base(fineIn1_2,conf,conf.nfilt,doBatchNorm,trainPhase)
    with tf.variable_scope('2_1'):
        fine2_1 = fine_base(fineIn2_1,conf,48,doBatchNorm,trainPhase)
    with tf.variable_scope('2_2'):
        fine2_2 = fine_base(fineIn2_2,conf,conf.nfilt,doBatchNorm,trainPhase)

    fsz = conf.fine_sz
    fine1_2_up = upscale('fine1_2',fine1_2,[fsz,fsz])
    fine2_1_up = upscale('fine2_1',fine2_1,[fsz,fsz])
    fine2_2_up = upscale('fine2_2',fine2_2,[fsz,fsz])
    fineSum = tf.add_n([fine1_1,fine1_2_up,fine2_1_up,fine2_2_up])
#     fineSum = tf.add_n([fine1_1,fine1_2,fine2_1,fine2_2,fine7])
    # for fine apparently adding is better than concatenating!!
#     conv5_cat = tf.concat(3,[fine1_1,fine1_2_up,fine2_1_up,fine2_2_up])
    return fineSum

def fineOut(fineIn1_1,fineIn1_2,fineIn2_1,fineIn2_2,conf,doBatchNorm,trainPhase):
    inter = []
    with tf.variable_scope('fine_siamese') as scope:
        tvar = fineNetwork(fineIn1_1[0], fineIn1_2[0],
                           fineIn2_1[0], fineIn2_2[0],
                           conf, doBatchNorm,
                           trainPhase)
        inter.append(tvar)
        # scope.reuse_variables()
        tf.get_variable_scope().reuse_variables()

        for ndx in range(1,len(fineIn1_1)):
            tvar = fineNetwork(fineIn1_1[ndx], fineIn1_2[ndx],
                               fineIn2_1[ndx], fineIn2_2[ndx],
                               conf, doBatchNorm,
                               trainPhase)
            inter.append(tvar)

    fineLast = []
    for ndx in range(len(fineIn1_1)):
        with tf.variable_scope('point_' + str(ndx)):
            weights = tf.get_variable("weights", [1,1,old_div(conf.fine_nfilt,2),1],
                initializer=tf.random_normal_initializer(stddev=0.05))
            biases = tf.get_variable("biases", 1,
                initializer=tf.constant_initializer(0))
            conv = tf.nn.conv2d(inter[ndx], weights,
                strides=[1, 1, 1, 1], padding='SAME')
            fineLast.append(conv + biases)

    out = tf.concat(fineLast,3)
    return out

