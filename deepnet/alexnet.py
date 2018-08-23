from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[1]:

################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import range
from past.utils import old_div
from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib.request, urllib.parse, urllib.error
from numpy import random

import tensorflow as tf

from caffe_classes import class_names


# In[4]:

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())



def alexnet(x):
    # x = tf.Variable(i)
    net_data = load("bvlc_alexnet.npy").item()

    layers = {}
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    layers['conv1'] = conv1_in
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    layers['lrn1'] = lrn1
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    layers['maxpool1'] = maxpool1

    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    layers['conv2'] = conv2_in
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    layers['lrn2'] = lrn2
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    layers['maxpool2'] = maxpool2
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    maxpool5_re = tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))])
    fc6_in = tf.nn.xw_plus_b(maxpool5_re,fc6W,fc6b)
    fc6 = tf.nn.relu(fc6_in)
#     fc6 = tf.nn.relu_layer(maxpool5_re, fc6W, fc6b)

    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
#     fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    fc7_in = tf.nn.xw_plus_b(fc6,fc7W,fc7b)
    fc7 = tf.nn.relu(fc7_in)

    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    layers['conv3'] = conv3_in
    layers['conv4'] = conv4_in
    layers['conv5'] = conv5_in
    layers['maxpool5'] = maxpool5
    layers['fc6'] = fc6_in
    layers['fc7'] = fc7_in
    layers['fc8'] = fc8
    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc8)
    return prob,layers


# In[5]:

import tensorflow as tf
train_x = np.zeros((1, 227,227,3)).astype(float)
train_y = np.zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]
x = tf.placeholder(tf.float32,[1,227,227,3])

prob,layers = alexnet(x)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# In[48]:

x_dummy = (old_div(np.random.random((1,)+ xdim),255.)).astype(float)
i = x_dummy.copy()
i[0,:,:,:] = (imread("poodle.png")[:,:,:3]).astype(float)
i = i-mean(i)

output = sess.run(prob,feed_dict={x:i})
################################################################################

#Output:

inds = argsort(output)[0,:]
for i in range(5):
    print(class_names[inds[-1-i]], output[0, inds[-1-i]])


# In[7]:

import os
import os.path
import random

vdir = '/media/kabram/My Passport/imagenet/ILSVRC2012_img_val'

vfiles = [os.path.join(vdir,ff) for ff in os.listdir(vdir)]
random.shuffle(vfiles)
print(vfiles[0:3])


# In[8]:

print(list(layers.keys()))
print(layers['fc6'].get_shape().as_list())


# In[100]:

lsel = ['conv2','conv3','conv4','conv5','fc6','fc7']
out = {}
ncount = 5000

vals = {}
outvar = []
for ll in lsel:
    vals[ll] = np.zeros([ncount,]+layers[ll].get_shape().as_list())
    outvar.append(layers[ll])
    
for ndx in range(ncount):
    isz = np.array([227,227])
    inImg = imread(vfiles[ndx])
    if len(inImg.shape)<3:
        inImg = np.tile(inImg[:,:,np.newaxis],[1,1,3])
    i = (inImg[:,:,:3]).astype(float)
    ir = imresize(i,isz)
    out = sess.run(outvar,feed_dict={x:ir[np.newaxis,...]})
    for lndx,ll in enumerate(lsel):
        vals[ll][ndx,...] = out[lndx]


# In[50]:

net_data = load("bvlc_alexnet.npy").item()


# In[71]:

import scipy
import scipy.stats

lname = 'conv3'
print(vals[lname].shape)
fndx = np.random.randint(0,vals[lname].shape[-1])
bb = net_data[lname][1][fndx]
a = vals[lname][...,fndx].flatten()
plt.hist(a,bins=30)
print(a.shape)
[k,p] = scipy.stats.mstats.normaltest(a[:,np.newaxis],0)
print(k,p,bb,fndx)
ss = np.random.randn(5000,1)*10+5
[k,p] = scipy.stats.mstats.normaltest(ss,0)
print(k,p)
plt.show()
plt.hist(ss,bins=30)


# In[112]:

import scipy
import scipy.stats

mm = []
vv = []
bb = []
for lndx,lname in enumerate(lsel):
    nd = vals[lname].shape[-1]
    vv.append(np.zeros([nd,1]))
    mm.append(np.zeros([nd,1]))
    bb.append(np.zeros([nd,1]))
    for ndx in range(nd):
        gg = vals[lname][...,ndx].flatten()[:,newaxis]
        vv[lndx][ndx,0] = np.std(gg)
        mm[lndx][ndx,0] = np.median(gg)
        bb[lndx][ndx,0] = net_data[lname][1][ndx]


# In[103]:

amm = np.concatenate(mm)
avv = np.concatenate(vv)
abb = np.concatenate(bb)
plt.scatter(amm-abb,avv)


# In[85]:

kk = gg.flatten()[:,np.newaxis]
print(kk.shape)
ss = np.var(kk)
print(ss)


# In[104]:

plt.hist(old_div((amm-abb),avv),bins=20)


# In[121]:

kk = np.corrcoef(mm[ndx].flatten()-bb[ndx].flatten(),vv[ndx].flatten())
print(kk)


# In[123]:

fig = figure(figsize=[10,10])
nl = len(mm)
print(lsel)
cc = np.zeros([len(mm),1])
for ndx in range(len(mm)):
    ax = fig.add_subplot(4,old_div(nl,2),ndx+1)
    ax.hist(old_div((mm[ndx]-bb[ndx]),vv[ndx]))
    ax = fig.add_subplot(4,old_div(nl,2),ndx+nl+1)
#     ax.scatter(mm[ndx]-bb[ndx],vv[ndx])
    ax.scatter(mm[ndx],vv[ndx])
    cc[ndx,0] = np.corrcoef(mm[ndx].flatten().flatten(),vv[ndx].flatten())[0,1]
print(cc)


# In[128]:

fig = figure(figsize=[10,5])
print(lsel)
for ndx in range(len(mm)):
    ax = fig.add_subplot(2,old_div(nl,2),ndx+1)
    ax.hist((bb[ndx]))

