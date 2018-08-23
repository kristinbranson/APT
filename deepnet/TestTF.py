from __future__ import print_function

# coding: utf-8

# In[1]:

import tensorflow as tf

import os
import sys
caffenetpath = '/home/mayank/work/tensorflow/caffe-tensorflow'
sys.path.append(caffenetpath)
import pawCaffe
# import caffenet

from scipy import misc
import numpy as np
jj = misc.imread('/home/mayank/work/quackNN/cache/images/im_1.png')

# import matplotlib.pyplot as plt
# plt.imshow(jj)
# plt.show()


jj = np.reshape(jj,[1,128,128,3])
print(jj.shape)
# kk = tf.convert_to_tensor(jj,dtype = np.float32)
# kk.get_shape()

vv = tf.placeholder("float",shape=[1,128,128,3])
gg = tf.placeholder("float",shape=[1,2])

net = pawCaffe.TrainValNet({'data':vv})
# ll = misc.imresize(jj,[256 ,256])
# kk = tf.convert_to_tensor(ll,dtype = np.float32)
pred = net.get_output()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, gg))



# In[2]:

kk = jj.astype('float')-33
labels = np.array([[0,1]])
zz= tf.all_variables()
print(zz[0].name)
ii = tf.gradients(cost,zz[0])
print(ii)
with tf.Session() as sesh:
    net.load('/home/mayank/work/tensorflow/caffe-tensorflow/pawCaffe.npy',sesh)
    output = sesh.run([cost,
                       net.layers['conv1'],
                       net.layers['pool1'],
                       ii[0]],
                       feed_dict={vv:kk,gg:labels})
print(len(output))


# In[3]:

tt1 = tf.placeholder('float',shape=[1,2])
tt2 = tf.placeholder('float',shape=[1,2])
costt = tf.nn.softmax_cross_entropy_with_logits(tt1,tt2)
labelst = np.array([[0.,1.]])
predt = np.array([[0.,1.]])
with tf.Session() as sesh:
    outt = sesh.run(costt,feed_dict={tt1:predt,tt2:labelst})

outt    

