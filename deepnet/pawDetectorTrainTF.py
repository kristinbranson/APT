
# coding: utf-8

# In[ ]:

'''
Mayank Jan 12 2016
Paw detector modified from:
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from past.utils import old_div
import tensorflow as tf

import os,sys
sys.path.append('/home/mayank/work/caffe/python')

import caffe
import lmdb
import caffe.proto.caffe_pb2

from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


# In[ ]:

# Parameters
learning_rate = 0.0005
training_iters = 20000
batch_size = 128
display_step = 30

# Network Parameters
n_input = 128 # patch size
n_classes = 2 # 
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
keep_prob = tf.placeholder(tf.float32) # dropout(keep probability)
x = tf.placeholder(tf.float32, [None, n_input,n_input,3])
y = tf.placeholder(tf.float32, [None, n_classes])

x_conv = tf.placeholder(tf.float32, [1, None,None,3])


val_env = lmdb.open('/home/mayank/work/ChenCode/cache/paw/LMDB_val/',readonly=True)
val_txn = val_env.begin()
val_cursor = val_txn.cursor()
lmdb_env = lmdb.open('/home/mayank/work/ChenCode/cache/paw/LMDB_train/',readonly=True)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()


# In[ ]:

def readLMDB(cursor,num,n_input,n_classes):
    images = np.zeros((num,n_input,n_input,3))
    labels = np.zeros((num,n_classes))
    datum = caffe.proto.caffe_pb2.Datum()

#     print(images.shape)
    for ndx in range(num):
        if not next(cursor):
            cursor.first()
#             print('restarting at %d' % ndx)
            
        value = cursor.value()
        datum.ParseFromString(value)
        curlabel = datum.label
        data = caffe.io.datum_to_array(datum)
        data = np.transpose(data,(1,2,0))
        images[ndx,:,:,:] = data
        labels[ndx,curlabel] = 1
    return images,labels
        


# In[ ]:

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(
                l_input, w, strides=[1, 1, 1, 1], padding='SAME')
            ,b), 
        name=name)

def max_pool(name, l_input, k,s):
    return tf.nn.max_pool(
        l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], 
        padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(
        l_input, lsize, bias=1.0, alpha=0.0001 , beta=0.75, 
        name=name)

def paw_net(_X, _weights, _biases, _dropout):
    
    # L1
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=3,s=2)
    norm1 = norm('norm1', pool1, lsize=2)

    #L2
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=3,s=2)
    norm2 = norm('norm2', pool2, lsize=4)

    # L3,L4,L5
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
  
    # Reshape conv5 output to fit dense layer input
    fc6 = tf.reshape(conv5, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    fc6 = tf.nn.relu(tf.matmul(fc6, _weights['wd1']) + _biases['bd1'], name='fc6') 
    fc6 = tf.nn.dropout(fc6, _dropout)

    
    fc7 = tf.nn.relu(tf.matmul(fc6, _weights['wd2']) + _biases['bd2'], name='fc6') 
    fc7 = tf.nn.dropout(fc7, _dropout)
    
    # Output, class prediction
    out = tf.matmul(fc7, _weights['out']) + _biases['out']
    
    layers = {'conv1': conv1,'pool1':pool1,'norm1':norm1,
              'conv2': conv2,'pool2':pool2,'norm2':norm2,
              'conv3':conv3, 'conv4':conv4,'conv5':conv5,
              'fc6':fc6,'fc7':fc7,'fc8':out
             }

    return out,layers


# In[ ]:

def paw_net_conv(_X, _weights, _biases,):
    
    # L1
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=3,s=2)
    norm1 = norm('norm1', pool1, lsize=2)

    #L2
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=3,s=2)
    norm2 = norm('norm2', pool2, lsize=4)

    # L3,L4,L5
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
  
    # Reshape conv5 output to fit dense layer input
#     fc6 = tf.reshape(conv5, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
#     fc6 = tf.nn.relu(tf.matmul(fc6, _weights['wd1']) + _biases['bd1'], name='fc6') 
#     fc6 = tf.nn.dropout(fc6, _dropout)

    
#     fc7 = tf.nn.relu(tf.matmul(fc6, _weights['wd2']) + _biases['bd2'], name='fc6') 
#     fc7 = tf.nn.dropout(fc7, _dropout)
    
#     # Output, class prediction
#     out = tf.matmul(fc7, _weights['out']) + _biases['out']
  
    conv6 = conv2d('conv6',conv5,_weights['wd_mod1'],_biases['bd1'])
    conv7 = conv2d('conv7',conv6,_weights['wd_mod2'],_biases['bd2'])
    out = tf.nn.bias_add(tf.nn.conv2d(
                conv7, _weights['wd_mod3'], 
                strides=[1, 1, 1, 1], padding='SAME'),_biases['out'])

    
    conv2d('conv8',conv7,_weights['wd_mod3'],_biases['out'])
    
    layers = {'conv1': conv1,'pool1':pool1,'norm1':norm1,
              'conv2': conv2,'pool2':pool2,'norm2':norm2,
              'conv3':conv3, 'conv4':conv4,'conv5':conv5,
              'conv6':conv6, 'conv7':conv7,'conv8':out,
             }

    return out,layers


# In[ ]:

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 48],stddev=0.01)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 48, 128],stddev=0.01)),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'wc5': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'wd1': tf.Variable(tf.random_normal([32*32*128, 1024],stddev=0.005)),
    'wd2': tf.Variable(tf.random_normal([1024, 1024],stddev=0.005)),
    'out': tf.Variable(tf.random_normal([1024, n_classes],stddev=0.01))
}
biases = {
    'bc1': tf.Variable(tf.zeros([48])),
    'bc2': tf.Variable(tf.ones([128])),
    'bc3': tf.Variable(tf.ones([128])),
    'bc4': tf.Variable(tf.ones([128])),
    'bc5': tf.Variable(tf.ones([128])),
    'bd1': tf.Variable(tf.ones([1024])),
    'bd2': tf.Variable(tf.ones([1024])),
    'out': tf.Variable(tf.zeros([n_classes]))
}


# In[ ]:

# Store layers weight & bias
weights_conv = {
    'wc1': weights['wc1'],
    'wc2': weights['wc2'],
    'wc3': weights['wc3'],
    'wc4': weights['wc4'],
    'wc5': weights['wc5'],
    'wd_mod1': tf.reshape(weights['wd1'],[32,32,128,1024]),
    'wd_mod2': tf.reshape(weights['wd2'],[1,1,1024,1024]),
    'wd_mod3': tf.reshape(weights['out'],[1,1,1024,2])
}


# In[ ]:

# Construct model
[pred,layers] = paw_net(x, weights, biases, keep_prob)

[pred_conv,layers_conv] = paw_net_conv(x_conv, weights_conv, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# In[ ]:

# ***** train a detector *****
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        batch_xs, batch_ys = readLMDB(lmdb_cursor,batch_size,n_input,n_classes)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
#         acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
#         loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
#         print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        if step % display_step == 0:
            # Calculate batch accuracy
            val_xs, val_ys = readLMDB(val_cursor,batch_size*4,n_input,n_classes)
            acc = sess.run(accuracy, feed_dict={x: val_xs, y: val_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: val_xs, y: val_ys, keep_prob: 1.})
            print("**** Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Validation Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
   


# In[ ]:

# **** Compare a plain detector to convolution one *****

import cv2

cap = cv2.VideoCapture('/home/mayank/Dropbox/AdamVideos/3rd_T/20140501/M119_20140501_v006/movie_comb.avi')


# In[ ]:

# Capture frame-by-frame
ret, frame = cap.read()

frame = frame[:,0:352,:]
print(frame.shape)
# Display the resulting frame
plt.imshow(frame)


# In[ ]:

sess = tf.InteractiveSession()
sess.run(init)


# In[ ]:

sess.close()


# In[ ]:

frame = frame.astype('float32')
aa = 0
bb = 55
sta = aa*4+1
stb = bb*4+1
patch = frame[np.newaxis,sta:sta+128,stb:stb+128,:]
plt.imshow(old_div(frame[sta:sta+128,stb:stb+128,:],255))
frame_feed = frame[np.newaxis,:]
label = np.array([[0.,1.]]).astype('float32')


out = sess.run([pred,pred_conv],feed_dict=
               {x:patch-33,y:label,
                x_conv:frame_feed-33,
               keep_prob:1.})

np.set_printoptions(precision=6)
print(out[0])
jja = aa+15
jjb = bb+15
print([jja,jjb])
print(out[1][:,jja,jjb,:])

