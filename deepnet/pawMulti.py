
# coding: utf-8

# In[28]:

'''
Mayank Jan 12 2016
Paw detector modified from:
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
import pawconfig as conf

from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import multiPawTools
import math

import cv2
import matplotlib.animation as manimation
sys.path.append('/home/mayank/work/pyutils')
import myutils
import tempfile
import copy
from convNetBase import *


# In[6]:

def paw_net_multi(X0,X1,X2, _base_weights,_weights, _biases, _dropout):
    
    conv5_0 = paw_net_multi_base(X0,_base_weights['base0'],_dropout)
    conv5_1 = paw_net_multi_base(X1,_base_weights['base1'],_dropout)
    conv5_2 = paw_net_multi_base(X2,_base_weights['base2'],_dropout)

    conv5_cat = tf.concat(3,[conv5_0,conv5_1,conv5_2])
    
    # Reshape conv5 output to fit dense layer input
    fc6 = tf.reshape(conv5_cat, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    fc6 = tf.nn.relu(tf.matmul(fc6, _weights['wd1']) + _biases['bd1'], name='fc6') 
    fc6 = tf.nn.dropout(fc6, _dropout)

    
    fc7 = tf.nn.relu(tf.matmul(fc6, _weights['wd2']) + _biases['bd2'], name='fc6') 
    fc7 = tf.nn.dropout(fc7, _dropout)
    
    # Output, class prediction
    out = tf.matmul(fc7, _weights['out']) + _biases['out']
    
    return out


# In[6]:

def paw_net_multi_conv(X0,X1,X2, _weights,_dropout):
    imsz = conf.imsz
    pool_scale = conf.pool_scale
    rescale = conf.rescale
    conv5_0 = paw_net_multi_base(X0,_weights['base0'])
    conv5_1 = paw_net_multi_base(X1,_weights['base1'])
    conv5_2 = paw_net_multi_base(X2,_weights['base2'])

    sz0 = int(math.ceil(float(imsz[0])/pool_scale/rescale))
    sz1 = int(math.ceil(float(imsz[1])/pool_scale/rescale))
    conv5_1_up = upscale('5_1',conv5_1,[sz0,sz1])
    conv5_2_up = upscale('5_2',conv5_2,[sz0,sz1])
    conv5_cat = tf.concat(3,[conv5_0,conv5_1_up,conv5_2_up])
    
    # Reshape conv5 output to fit dense layer input
    conv6 = conv2d('conv6',conv5_cat,_weights['wd1'],_weights['bd1']) 
    conv6 = tf.nn.dropout(conv6,_dropout)
    conv7 = conv2d('conv7',conv6,_weights['wd2'],_weights['bd2']) 
    conv7 = tf.nn.dropout(conv7,_dropout)
    # Output, class prediction
    out = tf.nn.bias_add(tf.nn.conv2d(
            conv7, _weights['wd3'], 
            strides=[1, 1, 1, 1], padding='SAME'),_weights['bd3'])
    return out


# In[ ]:

def createPlaceHolders():
    imsz = conf.imsz
    # tf Graph input
    keep_prob = tf.placeholder(tf.float32) # dropout(keep probability)
    x0 = tf.placeholder(tf.float32, [None, 
                                     old_div(imsz[0],conf.rescale),
                                     old_div(imsz[1],conf.rescale),1])
    x1 = tf.placeholder(tf.float32, [None, 
                                     imsz[0]/conf.scale/conf.rescale,
                                     imsz[1]/conf.scale/conf.rescale,1])
    x2 = tf.placeholder(tf.float32, [None, 
                                     imsz[0]/conf.scale/conf.scale/conf.rescale,
                                     imsz[1]/conf.scale/conf.scale/conf.rescale,1])

    lsz0 = int(math.ceil(float(imsz[0])/conf.pool_scale/conf.rescale))
    lsz1 = int(math.ceil(float(imsz[1])/conf.pool_scale/conf.rescale))
    y = tf.placeholder(tf.float32, [None, lsz0,lsz1,conf.n_classes])
    return x0,x1,x2,y,keep_prob


# In[ ]:

def multiScaleImages(inImg):
    x0_in = multiPawTools.scaleImages(inImg,conf.rescale)
    x1_in = multiPawTools.scaleImages(x0_in,conf.scale)
    x2_in = multiPawTools.scaleImages(x1_in,conf.scale)
    return x0_in,x1_in,x2_in


# In[ ]:

def train():
    # Parameters
    learning_rate = conf.learning_rate
    training_iters = conf.training_iters
    batch_size = conf.batch_size
    display_step = conf.display_step

    # Network Parameters
    n_input = conf.psz
    n_classes = conf.n_classes # 
    dropout = conf.dropout # Dropout, probability to keep units
    x0,x1,x2,y,keep_prob = createPlaceHolders()
    
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    env = lmdb.open(lmdbfilename, map_size=conf.map_size)
    valenv = lmdb.open(vallmdbfilename, map_size=conf.map_size)
    weights = initNetConvWeights()
    
    with env.begin(write=True) as txn,valenv.begin(write=True) as valtxn:
        train_cursor = txn.cursor()
        val_cursor = valtxn.cursor()

        # Construct model
        pred = paw_net_multi_conv(x0,x1,x2, weights, keep_prob)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.l2_loss(pred- y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            print("Initialized the network")
            step = 1
            # Keep training until reach max iterations
            while step < training_iters:
                batch_xs, locs = multiPawTools.readLMDB(train_cursor,batch_size,n_classes)

                x0_in,x1_in,x2_in = multiScaleImages(batch_xs.transpose([0,2,3,1]))
                labelims = multiPawTools.createLabelImages(locs,
                                           conf.imsz,conf.pool_scale*conf.rescale,
                                           conf.label_blur_rad,1) 
                sess.run(optimizer, 
                         feed_dict={x0: x0_in,
                                    x1: x1_in,
                                    x2: x2_in,
                                    y: labelims, keep_prob: dropout})

                if step % display_step == 0:
                    train_loss = sess.run(cost, feed_dict={x0:x0_in,
                                                     x1:x1_in,
                                                     x2:x2_in,
                                               y: labelims, keep_prob: 1.})
                    
                    numrep = int(old_div(conf.numTest,conf.batch_size))+1
                    acc = 0; loss = 0
                    for rep in range(numrep):
                        val_xs, locs = multiPawTools.readLMDB(val_cursor,batch_size,n_classes)
                        x0_in,x1_in,x2_in = multiScaleImages(val_xs.transpose([0,2,3,1]))

                        labelims = multiPawTools.createLabelImages(locs,
                                                   conf.imsz,conf.pool_scale*conf.rescale,
                                                   conf.label_blur_rad,1)
                        loss += sess.run(cost, feed_dict={x0:x0_in,
                                                         x1:x1_in,
                                                         x2:x2_in,
                                                   y: labelims, keep_prob: 1.})
                    loss = old_div(loss,numrep)
                    print(" Iter " + str(step) + ", Training Loss= " + "{:.6f}".format(train_loss))
                    print(" Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss)) 
                if step % conf.save_step == 0:
                    curoutname = '%s_%d.ckpt'% (conf.outname,step)
                    outfilename = os.path.join(conf.cachedir,curoutname)
                    saver.save(sess,outfilename)
                    print('Saved state to %s' %(outfilename))

                step += 1
            print("Optimization Finished!")
            curoutname = '%s_%d.ckpt'% (conf.outname,step)
            outfilename = os.path.join(conf.cachedir,curoutname)
            saver.save(sess,outfilename)
            print('Saved state to %s' %(outfilename))



# In[ ]:

def initPredSession():
    x0,x1,x2,y,keep_prob = createPlaceHolders()
    weights = initNetConvWeights()
    pred = paw_net_multi_conv(x0,x1,x2, weights, keep_prob)
    saver = tf.train.Saver()
    pholders = (x0,x1,x2,y,keep_prob)
    return pred, saver,pholders


# In[ ]:

def predict(img,sess,pred,pholders):
    x0_in,x1_in,x2_in = multiScaleImages(img[np.newaxis,:,:,:])
    imsz = conf.imsz
    lsz0 = int(math.ceil(float(imsz[0])/conf.pool_scale/conf.rescale))
    lsz1 = int(math.ceil(float(imsz[1])/conf.pool_scale/conf.rescale))

    labelim = np.zeros([1,lsz0,lsz1,1])

    out = sess.run(pred,feed_dict={pholders[0]:x0_in,
                     pholders[1]:x1_in,
                     pholders[2]:x2_in,
                     pholders[3]:labelim,
                     pholders[4]: 1.})
    return out


# In[ ]:

def predictMovie(model_file,inmovie,outmovie):
    pred,saver,pholders = initPredSession()
    tdir = tempfile.mkdtemp()

    cap = cv2.VideoCapture(inmovie)
    nframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    plt.gray()
    # with writer.saving(fig,"test_results.mp4",4):
    fig = plt.figure()
    
    with tf.Session() as sess:
        saver.restore(sess, model_file)
        
        count = 0
        for fnum in range(nframes):
            plt.clf()
            plt.axis('off')
            framein = myutils.readframe(cap,fnum)
            framein = framein[:,0:(old_div(framein.shape[1],2)),0:1]
            out = predict(copy.copy(framein),sess,pred,pholders)
            plt.imshow(framein[:,:,0])
            maxndx = np.argmax(out[0,:,:,0])
            loc = np.unravel_index(maxndx,out.shape[1:3])
            scalefactor = conf.rescale*conf.pool_scale
            plt.scatter(loc[1]*scalefactor,loc[0]*scalefactor,hold=True)

            fname = "test_{:06d}.png".format(count)
            plt.savefig(os.path.join(tdir,fname))
            count+=1

#     ffmpeg_cmd = "ffmpeg -r 30 " + \
#     "-f image2 -i '/path/to/your/picName%d.png' -qscale 0 '/path/to/your/new/video.avi'

    tfilestr = os.path.join(tdir,'test_*.png')
    mencoder_cmd = "mencoder mf://" + tfilestr +     " -frames " + "{:d}".format(count) + " -mf type=png:fps=15 -o " +     outmovie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
#     print(mencoder_cmd)
    os.system(mencoder_cmd)
    cap.release()

