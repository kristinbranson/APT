
# coding: utf-8

# In[28]:

'''
Mayank Feb 3 2016
'''
from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from past.utils import old_div
import tensorflow as tf
import os,sys
import tempfile
import copy
sys.path.append('/home/mayank/work/caffe/python')
sys.path.append('/home/mayank/work/pyutils')

import caffe
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math
import cv2
import numpy as np
import scipy

import multiPawTools
import myutils
from convNetBase import *

# import stephenHeadConfig as conf
import multiResData


# In[1]:


def train(conf):
    # Parameters
    learning_rate = conf.learning_rate
    training_iters = conf.training_iters
    batch_size = conf.batch_size
    display_step = conf.display_step

    # Network Parameters
    n_input = conf.psz
    n_classes = conf.n_classes # 
    dropout = conf.dropout # Dropout, probability to keep units
    imsz = conf.imsz
    rescale = conf.rescale
    scale = conf.scale
    pool_scale = conf.pool_scale
    x0,x1,x2,y,keep_prob = createPlaceHolders(imsz,
                              rescale,scale,pool_scale,n_classes)
    weights = initNetConvWeights(conf.nfilt,conf.nfcfilt,n_classes,
                                rescale,pool_scale,conf.psz)

    # training data stuff
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    env = lmdb.open(lmdbfilename, readonly = True)
    valenv = lmdb.open(vallmdbfilename, readonly = True)
    
    with env.begin() as txn,valenv.begin() as valtxn:
        train_cursor = txn.cursor()
        val_cursor = valtxn.cursor()

        # Construct model
        pred = net_multi_conv(x0,x1,x2, weights, keep_prob,
                              imsz,rescale,pool_scale)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.l2_loss(pred- y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            print("Initialized the network")
            step = 0
            # Keep training until reach max iterations
            while step < training_iters:
                batch_xs, locs = multiPawTools.readLMDB(train_cursor,
                                        batch_size,imsz,multiResData)

                locs = multiResData.sanitize_locs(locs)
                
                x0_in,x1_in,x2_in = multiPawTools.multiScaleImages(
                    batch_xs.transpose([0,2,3,1]),rescale,scale)
                
                labelims = multiPawTools.createLabelImages(locs,
                                   conf.imsz,conf.pool_scale*conf.rescale,
                                   conf.label_blur_rad) 
                
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
                        val_xs, locs = multiPawTools.readLMDB(val_cursor,
                                          batch_size,imsz,multiResData)
                        x0_in,x1_in,x2_in = multiPawTools.multiScaleImages(
                            val_xs.transpose([0,2,3,1]),rescale,scale)

                        labelims = multiPawTools.createLabelImages(locs,
                                                   conf.imsz,conf.pool_scale*conf.rescale,
                                                   conf.label_blur_rad)
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

