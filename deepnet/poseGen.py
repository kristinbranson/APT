from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[2]:

from builtins import range
from past.utils import old_div
import tensorflow as tf
import os,sys
import lmdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cv2
import tempfile
import copy
import re

from batch_norm import batch_norm_2D
import myutils
import PoseTools
import localSetup
import operator
import copy
from functools import reduce


# In[ ]:

def addDropoutLayer(ptrainObj,dropout,conf):
    l7 = ptrainObj.baseLayers['conv7']
    with tf.variable_scope('base/layer8') as scope:
        scope.reuse_variables()
        l7_do = tf.nn.dropout(l7,dropout,[conf.batch_size,1,1,conf.nfcfilt])
        l8_weights = tf.get_variable("weights", [1,1,conf.nfcfilt,conf.n_classes],
            initializer=tf.random_normal_initializer(stddev=0.01))
        l8_biases = tf.get_variable("biases", conf.n_classes,
            initializer=tf.constant_initializer(0))
        l8 = tf.nn.conv2d(l7_do,l8_weights,strides=[1,1,1,1],padding='SAME')+l8_biases
    return l8
    


# In[ ]:

def poseGenNet(locs,scores,l8,conf,ptrainObj,trainPhase):

    
    scores_sz = tf.Tensor.get_shape(scores).as_list()
    scores_numel = reduce(operator.mul, scores_sz[1:], 1)
    scores_re = tf.reshape(scores,[-1,scores_numel])
#     with tf.variable_scope('scores_fc'):
#         weights = tf.get_variable("weights", [scores_numel, conf.nfcfilt],
#             initializer=tf.random_normal_initializer(stddev=0.001))
#         biases = tf.get_variable("biases", conf.nfcfilt,
#             initializer=tf.constant_initializer(0))
        
#         scores_fc = tf.nn.relu(batch_norm_2D(tf.matmul(scores_re,weights)+biases,trainPhase))

        
    loc_sz = tf.Tensor.get_shape(locs).as_list()
    loc_numel = reduce(operator.mul, loc_sz[1:], 1)
    loc_re = tf.reshape(locs,[-1,loc_numel])
    joint = tf.concat(0,[scores_re,loc_re])
    with tf.variable_scope('loc_fc'):
        weights = tf.get_variable("weights", [loc_numel, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.01))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        joint_fc = tf.nn.relu(batch_norm_2D(tf.matmul(joint,weights)+biases,trainPhase))
        
#     joint_fc = tf.concat(1,[scores_fc,loc_fc])
    
    with tf.variable_scope('fc1'):
        weights = tf.get_variable("weights", [conf.nfcfilt*2, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.01))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        joint_fc1 = tf.nn.relu(batch_norm_2D(tf.matmul(joint_fc,weights)+biases,trainPhase))

    with tf.variable_scope('fc2'):
        weights = tf.get_variable("weights", [conf.nfcfilt, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.001))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        joint_fc2 = tf.nn.relu(batch_norm_2D(tf.matmul(joint_fc1,weights)+biases,trainPhase))
        
    with tf.variable_scope('out'):
        weights = tf.get_variable("weights", [conf.nfcfilt, conf.n_classes*2],
            initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", conf.n_classes*2,
            initializer=tf.constant_initializer(0))
        
        out = tf.matmul(joint_fc2,weights)+biases
        
    with tf.variable_scope('out_m'):
        weights = tf.get_variable("weights", [conf.nfcfilt, 2],
            initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", 2,
            initializer=tf.constant_initializer(0))
        
        out_m = tf.matmul(joint_fc2,weights)+biases
        
    layer_dict = {'scores_fc':scores_fc,
                  'loc_fc':loc_fc,
                  'joint_fc1':joint_fc1,
                  'joint_fc2':joint_fc2,
                  'out':out,
                  'out_m':out_m
                 }
    return out, out_m, layer_dict
    
    


# In[ ]:

def createGenPH(conf):
    scores = tf.placeholder(tf.float32,[None,conf.n_classes],name='scores')
    locs = tf.placeholder(tf.float32,[None,conf.n_classes,2],name='locs')
    learning_rate_ph = tf.placeholder(tf.float32,shape=[],name='learning_rate_gen')
    y = tf.placeholder(tf.float32,[None,conf.n_classes*2],name='y')
    y_m = tf.placeholder(tf.float32,[None,2],name='y_m')
    phase_train = tf.placeholder(tf.bool, name='phase_train')                 
    dropout = tf.placeholder(tf.float32, shape=[],name='gen_dropout')        
    phDict = {'scores':scores,'locs':locs,'learning_rate':learning_rate_ph,
              'y':y,'y_m':y_m,'phase_train':phase_train,'dropout':dropout}
    return phDict


# In[ ]:

def createFeedDict(phDict):
    feed_dict = {phDict['scores']:[],
                 phDict['locs']:[],
                 phDict['y']:[],
                 phDict['y_m']:[],
                 phDict['learning_rate']:1.,
                 phDict['phase_train']:False,
                 phDict['dropout']:1.
                }
    return feed_dict


# In[ ]:

def createGenSaver(conf):
    genSaver = tf.train.Saver(var_list = PoseTools.get_vars('poseGen'), max_to_keep=conf.maxckpt)
    return genSaver


# In[ ]:

def restoreGen(sess,conf,genSaver,restore=True):
    outfilename = os.path.join(conf.cachedir,conf.genoutname)
    latest_ckpt = tf.train.get_checkpoint_state(conf.cachedir,
                                        latest_filename = conf.genckptname)
    if not latest_ckpt or not restore:
        startat = 0
        sess.run(tf.initialize_variables(PoseTools.get_vars('poseGen')))
        print("Not loading gen variables. Initializing them")
        didRestore = False
    else:
        genSaver.restore(sess,latest_ckpt.model_checkpoint_path)
        matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
        startat = int(matchObj.group(1))+1
        print("Loading gen variables from %s"%latest_ckpt.model_checkpoint_path)
        didRestore = True
        
    return didRestore,startat


# In[ ]:

def saveGen(sess,step,genSaver,conf):
    outfilename = os.path.join(conf.cachedir,conf.genoutname)
    genSaver.save(sess,outfilename,global_step=step,
               latest_filename = conf.genckptname)


# In[ ]:

def genFewMovedNegSamples(locs,conf,nmove=1):
    # move few of the points randomly
    
    minlen = conf.gen_minlen
    minlen = float(minlen)
    maxlen = 2*minlen
    
    rlocs = copy.deepcopy(locs)

    sz = conf.imsz
    for curi in range(locs.shape[0]):
        for curp in range(nmove):
            rand_point = np.random.randint(conf.n_classes)
            rx = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                np.sign(np.random.rand()-0.5)
            ry = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                np.sign(np.random.rand()-0.5)

            rlocs[curi,rand_point,0] = rlocs[curi,rand_point,0] + rx*conf.rescale*conf.pool_scale
            rlocs[curi,rand_point,1] = rlocs[curi,rand_point,1] + ry*conf.rescale*conf.pool_scale
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[...,0]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[...,0] = xlocs
    ylocs = rlocs[...,1]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[...,1] = ylocs
    return rlocs


# In[ ]:

def genLocs(locs,predlocs,conf):
    dlocs = np.apply_over_axes(np.sum,(locs-predlocs)**2,axes=[1,2])
    dlocs = old_div(np.sqrt(dlocs),conf.n_classes)
    close = np.reshape(dlocs < (old_div(conf.gen_minlen,2)),[-1])
    newlocs = copy.deepcopy(predlocs)
    newlocs[close,...] = genFewMovedNegSamples(newlocs[close,...],conf,nmove=3)
    return newlocs


# In[ ]:

def prepareOpt(baseNet,l8,dbtype,feed_dict,sess,conf,phDict,distort,nsamples=10):
    baseNet.updateFeedDict(dbtype,distort)
    locs = baseNet.locs
    bout = sess.run(l8,feed_dict=baseNet.feed_dict)
    predlocs = PoseTools.get_base_pred_locs(bout, conf)
    
    #repeat locs nsamples times
    ls = locs.shape
    locs = np.tile(locs[:,np.newaxis,:,:],[1,nsamples,1,1])
    locs = np.reshape(locs,[ls[0]*nsamples,ls[1],ls[2]])
    predlocs = np.tile(predlocs[:,np.newaxis,:,:],[1,nsamples,1,1])
    predlocs = np.reshape(predlocs,[ls[0]*nsamples,ls[1],ls[2]])
    
    newlocs = genLocs(locs,predlocs,conf)
    new_mean = newlocs.mean(axis=1)
    
    locs_mean = locs.mean(axis=1)
    dlocs = locs-locs_mean[:,np.newaxis,:]
    newlocs = newlocs-new_mean[:,np.newaxis,:]
    
    d_mean = locs_mean-new_mean
    
    scores = np.zeros(locs.shape[0:2])
    scale = conf.rescale*conf.pool_scale
    rlocs = (np.round(old_div(newlocs,scale))).astype('int')
    for ndx in range(predlocs.shape[0]):
        for cls in range(conf.n_classes):
            bndx = int(math.floor(old_div(ndx,nsamples)))
            scores[ndx,cls] = bout[bndx,rlocs[ndx,cls,1],rlocs[ndx,cls,0],cls]

    feed_dict[phDict['y']] = np.reshape(dlocs,[-1,2*conf.n_classes])
    feed_dict[phDict['y_m']] = d_mean
    feed_dict[phDict['scores']] = scores
    feed_dict[phDict['locs']] = newlocs
    return new_mean, locs_mean
#     gg = 3/0


# In[ ]:

def train(conf,restore=True):
    
    phDict = createGenPH(conf)
    feed_dict = createFeedDict(phDict)
    feed_dict[phDict['phase_train']] = True
    feed_dict[phDict['dropout']] = 0.5
    feed_dict[phDict['y']] = np.zeros((conf.batch_size,conf.n_classes*2))
    baseNet = PoseTools.create_network(conf, 1)
    l8 = addDropoutLayer(baseNet,phDict['dropout'],conf)
    with tf.variable_scope('poseGen'):
        out,out_m,layer_dict = poseGenNet(phDict['locs'],phDict['scores'],l8,
                                     conf,baseNet,phDict['phase_train'])
        
    genSaver = createGenSaver(conf)
    y = phDict['y']
    y_m = phDict['y_m']
    ind_loss = old_div(tf.nn.l2_loss(out-y),conf.n_classes)
    mean_loss = tf.nn.l2_loss(out_m-y_m)
    loss = ind_loss + mean_loss
    in_loss = tf.nn.l2_loss(phDict['y']-tf.reshape(phDict['locs'],[-1,2*conf.n_classes]))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
    baseNet.open_dbs()
    baseNet.feed_dict[phDict['dropout']] = feed_dict[phDict['dropout']]
    
    with baseNet.env.begin() as txn,baseNet.valenv.begin() as valtxn,tf.Session() as sess:

        baseNet.create_cursors()
        baseNet.restoreBase(sess,True)
        didRestore,startat = restoreGen(sess,conf,genSaver,restore)
        baseNet.initializeRemainingVars(sess)
        for step in range(startat,conf.gen_training_iters+1):
            prepareOpt(baseNet,l8,baseNet.DBType.Train,feed_dict,sess,conf,
                       phDict,distort=True)
            feed_dict[phDict['phase_train']] = True
            sess.run(train_step, feed_dict=feed_dict)

            if step % 25 == 0:
                prepareOpt(baseNet,l8,baseNet.DBType.Train,feed_dict,
                           sess,conf,phDict,distort=False)
                feed_dict[phDict['phase_train']] = False
                train_loss = sess.run([loss,in_loss,out,out_m,ind_loss,mean_loss], feed_dict=feed_dict)
                train_mean_loss = old_div(np.sum((train_loss[3]-feed_dict[phDict['y_m']])**2 ),2)
                train_ind_loss = old_div(np.sum((train_loss[2]-feed_dict[phDict['y']])**2 ),2)
                test_loss = 0
                test_in_loss = 0
                test_ind_loss = 0 
                test_mean_loss = 0 
                
                
                nrep = 10
                for rep in range(nrep):
                    prepareOpt(baseNet,l8,baseNet.DBType.Val,feed_dict,sess,conf,
                               phDict,distort=False)
                    tloss = sess.run([loss,in_loss,out,out_m], feed_dict=feed_dict)
                    test_loss += tloss[0]
                    test_in_loss += tloss[1]
                    test_mean_loss += old_div(np.sum((tloss[3]-feed_dict[phDict['y_m']])**2 ),2)
                    test_ind_loss += old_div(np.sum((tloss[2]-feed_dict[phDict['y']])**2 ),2)

                print("Iter:{:d}, train:{:.4f},mean:{:.4f},ind:{:.4f} test:{:.4f},mean:{:.4f},ind:{:.4f} ".format(step, 
                      np.sqrt(old_div(train_loss[0],conf.batch_size)),
                      np.sqrt(old_div(train_mean_loss,conf.batch_size)),
                      np.sqrt(old_div((old_div(train_ind_loss,conf.batch_size)),conf.n_classes)),
                      np.sqrt(old_div((old_div(test_loss,nrep)),conf.batch_size)),
                      np.sqrt(old_div((old_div(test_mean_loss,nrep)),conf.batch_size)),
                      np.sqrt(old_div((old_div((old_div(test_ind_loss,nrep)),conf.batch_size)),conf.n_classes))))
                
            if step % 100 == 0:
                saveGen(sess,step,genSaver,conf)

