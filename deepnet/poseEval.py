from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[2]:

from builtins import range
from past.utils import old_div
import tensorflow as tf
import os,sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cv2
import tempfile
import copy
import re

from batch_norm import *
import myutils
import PoseTools
import localSetup
import operator
import copy
import convNetBase as CNB
import multiResData
from functools import reduce


# In[ ]:

def conv_relu(X, kernel_shape, conv_std,bias_val,doBatchNorm,trainPhase,addSummary=True):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer(stddev=conv_std))
    biases = tf.get_variable("biases", kernel_shape[-1],
        initializer=tf.constant_initializer(bias_val))
    if addSummary:
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
#     PoseTools.variable_summaries(biases)
    conv = tf.nn.conv2d(X, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    if doBatchNorm:
        conv = batch_norm(conv,trainPhase)
    with tf.variable_scope('conv'):
        PoseTools.variable_summaries(conv)
    return tf.nn.relu(conv - biases)

def max_pool(name, l_input, k,s):
    return tf.nn.max_pool(
        l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], 
        padding='SAME', name=name)

def createPlaceHolders(conf):
#     imsz = conf.imsz
    # tf Graph input
    imsz = conf.imsz
    rescale = conf.rescale
    scale = conf.scale
    pool_scale = conf.pool_scale
    n_classes = conf.n_classes
    imgDim = conf.imgDim
    
    keep_prob = tf.placeholder(tf.float32,name='dropout') # dropout(keep probability)
    inScale = rescale*conf.eval_scale
    
    nex = conf.batch_size*(conf.eval_num_neg+1)
    x0 = tf.placeholder(tf.float32, [nex,
                                     old_div(imsz[0],inScale),
                                     old_div(imsz[1],inScale),imgDim],name='x0')
    x1 = tf.placeholder(tf.float32, [nex,
                                     imsz[0]/scale/inScale,
                                     imsz[1]/scale/inScale,imgDim],name='x1')
    x2 = tf.placeholder(tf.float32, [nex,
                                     imsz[0]/scale/scale/inScale,
                                     imsz[1]/scale/scale/inScale,imgDim],name='x2')

    scores_scale = scale*scale*inScale
    s0 = tf.placeholder(tf.float32, [nex,
                                     old_div(imsz[0],scores_scale),
                                     old_div(imsz[1],scores_scale),n_classes],name='s0')
    s1 = tf.placeholder(tf.float32, [nex,
                                     imsz[0]/scale/scores_scale,
                                     imsz[1]/scale/scores_scale,n_classes],name='s1')
    s2 = tf.placeholder(tf.float32, [nex,
                                     imsz[0]/scale/scale/scores_scale,
                                     imsz[1]/scale/scale/scores_scale,n_classes],name='s2')
    
    y = tf.placeholder(tf.float32, [nex*n_classes,],'out')
    
    X = [x0,x1,x2]
    S = [s0,s1,s2]
    phase_train = tf.placeholder(tf.bool,name='phase_train')
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')
    
    ph = {'X':X,'S':S,'y':y,'keep_prob':keep_prob,
          'phase_train':phase_train,'learning_rate':learning_rate}
    return ph


def createFeedDict(ph):
    feed_dict = { ph['X'][0]:[],
                  ph['X'][1]:[],
                  ph['X'][2]:[],
                  ph['S'][0]:[],
                  ph['S'][1]:[],
                  ph['S'][2]:[],
                  ph['y']:[],
                  ph['learning_rate']:1,
                  ph['phase_train']:False,
                  ph['keep_prob']:1.
                  }
    
    return feed_dict

def net_multi_base_named(X,nfilt,doBatchNorm,trainPhase,doPool=True,addSummary=True):
    inDimX = X.get_shape()[3]
    with tf.variable_scope('layer1_X'):
        conv1 = conv_relu(X,[5, 5, inDimX, 48],0.01,0,doBatchNorm,trainPhase,addSummary)
    
#     inDimS = S.get_shape()[3]
#     with tf.variable_scope('layer1_S'):
#         conv1s = conv_relu(S,[5, 5, inDimS, 48],0.01,0,doBatchNorm,trainPhase)
            
#     conv1_cat = tf.concat(3,[conv1x,conv1s])
    if doPool:
        pool1 = max_pool('pool1',conv1,k=3,s=2)
    else:
        pool1 = conv1
            
    with tf.variable_scope('layer2'):
        conv2 = conv_relu(pool1,[3,3,48,nfilt],0.01,0,doBatchNorm,trainPhase,addSummary)
    if doPool:
        pool2 = max_pool('pool2',conv2,k=3,s=2)
    else:
        pool2 = conv2
            
    with tf.variable_scope('layer3'):
        conv3 = conv_relu(pool2,[3,3,nfilt,nfilt],0.01,0,doBatchNorm,trainPhase,addSummary)
    with tf.variable_scope('layer4'):
        conv4 = conv_relu(conv3,[3,3,nfilt,nfilt],0.01,0,doBatchNorm,trainPhase,addSummary)
    with tf.variable_scope('layer5'):
        conv5 = conv_relu(conv4,[3,3,nfilt,old_div(nfilt,4)],0.01,0,doBatchNorm,trainPhase,addSummary)
        
    out_dict = {'conv1':conv1,'conv2':conv2,'conv3':conv3,
                'conv4':conv4,'conv5':conv5}
    return conv5,out_dict
        

def net_multi_conv(ph,conf):
    X = ph['X']
    S = ph['S']
    X0,X1,X2 = X
    S0,S1,S2 = S
    
    trainPhase = ph['phase_train']
    _dropout = ph['keep_prob']
    
    imsz = conf.imsz; rescale = conf.rescale
    pool_scale = conf.pool_scale
    nfilt = conf.nfilt
    doBatchNorm = conf.doBatchNorm
    
    #     conv5_0,base_dict_0 = net_multi_base(X0,_weights['base0'])
    #     conv5_1,base_dict_1 = net_multi_base(X1,_weights['base1'])
    #     conv5_2,base_dict_2 = net_multi_base(X2,_weights['base2'])
    with tf.variable_scope('scale0'):
        conv5_0,base_dict_0 = net_multi_base_named(X0,nfilt,doBatchNorm,trainPhase,True,True)
    with tf.variable_scope('scale1'):
        conv5_1,base_dict_1 = net_multi_base_named(X1,nfilt,doBatchNorm,trainPhase,True,False)
    with tf.variable_scope('scale2'):
        conv5_2,base_dict_2 = net_multi_base_named(X2,nfilt,doBatchNorm,trainPhase,True,False)
    with tf.variable_scope('scale0_scores'):
        conv5s_0,base_dict_s = net_multi_base_named(S0,nfilt,doBatchNorm,trainPhase,False,True)
    with tf.variable_scope('scale1_scores'):
        conv5s_1,base_dict_s = net_multi_base_named(S1,nfilt,doBatchNorm,trainPhase,False,False)
    with tf.variable_scope('scale2_scores'):
        conv5s_2,base_dict_s = net_multi_base_named(S2,nfilt,doBatchNorm,trainPhase,False,False)

#     sz0 = int(math.ceil(float(imsz[0])/pool_scale/rescale/conf.eval_scale))
#     sz1 = int(math.ceil(float(imsz[1])/pool_scale/rescale/conf.eval_scale))
    sz = tf.Tensor.get_shape(conv5_2).as_list()
    sz0 = sz[1]
    sz1 = sz[2]
    conv5_0_down = CNB.upscale('5_0',conv5_0,[sz0,sz1])
    conv5_1_down = CNB.upscale('5_1',conv5_1,[sz0,sz1])
    conv5s_0_down = CNB.upscale('5s_0',conv5s_0,[sz0,sz1])
    conv5s_1_down = CNB.upscale('5s_1',conv5s_1,[sz0,sz1])

    # crop lower res layers to match higher res size
#     conv5_0_sz = tf.Tensor.get_shape(conv5_0).as_list()
#     conv5_1_sz = tf.Tensor.get_shape(conv5_1_up).as_list()
#     crop_0 = int((sz0-conv5_0_sz[1])/2)
#     crop_1 = int((sz1-conv5_0_sz[2])/2)

#     curloc = [0,crop_0,crop_1,0]
#     patchsz = tf.to_int32([-1,conv5_0_sz[1],conv5_0_sz[2],-1])
#     conv5_1_up = tf.slice(conv5_1_up,curloc,patchsz)
#     conv5_2_up = tf.slice(conv5_2_up,curloc,patchsz)
#     conv5_1_final_sz = tf.Tensor.get_shape(conv5_1_up).as_list()

    conv5_cat = tf.concat(3,[conv5_0_down,conv5_1_down,conv5_2,conv5s_0_down,conv5s_1_down,conv5s_2])
    
    nex = conf.batch_size*(conf.eval_num_neg+1)
    conv5_reshape = tf.reshape(conv5_cat,[nex,-1])
    conv5_dims = conv5_reshape.get_shape()[1].value
    with tf.variable_scope('layer6'):
#         weights = tf.get_variable("weights", [conv5_dims,conf.nfcfilt/2],
#             initializer=tf.random_normal_initializer(stddev=0.005))
        weights = tf.get_variable("weights", [conv5_dims,old_div(conf.nfcfilt,2)],
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", old_div(conf.nfcfilt,2),
            initializer=tf.constant_initializer(0))
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(biases)

        conv6 = tf.nn.relu(batch_norm_2D(tf.matmul(conv5_reshape, weights),trainPhase)-biases)
        with tf.variable_scope('conv'):
            PoseTools.variable_summaries(conv6)

    with tf.variable_scope('layer7'):
#         weights = tf.get_variable("weights", [conf.nfcfilt/2,conf.nfcfilt/2],
#             initializer=tf.random_normal_initializer(stddev=0.005))
        weights = tf.get_variable("weights", [old_div(conf.nfcfilt,2),old_div(conf.nfcfilt,4)],
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", old_div(conf.nfcfilt,4),
            initializer=tf.constant_initializer(0))
        
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(biases)

        conv7 = tf.nn.relu(batch_norm_2D(tf.matmul(conv6, weights),trainPhase)-biases)
        with tf.variable_scope('conv'):
            PoseTools.variable_summaries(conv7)

    with tf.variable_scope('layer8'):
        l8_weights = tf.get_variable("weights", [old_div(conf.nfcfilt,4),conf.n_classes],
            initializer=tf.random_normal_initializer(stddev=0.01))
        l8_biases = tf.get_variable("biases", conf.n_classes,
            initializer=tf.constant_initializer(0))
        out = tf.matmul(conv7, l8_weights) - l8_biases
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(l8_weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(l8_biases)
        with tf.variable_scope('out'):
            PoseTools.variable_summaries(out)
        nex = conf.batch_size*(conf.eval_num_neg+1)
        out = tf.reshape(out,[nex*conf.n_classes]) 
        #this should keep all the outputs of an example together

    out_dict = {'base_dict_0':base_dict_0,
                'base_dict_1':base_dict_1,
                'base_dict_2':base_dict_2,
                'conv6':conv6,
                'conv7':conv7,
               }
    
    return out,out_dict

def openDBs(conf,trainType=0):
        if trainType == 0:
            trainfilename =os.path.join(conf.cachedir,conf.trainfilename) + '.tfrecords'
            valfilename =os.path.join(conf.cachedir,conf.valfilename) + '.tfrecords'
            train_queue = tf.train.string_input_producer([trainfilename])
            val_queue = tf.train.string_input_producer([valfilename])
        else:
            trainfilename =os.path.join(conf.cachedir,conf.fulltrainfilename) + '.tfrecords'
            valfilename =os.path.join(conf.cachedir,conf.fulltrainfilename) + '.tfrecords'
            train_queue = tf.train.string_input_producer([trainfilename])
            val_queue = tf.train.string_input_producer([valfilename])
        return [train_queue,val_queue]

def createCursors(sess,queue,conf):
            
        train_queue,val_queue = queue
        train_ims,train_locs,temp = multiResData.read_and_decode(train_queue,conf)
        val_ims,val_locs,temp = multiResData.read_and_decode(val_queue,conf)
        train_data = [train_ims,train_locs]
        val_data = [val_ims,val_locs]
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        return [train_data,val_data],coord,threads
        

def readImages(conf,dbType,distort,sess,data):
    train_data,val_data = data
    cur_data = val_data if (dbType=='val')             else train_data
    xs = []; locs = []
    
    for ndx in range(conf.batch_size):
        [curxs,curlocs] = sess.run(cur_data)
        if np.ndim(curxs)<3:
            xs.append(curxs[np.newaxis,:,:])
        else:
            xs.append(curxs)
        locs.append(curlocs)
    xs = np.array(xs)    
    locs = np.array(locs)
    locs = multiResData.sanitize_locs(locs)
    if distort:
        if conf.horzFlip:
            xs,locs = PoseTools.randomly_flip_lr(xs, locs)
        if conf.vertFlip:
            xs,locs = PoseTools.randomly_flip_ud(xs, locs)
        xs,locs = PoseTools.randomly_rotate(xs, locs, conf)
        xs = PoseTools.randomly_adjust(xs, conf)

    return xs,locs


# In[ ]:

def updateFeedDict(conf,dbType,distort,sess,data,feed_dict,ph):
    xs,locs = readImages(conf,dbType,distort,sess,data)
    labelims = PoseTools.create_label_images(locs,
                                             conf.imsz,
                                             conf.rescale * conf.eval_scale,
                                             conf.label_blur_rad)
    
    nlocs = genNegSamples(labelims,locs,conf,nsamples=1,minlen=conf.eval_minlen,N=conf.N2move4neg)
    
   
#     alllocs = np.concatenate([locs[...,np.newaxis],nlocs],axis=-1)
    alllocs = nlocs
    dd = alllocs-locs[...,np.newaxis] # distance of neg points to actual locations
    ddist = np.sqrt(np.sum(dd**2,axis=2))
    ind_labels = old_div((ddist - old_div(conf.eval_minlen,2) ),conf.eval_minlen)
    ind_labels = ind_labels.clip(min=0,max=1)
    ind_labels = np.transpose(ind_labels,[0,2,1])
    ind_labels = ind_labels.reshape((-1,))
#     ind_labels = np.concatenate([ind_labels,1-ind_labels],axis=1)
    alllocs = alllocs.transpose([0,3,1,2])
    alllocs = alllocs.reshape((-1,)+alllocs.shape[2:])
    
    allxs = np.tile(xs[...,np.newaxis],conf.eval_num_neg+1)
    allxs = allxs.transpose([0,4,1,2,3])
    allxs = np.reshape(allxs,[-1,allxs.shape[-3],allxs.shape[-2],allxs.shape[-1]])
    
    x0,x1,x2 = PoseTools.multi_scale_images(allxs.transpose([0, 2, 3, 1]),
                                            conf.rescale * conf.eval_scale,
                                            conf.scale, conf.l1_cropsz, conf)


    scores_scale = conf.rescale*conf.eval_scale*conf.scale*conf.scale
    s0 = PoseTools.create_label_images(alllocs,
                                       conf.imsz,
                                       scores_scale,
                                       conf.label_blur_rad)
    s1 = PoseTools.create_label_images(alllocs,
                                       conf.imsz,
                                       scores_scale * conf.scale,
                                       conf.label_blur_rad)
    s2 = PoseTools.create_label_images(alllocs,
                                       conf.imsz,
                                       scores_scale * conf.scale * conf.scale,
                                       conf.label_blur_rad)
    
#     s0,s1,s2 = PoseTools.multiScaleLabelImages(labelims,1,conf.scale,[])
    
#     y = np.zeros([xs.shape[0],nlocs.shape[-1]+1,2])
#     y[:,:1,0] = 1. 
#     y[:,1:,1] = 1.
#     y = np.reshape(y,[-1,y.shape[-1]])
 
    feed_dict[ph['X'][0]] = x0
    feed_dict[ph['X'][1]] = x1
    feed_dict[ph['X'][2]] = x2
    feed_dict[ph['S'][0]] = s0
    feed_dict[ph['S'][1]] = s1
    feed_dict[ph['S'][2]] = s2
    feed_dict[ph['y']] = ind_labels
    return alllocs


# In[ ]:

def restoreEval(sess,evalsaver,restore,conf,feed_dict):
    outfilename = os.path.join(conf.cachedir,conf.evaloutname)
    latest_ckpt = tf.train.get_checkpoint_state(conf.cachedir,
                                        latest_filename = conf.evalckptname)
    if not latest_ckpt or not restore:
        evalstartat = 0
        sess.run(tf.variables_initializer(PoseTools.get_vars('eval')), feed_dict=feed_dict)
        print("Not loading Eval variables. Initializing them")
    else:
        evalsaver.restore(sess,latest_ckpt.model_checkpoint_path)
        matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
        evalstartat = int(matchObj.group(1))+1
        print("Loading eval variables from %s"%latest_ckpt.model_checkpoint_path)
    return  evalstartat
    

def saveEval(sess,evalsaver,step,conf):
    outfilename = os.path.join(conf.cachedir,conf.evaloutname)
    evalsaver.save(sess,outfilename,global_step=step,
               latest_filename = conf.evalckptname)
    print('Saved state to %s-%d' %(outfilename,step))

def createEvalSaver(conf):
    evalsaver = tf.train.Saver(var_list = PoseTools.get_vars('eval'),
                               max_to_keep=conf.maxckpt)
    return evalsaver

def initializeRemainingVars(sess,feed_dict):
    varlist = tf.global_variables()
    for var in varlist:
        try:
            sess.run(tf.assert_variables_initialized([var]))
        except tf.errors.FailedPreconditionError:
            sess.run(tf.variables_initializer([var]))
            print('Initializing variable:%s'%var.name)



# In[ ]:

def poseEvalNetInit(conf):
    
    ph = createPlaceHolders(conf)
    feed_dict = createFeedDict(ph)
    with tf.variable_scope('eval'):
        out,out_dict = net_multi_conv(ph,conf)
    trainType = 0
    queue = openDBs(conf,trainType=trainType)
    if trainType == 1:
        print("Training with all the data!")
        print("Validation data is same as training data!!!! ")
    return ph,feed_dict,out,queue,out_dict
    


# In[ ]:

def poseEvalTrain(conf,restore=True):
    ph,feed_dict,out,queue,_ = poseEvalNetInit(conf)
    feed_dict[ph['phase_train']] = True
    feed_dict[ph['keep_prob']] = 1.
    evalSaver = createEvalSaver(conf) 
    pwts = conf.eval_num_neg
    
    # For classification
#     weights = ph['y'][:,0]*(pwts-1)+1
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(out,ph['y'])
#     loss = tf.reduce_mean(cross_entropy)
#     correct_pred = tf.cast(tf.equal(tf.argmax(out,1),tf.argmax(ph['y'],1)),tf.float32)
#     correct_pred_weighted = tf.mul(correct_pred)
#     accuracy_weighted = tf.reduce_sum(correct_pred_weighted)/tf.reduce_sum(weights)

    # for regression.
    loss = tf.nn.l2_loss(out-ph['y'])
    correct_pred = tf.cast(tf.equal(out>0.5,ph['y']>0.5),tf.float32)
    accuracy = tf.reduce_mean(correct_pred)
    
    tf.summary.scalar('cross_entropy',loss)
    tf.summary.scalar('accuracy',accuracy)
    
    opt = tf.train.AdamOptimizer(learning_rate=                       ph['learning_rate']).minimize(loss)
    
    merged = tf.summary.merge_all()
    
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(conf.cachedir + '/eval_train_summary',sess.graph)
        test_writer = tf.summary.FileWriter(conf.cachedir + '/eval_test_summary',sess.graph)
        data,coord,threads = createCursors(sess,queue,conf)
        updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
        evalstartat = restoreEval(sess,evalSaver,restore,conf,feed_dict)
        initializeRemainingVars(sess,feed_dict)
        for step in range(evalstartat,conf.eval_training_iters+1):
            excount = step*conf.batch_size
            cur_lr = conf.eval_learning_rate
                        #* conf.gamma**math.floor(excount/conf.step_size)
            feed_dict[ph['learning_rate']] = cur_lr
            feed_dict[ph['keep_prob']] = 1.
            feed_dict[ph['phase_train']] = True
            updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
            train_summary,_ = sess.run([merged,opt], feed_dict=feed_dict)
            train_writer.add_summary(train_summary,step)

            if step % conf.display_step == 0:
                updateFeedDict(conf,'train',sess=sess,distort=True,data=data,feed_dict=feed_dict,ph=ph)
                feed_dict[ph['keep_prob']] = 1.
                feed_dict[ph['phase_train']] = False
                train_loss,train_acc = sess.run([loss,accuracy],feed_dict=feed_dict)
                numrep = int(old_div(conf.numTest,conf.batch_size))+1
                val_loss = 0.
                val_acc = 0.
#                 val_acc_wt = 0.
                for rep in range(numrep):
                    updateFeedDict(conf,'val',distort=False,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
                    vloss,vacc = sess.run([loss,accuracy],feed_dict=feed_dict)
                    val_loss += vloss
                    val_acc += vacc
#                     val_acc_wt += vacc_wt
                val_loss = old_div(val_loss,numrep)
                val_acc = old_div(val_acc,numrep)
#                 val_acc_wt /= numrep
                test_summary,_ = sess.run([merged,loss],feed_dict=feed_dict)
                test_writer.add_summary(test_summary,step)
                print('Val -- Acc:{:.4f} Loss:{:.4f} Train Acc:{:.4f} Loss:{:.4f} Iter:{}'.format(
                    val_acc,val_loss,train_acc,train_loss,step))
            if step % conf.save_step == 0:
                saveEval(sess,evalSaver,step,conf)
        print("Optimization Done!")
        saveEval(sess,evalSaver,step,conf)
        train_writer.close()
        test_writer.close()
        coord.request_stop()
        coord.join(threads)
        


# In[ ]:

def poseEvalNetTiny(lin,locs,conf,trainPhase,dropout):

    lin_sz = tf.Tensor.get_shape(lin).as_list()
    lin_numel = reduce(operator.mul, lin_sz[1:], 1)
    lin_re = tf.reshape(lin,[-1,lin_numel])
    lin_re = tf.nn.dropout(lin_re,dropout)
    with tf.variable_scope('lin_fc'):
        weights = tf.get_variable("weights", [lin_numel, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        lin_fc = tf.nn.relu(batch_norm_2D(tf.matmul(lin_re,weights)+biases,trainPhase))

        
    loc_sz = tf.Tensor.get_shape(locs).as_list()
    loc_numel = reduce(operator.mul, loc_sz[1:], 1)
    loc_re = tf.reshape(locs,[-1,loc_numel])
    with tf.variable_scope('loc_fc'):
        weights = tf.get_variable("weights", [loc_numel, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        loc_fc = tf.nn.relu(batch_norm_2D(tf.matmul(loc_re,weights)+biases,trainPhase))
        
    joint_fc = tf.concat(1,[lin_fc,loc_fc])
    
    with tf.variable_scope('fc1'):
        weights = tf.get_variable("weights", [conf.nfcfilt*2, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        joint_fc1 = tf.nn.relu(batch_norm_2D(tf.matmul(joint_fc,weights)+biases,trainPhase))

    with tf.variable_scope('fc2'):
        weights = tf.get_variable("weights", [conf.nfcfilt, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        joint_fc2 = tf.nn.relu(batch_norm_2D(tf.matmul(joint_fc1,weights)+biases,trainPhase))
        
    with tf.variable_scope('out'):
        weights = tf.get_variable("weights", [conf.nfcfilt, 2],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", 2,
            initializer=tf.constant_initializer(0))
        
        out = tf.matmul(joint_fc2,weights)+biases
        
    layer_dict = {'lin_fc':lin_fc,
                  'loc_fc':loc_fc,
                  'joint_fc1':joint_fc1,
                  'joint_fc2':joint_fc2,
                  'out':out
                 }
    return out,layer_dict
    
    


# In[ ]:

def createEvalPH(conf):
    lin = tf.placeholder(tf.float32,[None,conf.n_classes,conf.nfcfilt])
    locs = tf.placeholder(tf.float32,[None,conf.n_classes,2])
    learning_rate_ph = tf.placeholder(tf.float32,shape=[])
    y = tf.placeholder(tf.float32,[None,2])
    phase_train = tf.placeholder(tf.bool, name='phase_train')                 
    dropout = tf.placeholder(tf.float32, shape=[])                 
    phDict = {'lin':lin,'locs':locs,'learning_rate':learning_rate_ph,
              'y':y,'phase_train':phase_train,'dropout':dropout}
    return phDict


# In[ ]:

# def createFeedDict(phDict):
#     feed_dict = {phDict['lin']:[],
#                  phDict['locs']:[],
#                  phDict['y']:[],
#                  phDict['learning_rate']:1.,
#                  phDict['phase_train']:False,
#                  phDict['dropout']:1.
#                 }
#     return feed_dict


# In[ ]:

def genLabels(rlocs,locs,conf):
    d2locs = np.sqrt(((rlocs-locs[...,np.newaxis])**2).sum(-2))
    ll = np.arange(1,conf.n_classes+1)
    labels = np.tile(ll[:,np.newaxis],[d2locs.shape[0],1,d2locs.shape[2]])
    labels[d2locs>conf.poseEvalNegDist] = -1.
    labels[d2locs<conf.poseEvalNegDist] = 1.
    labels = np.concatenate([labels[:,np.newaxis],1-labels[:,np.newaxis]],-1)


# In[ ]:

def genRandomNegSamples(bout,l7out,locs,conf,nsamples=10):
    sz = (np.array(l7out.shape[1:3])-1)*conf.rescale*conf.pool_scale
    bsize = conf.batch_size
    rlocs = np.zeros(locs.shape + (nsamples,))
    rlocs[:,:,0,:] = np.random.randint(sz[1],size=locs.shape[0:2]+(nsamples,))
    rlocs[:,:,1,:] = np.random.randint(sz[0],size=locs.shape[0:2]+(nsamples,))
    return rlocs


# In[ ]:

def genGaussianPosSamples(bout,l7out,locs,conf,nsamples=10,maxlen = 4):
    scale = conf.rescale*conf.pool_scale
    sigma = float(maxlen)*0.5*scale
    sz = (np.array(l7out.shape[1:3])-1)*scale
    bsize = conf.batch_size
    rlocs = np.round(np.random.normal(size=locs.shape+(15*nsamples,))*sigma)
    # remove rlocs that are far away.
    dlocs = np.all( np.sqrt( (rlocs**2).sum(2))< (maxlen*scale),1)
    clocs = np.zeros(locs.shape+(nsamples,))
    for ii in range(dlocs.shape[0]):
        ndx = np.where(dlocs[ii,:])[0][:nsamples]
        clocs[ii,:,:,:] = rlocs[ii,:,:,ndx].transpose([1,2,0])

    rlocs = locs[...,np.newaxis] + clocs
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genGaussianNegSamples(bout,locs,conf,nsamples=10,minlen = 8):
    sigma = minlen
#     sz = (np.array(bout.shape[1:3])-1)*scale
    sz = np.array(bout.shape[1:3])-1
    bsize = conf.batch_size
    rlocs = np.round(np.random.normal(size=locs.shape+(5*nsamples,))*sigma)
    # remove rlocs that are small.
    dlocs = np.sqrt( (rlocs**2).sum(2)).sum(1)
    clocs = np.zeros(locs.shape+(nsamples,))
    for ii in range(dlocs.shape[0]):
        ndx = np.where(dlocs[ii,:]> (minlen*conf.n_classes) )[0][:nsamples]
        clocs[ii,:,:,:] = rlocs[ii,:,:,ndx].transpose([1,2,0])

    rlocs = locs[...,np.newaxis] + clocs
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genMovedNegSamples(bout,locs,conf,nsamples=10,minlen=8):
    # Add same x and y to locs
    
    minlen = old_div(float(minlen),2)
    maxlen = 2*minlen
    rlocs = np.zeros(locs.shape + (nsamples,))
#     sz = (np.array(bout.shape[1:3])-1)*conf.rescale*conf.pool_scale
    sz = np.array(bout.shape[1:3])-1

    for curi in range(locs.shape[0]):
        rx = np.round(np.random.rand(nsamples)*(maxlen-minlen) + minlen)*            np.sign(np.random.rand(nsamples)-0.5)
        ry = np.round(np.random.rand(nsamples)*(maxlen-minlen) + minlen)*            np.sign(np.random.rand(nsamples)-0.5)

        rlocs[curi,:,0,:] = locs[curi,:,0,np.newaxis] + rx
        rlocs[curi,:,1,:] = locs[curi,:,1,np.newaxis] + ry
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genNMovedNegSamples(bout,locs,N,conf,nsamples=10,minlen=8):
    # Move one of the points.
    minlen = float(minlen)
    maxlen = 2*minlen
    
    rlocs = np.tile(locs[...,np.newaxis],[1,1,1,nsamples])
    sz = np.array(bout.shape[1:3])-1

    for curi in range(locs.shape[0]):
        for curs in range(nsamples):
            curN = np.random.randint(conf.n_classes)
            for rand_point in np.random.choice(conf.n_classes,size=[curN,],replace=False):
                rx = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                    np.sign(np.random.rand()-0.5)
                ry = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                    np.sign(np.random.rand()-0.5)

                rlocs[curi,rand_point,0,curs] = locs[curi,rand_point,0] + rx
                rlocs[curi,rand_point,1,curs] = locs[curi,rand_point,1] + ry
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genNegSamples(bout,locs,conf,nsamples=10,minlen=8,N=1):
    rlocs = np.concatenate([
#                           genRandomNegSamples(bout,l7out,locs,conf,nsamples),
#                           genGaussianNegSamples(bout,locs,conf,nsamples,minlen),
#                           genMovedNegSamples(bout,locs,conf,nsamples,minlen), 
                          genNMovedNegSamples(bout,locs,N,conf,nsamples,minlen)], 
                          axis=3)
#     rlabels = genLabels(rlocs,locs,conf)
    return rlocs#,rlabels


# In[ ]:

def genData(l7out,inlocs,conf):
    locs = np.round(old_div(inlocs,(conf.rescale*conf.pool_scale)))
    dd = np.zeros(locs.shape[0:2]+l7out.shape[-1:]+locs.shape[-1:])
    for curi in range(locs.shape[0]):
        for pp in range(locs.shape[1]):
            for s in range(locs.shape[3]):
                dd[curi,pp,:,s] = l7out[curi,int(locs[curi,pp,1,s]),int(locs[curi,pp,0,s]),:]
    return dd        


# In[ ]:

def prepareOpt(baseNet,dbtype,feed_dict,sess,conf,phDict,distort):
    nsamples = 10
    npos = 2
    baseNet.updateFeedDict(dbtype,distort)
    locs = baseNet.locs
    l7 = baseNet.baseLayers['conv7']
    [bout,l7out] = sess.run([baseNet.basePred,l7],feed_dict=baseNet.feed_dict)
    neglocs = genNegSamples(bout,l7out,locs,conf,nsamples=nsamples)
#     replocs = np.tile(locs[...,np.newaxis],npos*nsamples)
    replocs = genGaussianPosSamples(bout,l7out,locs,conf,nsamples*npos,maxlen=4)
    alllocs = np.concatenate([neglocs,replocs],axis=-1)
    alllocs = alllocs.transpose([0,3,1,2])
    alllocs = alllocs.reshape((-1,)+alllocs.shape[2:])
    retlocs = copy.deepcopy(alllocs)
    alllocs_m = alllocs.mean(1)
    alllocs = alllocs-alllocs_m[:,np.newaxis,:]
    
#    poslabels = np.ones(replocs.shape[0,1,3])
#    alllabels = np.concatenate([neglabels,poslabels],axis=-2)
#    alllabels = alllabels.transpose([0,2,1,3])
#    alllabels = alllabels.reshape((-1,)+alllabels.shape[-1])

    negdd = genData(l7out,neglocs,conf)
    posdd = genData(l7out,replocs,conf)
    alldd = np.concatenate([negdd,posdd],axis=-1)
    alldd = alldd.transpose([0,3,1,2])
    alldd = np.reshape(alldd,[-1,alldd.shape[-2],alldd.shape[-1]])

#     y = alllabels
    y = np.zeros([l7out.shape[0],neglocs.shape[-1]+replocs.shape[-1],2])
    y[:,:-nsamples*npos,0] = 1. 
    y[:,-nsamples*npos:,1] = 1.
    y = np.reshape(y,[-1,y.shape[-1]])
        
#     excount = step*conf.batch_size
#     cur_lr = learning_rate * \
#             conf.gamma**math.floor(excount/conf.step_size)
#     feed_dict[phDict['learning_rate']] = cur_lr
    feed_dict[phDict['y']] = y
    feed_dict[phDict['lin']] = alldd
    feed_dict[phDict['locs']] = alllocs
    return retlocs


# In[ ]:

def train(conf,restore=True):
    
    phDict = createEvalPH(conf)
    feed_dict = createFeedDict(phDict)
    feed_dict[phDict['phase_train']] = True
    feed_dict[phDict['dropout']] = 0.5
    with tf.variable_scope('poseEval'):
        out,layer_dict = poseEvalNet(phDict['lin'],phDict['locs'],
                                     conf,phDict['phase_train'],
                                     phDict['dropout'])
        
    evalSaver = createEvalSaver(conf)
    y = phDict['y']
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(out,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    baseNet = PoseTools.create_network(conf, 1)
    baseNet.openDBs()
    
    with baseNet.env.begin() as txn,baseNet.valenv.begin() as valtxn,tf.Session() as sess:

        baseNet.createCursors()
        baseNet.restoreBase(sess,True)
        didRestore,startat = restoreEval(sess,conf,evalSaver,restore)
        baseNet.initializeRemainingVars(sess)
        for step in range(startat,conf.eval_training_iters+1):
            prepareOpt(baseNet,baseNet.DBType.Train,feed_dict,sess,conf,
                       phDict,distort=True)
#             baseNet.feed_dict[baseNet.ph['keep_prob']] = 0.5
            feed_dict[phDict['phase_train']] = True
            feed_dict[phDict['dropout']] = 0.5
            sess.run(train_step, feed_dict=feed_dict)

            if step % 25 == 0:
                prepareOpt(baseNet,baseNet.DBType.Train,feed_dict,
                           sess,conf,phDict,distort=False)
#                 baseNet.feed_dict[baseNet.ph['keep_prob']] = 1
                feed_dict[phDict['phase_train']] = False
                feed_dict[phDict['dropout']] = 1
                train_cross_ent = sess.run(cross_entropy, feed_dict=feed_dict)
                test_cross_ent = 0
                test_acc = 0
                pos_acc = 0
                pred_acc_pos = 0
                pred_acc_pred = 0
                
                #generate blocks for different neg types
                nsamples = 10
                ntypes = 3
                npos = 2
                blk = np.arange(nsamples)
                inter = np.arange(0,conf.batch_size*nsamples*(ntypes+npos),nsamples*(ntypes+npos))
                n_inters = blk + inter[:,np.newaxis]
                n_inters = n_inters.flatten()
                nacc = np.zeros(ntypes)
                nclose = 0 
                in_locs = np.array([])
                nrep = 40
                for rep in range(nrep):
                    prepareOpt(baseNet,baseNet.DBType.Val,feed_dict,sess,conf,
                               phDict,distort=False)
                    test_cross_ent += sess.run(cross_entropy, feed_dict=feed_dict)
                    test_acc += sess.run(accuracy, feed_dict = feed_dict)
                    tout = sess.run(correct_prediction, feed_dict=feed_dict)
                    labels = feed_dict[phDict['y']]
                    pos_acc += old_div(float(np.count_nonzero(tout[labels[:,1]>0.5])),nsamples)
                    for nt in range(ntypes):
                        nacc[nt] += old_div(float(np.count_nonzero(tout[n_inters+nt*nsamples])),nsamples)
                    
                    tdd = feed_dict[phDict['lin']]
                    tlocs = feed_dict[phDict['locs']]
                    ty = feed_dict[phDict['y']]
                    # 
                    l7 = baseNet.baseLayers['conv7']
                    curpred = sess.run([baseNet.basePred,l7], feed_dict=baseNet.feed_dict)
                    baseLocs = PoseTools.get_base_pred_locs(curpred[0], conf)

                    neglocs = baseLocs[:,:,:,np.newaxis]
                    locs = np.array(baseNet.locs)[...,np.newaxis]
                    d2locs = np.sqrt( np.sum((neglocs-locs)**2,axis=(1,2,3)))
                    alllocs = np.concatenate([neglocs,locs],axis=3)
                    alldd = genData(curpred[1],alllocs,conf)
                    alllocs = alllocs.transpose([0,3,1,2])
                    alllocs = alllocs.reshape((-1,)+alllocs.shape[2:])
                    alllocs_m = alllocs.mean(1)
                    alllocs = alllocs-alllocs_m[:,np.newaxis,:]

                    alldd = alldd.transpose([0,3,1,2])
                    alldd = np.reshape(alldd,[-1,alldd.shape[-2],alldd.shape[-1]])

                    y = np.zeros([curpred[0].shape[0],alllocs.shape[-1],2])
                    y[d2locs>=25,:-1,0] = 1. 
                    y[d2locs<25,:-1,1] = 1. 
                    y[:,-1,1] = 1.
                    y = np.reshape(y,[-1,y.shape[-1]])

                    feed_dict[phDict['y']] = y
                    feed_dict[phDict['lin']] = alldd
                    feed_dict[phDict['locs']] = alllocs

                    corrpred = sess.run(correct_prediction,feed_dict=feed_dict)
                    pred_acc_pos += np.count_nonzero(corrpred[1::2])
                    pred_acc_pred += np.count_nonzero(corrpred[0::2])
                    nclose += np.count_nonzero(d2locs<25)
                    er_locs = ~corrpred[0::2]
                    in_locs = np.append(in_locs,d2locs[er_locs])
                print("Iter:{:d}, train:{:.4f} test:{:.4f} acc:{:.2f} posacc:{:.2f}".format(step,
                                                 train_cross_ent,old_div(test_cross_ent,nrep),
                                                 old_div(test_acc,nrep),pos_acc/nrep/conf.batch_size/npos))
                print("Neg:{}".format(nacc/nrep/conf.batch_size))
                print("Pred Acc Pos:{},Pred Acc Pred:{},numclose:{}".format(float(pred_acc_pos)/nrep/conf.batch_size,
                                                                            float(pred_acc_pred)/nrep/conf.batch_size,
                                                                            float(nclose)/nrep/conf.batch_size))
                print('Distance of incorrect predictions:{}'.format(in_locs))
                
            if step % 100 == 0:
                saveEval(sess,step,evalSaver,conf)

