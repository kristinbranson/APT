from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[2]:

from builtins import zip
from builtins import range
from past.utils import old_div
import tensorflow as tf
import os,sys
import matplotlib
import matplotlib.pyplot as plt
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
import multiResData


# In[3]:

def conv_relu(X, kernel_shape, conv_std,bias_val,doBatchNorm,trainPhase,addSummary=True):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.contrib.layers.xavier_initializer())
                              #tf.random_normal_initializer(stddev=conv_std))
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

def conv_relu_norm_init(X, kernel_shape, conv_std,bias_val,doBatchNorm,trainPhase,addSummary=True):
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

def FC_2D(S,nfilt,trainPhase,addSummary=True):
    
    inDim = S.get_shape()[1]
    weights = tf.get_variable("weights", [inDim,nfilt],
        initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", nfilt,
        initializer=tf.constant_initializer(0))
    if addSummary:
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(biases)

    fc_out = tf.nn.relu(batch_norm_2D(tf.matmul(S, weights),trainPhase)-biases)
    with tf.variable_scope('fc'):
        PoseTools.variable_summaries(fc_out)
    return fc_out

def FC_2D_norm_init(S,nfilt,trainPhase,conv_std,addSummary=True):
    
    inDim = S.get_shape()[1]
    weights = tf.get_variable("weights", [inDim,nfilt],
        initializer=tf.random_normal_initializer(stddev=conv_std))
    biases = tf.get_variable("biases", nfilt,
        initializer=tf.constant_initializer(0))
    if addSummary:
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(biases)

    fc_out = tf.nn.relu(batch_norm_2D(tf.matmul(S, weights),trainPhase)-biases)
    with tf.variable_scope('fc'):
        PoseTools.variable_summaries(fc_out)
    return fc_out


def max_pool(name, l_input, k,s):
    return tf.nn.max_pool(
        l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], 
        padding='SAME', name=name)

def createPlaceHolders(conf):
#     imsz = conf.imsz
    # tf Graph input
    imsz = conf.imsz
    psz = conf.poseEval2_psz
    rescale = conf.rescale
    scale = conf.scale
    pool_scale = conf.pool_scale
    n_classes = conf.n_classes
    imgDim = conf.imgDim
    
    inScale = rescale
    
    nex = conf.batch_size*(conf.eval_num_neg+1)
    x0 = []
    x1 = []
    x2 = []
    s0 = []
    a0 = []
    d0 = []

    for ndx in range(n_classes):
        x0.append(tf.placeholder(tf.float32, [nex,psz,psz,imgDim],name='x0_{}'.format(ndx)))
        x1.append(tf.placeholder(tf.float32, [nex,psz,psz,imgDim],name='x1_{}'.format(ndx)))
        x2.append(tf.placeholder(tf.float32, [nex,psz,psz,imgDim],name='x2_{}'.format(ndx)))
        s0.append(tf.placeholder(tf.float32, [nex,2*n_classes],name='s0_{}'.format(ndx)))
        a0.append(tf.placeholder(tf.float32, [nex,8*n_classes],name='a0_{}'.format(ndx)))
        d0.append(tf.placeholder(tf.float32, [nex,n_classes],name='d0_{}'.format(ndx)))

    y = tf.placeholder(tf.float32, [nex,n_classes,],'out')
    
    X = [x0,x1,x2]
    S = [s0,a0,d0]
#change 2 21022017    
    
    phase_train = tf.placeholder(tf.bool,name='phase_train')
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')
    
    ph = {'X':X,'S':S,'y':y,
          'phase_train':phase_train,'learning_rate':learning_rate}
    return ph


def createFeedDict(ph,conf):
    feed_dict = {}
    for ndx in range(conf.n_classes):
        feed_dict[ph['X'][0][ndx]] = []
        feed_dict[ph['X'][1][ndx]] = []
        feed_dict[ph['X'][2][ndx]] = []
        feed_dict[ph['S'][0][ndx]] = []
        feed_dict[ph['S'][1][ndx]] = []
        feed_dict[ph['S'][2][ndx]] = []
    feed_dict[ph['y']]=[]
    feed_dict[ph['learning_rate']]=1
    feed_dict[ph['phase_train']]=False
    
    return feed_dict

def net_multi_base_named(X,nfilt,doBatchNorm,trainPhase,addSummary=True):
    inDimX = X.get_shape()[3]
    nex = X.get_shape()[0].value
    
    with tf.variable_scope('layer1_X'):
        conv1 = conv_relu_norm_init(X,[5, 5, inDimX, 48],0.3,0,doBatchNorm,trainPhase,addSummary)
#         pool1 = max_pool('pool1',conv1,k=3,s=2)
        pool1 = conv1
            
    with tf.variable_scope('layer2'):
        conv2 = conv_relu(pool1,[3,3,48,nfilt],0.01,0,doBatchNorm,trainPhase,addSummary)
#         pool2 = max_pool('pool2',conv2,k=3,s=2)
        pool2 = conv2
            
    with tf.variable_scope('layer3'):
        conv3 = conv_relu(pool2,[3,3,nfilt,nfilt],0.01,0,doBatchNorm,trainPhase,addSummary)
        pool3 = max_pool('pool3',conv3,k=3,s=2)
        
    with tf.variable_scope('layer4'):
        conv4 = conv_relu(pool3,[3,3,nfilt,nfilt],0.01,0,doBatchNorm,trainPhase,addSummary)
        pool4 = max_pool('pool4',conv4,k=3,s=2)
    conv4_reshape = tf.reshape(pool4,[nex,-1])
    conv4_dims = conv4_reshape.get_shape()[1].value
        
    with tf.variable_scope('layer5'):
        conv5 = FC_2D_norm_init(conv4_reshape,128,trainPhase,0.01,addSummary)
        
    out_dict = {'conv1':conv1,'conv2':conv2,'conv3':conv3,
               'conv4':conv4,'conv5':conv5}
    return conv5,out_dict
        
def net_pose(S,nfilt,trainPhase):
    inDim = S.get_shape()[1]
    with tf.variable_scope('layer1_pose'):
        L1 = FC_2D(S,nfilt,trainPhase,True)
    with tf.variable_scope('layer2_pose'):
        L2 = FC_2D(L1,nfilt,trainPhase,True)
    with tf.variable_scope('layer3_pose'):
        L3 = FC_2D(L2,nfilt,trainPhase,True)
    out_dict = {'L1':L1,'L2':L2,'L3':L3}
    return L3,out_dict
    
def net_multi_conv(ph,conf):
    X = ph['X']
    S = ph['S']
    X0,X1,X2 = X
    S0,A0,D0 = S
    
    out_size = ph['y'].get_shape()[1]
    trainPhase = ph['phase_train']
    
    imsz = conf.imsz; rescale = conf.rescale
    pool_scale = conf.pool_scale
    nfilt = conf.nfilt
    doBatchNorm = conf.doBatchNorm
    
    L7_array = []
    base_dict_array = []
    for ndx in range(conf.n_classes):
    
        with tf.variable_scope('scale0_{}'.format(ndx)):
            conv5_0,base_dict_0 = net_multi_base_named(X0[ndx],nfilt,doBatchNorm,trainPhase,True)
        with tf.variable_scope('scale1_{}'.format(ndx)):
            conv5_1,base_dict_1 = net_multi_base_named(X1[ndx],nfilt,doBatchNorm,trainPhase,False)
        with tf.variable_scope('scale2_{}'.format(ndx)):
            conv5_2,base_dict_2 = net_multi_base_named(X2[ndx],nfilt,doBatchNorm,trainPhase,False)
        with tf.variable_scope('pose_s_fc_{}'.format(ndx)):
            L3_S,s_dict = net_pose(S0[ndx],nfilt,trainPhase)
        with tf.variable_scope('pose_a_fc_{}'.format(ndx)):
            L3_A,a_dict = net_pose(A0[ndx],nfilt,trainPhase)
        with tf.variable_scope('pose_d_fc_{}'.format(ndx)):
            L3_D,d_dict = net_pose(D0[ndx],nfilt,trainPhase)
        
#         conv5_cat = tf.concat(1,[conv5_0,conv5_1,conv5_2,L3_S,L3_A,L3_D])
#change 1 20022017
        conv5_cat = tf.concat([old_div(L3_S,3),old_div(L3_A,3),old_div(L3_D,3)],1)
    
        with tf.variable_scope('L6_{}'.format(ndx)):
            L6 = FC_2D(conv5_cat,128,trainPhase)
        with tf.variable_scope('L7_{}'.format(ndx)):
            L7 = FC_2D(L6,32,trainPhase)
        L7_array.append(L7)
        base_dict_array.append([base_dict_0,base_dict_2,base_dict_2,L6,L7])
    
    L7_cat = tf.concat(L7_array,1)
    with tf.variable_scope('L8'.format(ndx)):
        L8 = FC_2D(L7_cat,128,trainPhase)
        
    with tf.variable_scope('out'.format(ndx)):
        weights = tf.get_variable("weights", [L8.get_shape()[1].value,out_size],
            initializer=tf.random_normal_initializer(stddev=.05)) # tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", out_size,
            initializer=tf.constant_initializer(0))
        with tf.variable_scope('weights'):
            PoseTools.variable_summaries(weights)
        with tf.variable_scope('biases'):
            PoseTools.variable_summaries(biases)

        out = tf.matmul(L8, weights)-biases
        with tf.variable_scope('conv'):
            PoseTools.variable_summaries(out)

    out_dict = {'base_dict_array':base_dict_array,
                'L8':L8}
    
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

    count = 0
    while count < conf.batch_size:
        [curxs,curlocs] = sess.run(cur_data)
        
        # kk = curlocs[conf.eval2_selpt2,:]-curlocs[conf.eval2_selpt1,:]
        # dd = np.sqrt(kk[0]**2 + kk[1]**2 )
#         if dd>150:
#             continue

        if np.ndim(curxs)<3:
            xs.append(curxs[np.newaxis,:,:])
        else:
            xs.append(curxs)
        locs.append(curlocs)
        count = count+1
    
        
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



# In[2]:

def updateFeedDict(conf,dbType,distort,sess,data,feed_dict,ph):
    xs,locs = readImages(conf,dbType,distort,sess,data)
#     global shape_prior
    
    minlen = conf.eval_minlen
    nlocs = genNegSamples(locs,conf,minlen=minlen)
    alllocs = nlocs
    dd = alllocs-locs # distance of neg points to actual locations
    ddist = np.sqrt(np.sum(dd**2,axis=2))


    ind_labels = ddist/minlen/2
    ind_labels = 1-2*ind_labels
    ind_labels = ind_labels.clip(min=-1,max=1)

    psz = conf.poseEval2_psz
    x0,x1,x2 = PoseTools.multi_scale_images(xs.transpose([0, 2, 3, 1]),
                                            conf.rescale, conf.scale, conf.l1_cropsz, conf)
#     jj = np.random.randint(3)
#change 3 22022017
    jj = 2
    if jj is 0:
        locs_patch = nlocs
        locs_coords = nlocs
    elif jj is 1:
        locs_patch = nlocs
        locs_coords = locs
    elif jj is 2:
        locs_patch = locs
        locs_coords = nlocs
        
    ang = angle_from_locs(locs_coords)
    dist = dist_from_locs(locs_coords)
    for ndx in range(conf.n_classes):
        feed_dict[ph['X'][0][ndx]] = extract_patches(x0,locs_patch[:,ndx,:],psz)
        feed_dict[ph['X'][1][ndx]] = extract_patches(x1,old_div((locs_patch[:,ndx,:]),conf.scale),psz)
        feed_dict[ph['X'][2][ndx]] = extract_patches(x2,old_div((locs_patch[:,ndx,:]),(conf.scale**2)),psz)
        feed_dict[ph['S'][0][ndx]] = np.reshape(locs_coords-locs_coords[:,ndx:ndx+1,:],[conf.batch_size,2*conf.n_classes])
        feed_dict[ph['S'][1][ndx]] = np.reshape(ang[:,ndx,:,:],[conf.batch_size,8*conf.n_classes])
        feed_dict[ph['S'][2][ndx]] = dist[:,ndx,:]
    feed_dict[ph['y']] = np.reshape(ind_labels,[conf.batch_size,-1])
    return alllocs,locs,xs

def angle_from_locs(locs):
    norts = 8
    npts = locs.shape[1]
    bsz = locs.shape[0]
    yy = np.zeros([bsz,npts,npts,norts])
    for ndx in range(bsz):
        curl = locs[ndx,...]
        rloc = np.tile(curl, [npts,1,1] )
        kk = rloc - curl[:,np.newaxis,:]
        aa = np.arctan2( kk[:,:,1], kk[:,:,0]+1e-5 )*180/np.pi + 180
        for i in range(norts):
            pndx = np.abs(np.mod(aa-360/norts*i+45,360)-45-22.5)<32.5
            zz = np.zeros([npts,npts])
            zz[pndx] = 1
            yy[ndx,...,i] = zz
    return yy

def dist_from_locs(locs):
    npts = locs.shape[1]
    bsz = locs.shape[0]
    yy = np.zeros([bsz,npts,npts])
    for ndx in range(bsz):
        curl = locs[ndx,...]
        dd = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(curl))
        yy[ndx,...] = dd
    return yy

def shape_from_locs(locs):
    # shape context kinda labels 
    n_angle = 8
#     r_bins = np.logspace(4,10,n_radius-1,base=2)
#     r_bins = np.concatenate([[0,],r_bins,[np.inf,]])
#     r_bins = np.array([0,64,128,256,np.inf])
    r_bins = np.array([0,np.inf])
    n_radius = len(r_bins)-1
#     r_bins_high = np.array([90,150,290,inf])
    npts = locs.shape[1]
    bsz = locs.shape[0]
    yy = np.zeros([bsz,npts,npts,n_angle,n_radius])
    for ndx in range(bsz):
        curl = locs[ndx,...]
        dd = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(curl))
        dd_bins = np.digitize(dd,r_bins)
        rloc = np.tile(curl, [npts,1,1] )
        kk = rloc - curl[:,np.newaxis,:]
        aa = np.arctan2( kk[:,:,1], kk[:,:,0]+1e-5 )*180/np.pi + 180
        
        for i in range(n_angle):
            for dbin in range(n_radius):
                pndx = np.abs(np.mod(aa-360/n_angle*i+45,360)-45-22.5)<=22.5
                zndx = pndx & (dd_bins == dbin+1)
                zz = np.zeros([npts,npts])
                zz[zndx] = 1
                yy[ndx,...,i,dbin] = zz
    return yy
    
    

def extract_patches(img,locs,psz):
    zz = []
    pad_arg = [(psz,psz),(psz,psz),(0,0)]
    locs = np.round(locs).astype('int')
    for ndx in range(img.shape[0]):
        pimg = np.pad(img[ndx,...],pad_arg,'constant')
        zz.append( pimg[(locs[ndx,1]+psz-old_div(psz,2)):(locs[ndx,1]+psz+old_div(psz,2)),
                         (locs[ndx,0]+psz-old_div(psz,2)):(locs[ndx,0]+psz+old_div(psz,2)),:]);
    return np.array(zz)


# In[ ]:

def init_shape_prior(conf):
#     global shape_prior
    L = h5py.File(conf.labelfile,'r')
    
    if 'pts' in L:
        pts = np.array(L['pts'])
        v = conf.view
    else:
        pp = np.array(L['labeledpos'])
        nmovie = pp.shape[1]
        pts = np.zeros([0,conf.n_classes,2])
        v = 0
        for ndx in range(nmovie):
            curpts = np.array(L[pp[0,ndx]])
            frames = np.where(np.invert( np.any(np.isnan(curpts),axis=(1,2))))[0]
            nptsPerView = np.array(L['cfg']['NumLabelPoints'])[0,0]
            pts_st = int(conf.view*nptsPerView)
            selpts = pts_st + conf.selpts
            curlocs = curpts[:,:,selpts]
            curlocs = curlocs[frames,:,:]
            curlocs = curlocs.transpose([0,2,1])
            pts = np.append(pts,curlocs[:,:,:],axis=0)
    shape_prior = np.mean(shape_from_locs(pts)>0,axis=0)
    return shape_prior
    


# In[ ]:

def restoreEval(sess,evalsaver,restore,conf,feed_dict):
    outfilename = os.path.join(conf.cachedir,conf.eval2outname)
    latest_ckpt = tf.train.get_checkpoint_state(conf.cachedir,
                                        latest_filename = conf.eval2ckptname)
    sess.run(tf.global_variables_initializer(),feed_dict=feed_dict)
    if not latest_ckpt or not restore:
        evalstartat = 0
        print("Not loading Eval variables. Initializing them")
    else:
        evalsaver.restore(sess,latest_ckpt.model_checkpoint_path)
        matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
        evalstartat = int(matchObj.group(1))+1
        print("Loading eval variables from %s"%latest_ckpt.model_checkpoint_path)
    return  evalstartat
    

def saveEval(sess,evalsaver,step,conf):
    outfilename = os.path.join(conf.cachedir,conf.eval2outname)
    evalsaver.save(sess,outfilename,global_step=step,
               latest_filename = conf.eval2ckptname)
    print('Saved state to %s-%d' %(outfilename,step))

def createEvalSaver(conf):
    evalsaver = tf.train.Saver(var_list = PoseTools.get_vars('eval'),
                               max_to_keep=conf.maxckpt)
    return evalsaver

def initializeRemainingVars(sess,feed_dict):
    varlist = tf.global_variables()
#     for var in varlist:
#         try:
#             sess.run(tf.assert_variables_initialized([var]))
#         except tf.errors.FailedPreconditionError:
#             sess.run(tf.variables_initializer([var]))
#             print('Initializing variable:%s'%var.name)

#     with tf.variable_scope('',reuse=True):
#         varlist = sess.run( tf.report_uninitialized_variables( tf.global_variables( ) ) )
#         sess.run( tf.initialize_variables( list( tf.get_variable(name) for name in  varlist) ) )

def print_gradients(sess,feed_dict,loss):
    vv = tf.global_variables()
    aa = [v for v in vv if not re.search('Adam|batch_norm|beta|scale[1-2]|scale0_[1-9][0-9]*|fc_[1-9][0-9]*|L[6-7]_[1-9][0-9]*|biases',v.name)]

    grads = sess.run(tf.gradients(loss,aa),feed_dict=feed_dict)

    wts = sess.run(aa,feed_dict=feed_dict)

    grads_std = [g.std() for g in grads]
    wts_std = [w.std() for w in wts]

    grads_by_wts = [old_div(s,w) for s,w in zip(grads_std,wts_std)]



    bb = [[r,n.name] for r,n in zip(grads_by_wts,aa)]
    for b,k,g in zip(bb,grads_std,wts_std):
        print(b,k,g)


# In[ ]:

def poseEvalNetInit(conf):
    
    ph = createPlaceHolders(conf)
    feed_dict = createFeedDict(ph,conf)
    init_shape_prior(conf)
    with tf.variable_scope('eval'):
        out,out_dict = net_multi_conv(ph,conf)
#change 3 22022017        
#         out,out_dict = net_multi_conv(ph,conf)
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
    evalSaver = createEvalSaver(conf) 
    
    shape_prior = init_shape_prior(conf)
    prior_tensor = tf.constant(shape_prior)
    y_re = tf.reshape(ph['y'],[conf.batch_size,8,conf.n_classes])
    selpt1 = conf.eval2_selpt1
    selpt2 = conf.eval2_selpt2
    wt_den = shape_prior[selpt1,:,:,0].transpose([1,0])
    wt = tf.reduce_max( old_div(y_re,(wt_den+0.1)),axis=(1,2))
    loss = tf.reduce_sum( tf.reduce_sum((out-ph['y'])**2,axis=1)*wt)
#     loss = tf.nn.l2_loss(out-ph['y'])
    correct_pred = tf.cast(tf.equal(out>0.5,ph['y']>0.5),tf.float32)
    accuracy = tf.reduce_mean(correct_pred)
    
#     tf.summary.scalar('cross_entropy',loss)
#     tf.summary.scalar('accuracy',accuracy)
    
    opt = tf.train.AdamOptimizer(learning_rate=                       ph['learning_rate']).minimize(loss)
    
    merged = tf.summary.merge_all()
    
    
    with tf.Session() as sess:
#         train_writer = tf.summary.FileWriter(conf.cachedir + '/eval2_train_summary',sess.graph)
#         test_writer = tf.summary.FileWriter(conf.cachedir + '/eval2_test_summary',sess.graph)
        data,coord,threads = createCursors(sess,queue,conf)
        updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
        evalstartat = restoreEval(sess,evalSaver,restore,conf,feed_dict)
        initializeRemainingVars(sess,feed_dict)
        for step in range(evalstartat,conf.eval2_training_iters+1):
            excount = step*conf.batch_size
            cur_lr = conf.eval2_learning_rate * conf.gamma**math.floor(old_div(excount,conf.step_size))
            feed_dict[ph['learning_rate']] = cur_lr
            feed_dict[ph['phase_train']] = True
            updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
            train_summary,_ = sess.run([merged,opt], feed_dict=feed_dict)
#             train_writer.add_summary(train_summary,step)

            if step % conf.display_step == 0:
                updateFeedDict(conf,'train',sess=sess,distort=True,data=data,feed_dict=feed_dict,ph=ph)
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
#                 test_writer.add_summary(test_summary,step)
                print('Val -- Acc:{:.4f} Loss:{:.4f} Train Acc:{:.4f} Loss:{:.4f} Iter:{}'.format(
                    val_acc,val_loss,train_acc,train_loss,step))
#                 print_gradients(sess,feed_dict,loss)
            if step % conf.save_step == 0:
                saveEval(sess,evalSaver,step,conf)
        print("Optimization Done!")
        saveEval(sess,evalSaver,step,conf)
        train_writer.close()
        test_writer.close()
        coord.request_stop()
        coord.join(threads)
        


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

def genNMovedNegSamples(locs,conf,minlen=8):
    # Move a random number of points.
    minlen = float(minlen)
    maxlen = 2*minlen
    minlen = 0
    
    rlocs = copy.deepcopy(locs)
    sz = conf.imsz

    for curi in range(locs.shape[0]):
        curN = np.random.randint(conf.n_classes)
        for rand_point in np.random.choice(conf.n_classes,size=[curN,],replace=False):
            rx = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                np.sign(np.random.rand()-0.5)
            ry = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                np.sign(np.random.rand()-0.5)

            rlocs[curi,rand_point,0] = locs[curi,rand_point,0] + rx
            rlocs[curi,rand_point,1] = locs[curi,rand_point,1] + ry
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0] = xlocs
    ylocs = rlocs[:,:,1]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1] = ylocs
    return rlocs


# In[ ]:

def genNegSamples(locs,conf,minlen=8):
    return genNMovedNegSamples(locs,conf,minlen)

