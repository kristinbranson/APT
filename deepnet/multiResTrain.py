
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
import tempfile,copy,re

sys.path.append('/home/mayank/work/caffe/python')
sys.path.append('/home/mayank/work/pyutils')

import caffe,lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math,cv2,scipy,time,pickle
from cvc import cvc
import numpy as np

import multiPawTools,myutils,multiResData

from convNetBase import *


# In[1]:


def trainBase(conf,resume=True):
    # Parameters
    learning_rate = conf.base_learning_rate;  training_iters = conf.base_training_iters
    batch_size = conf.batch_size;        display_step = conf.display_step

    # Network Parameters
    n_input = conf.psz; n_classes = conf.n_classes; dropout = conf.dropout 
    imsz = conf.imsz;   rescale = conf.rescale;     scale = conf.scale
    pool_scale = conf.pool_scale
    
    x0,x1,x2,y,keep_prob = createPlaceHolders(imsz,
                              rescale,scale,pool_scale,n_classes)
    learning_rate_ph = tf.placeholder(tf.float32,shape=[])
    
    weights = initNetConvWeights(conf)
    # Construct model
    pred,layers = net_multi_conv(x0,x1,x2, weights, keep_prob,
                          imsz,rescale,pool_scale)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.l2_loss(pred- y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph).minimize(cost)

    # training data stuff
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    env = lmdb.open(lmdbfilename, readonly = True)
    valenv = lmdb.open(vallmdbfilename, readonly = True)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver(max_to_keep=conf.maxckpt)

    trainData = {'train_err':[],'val_err':[],'step_no':[]}
    with env.begin() as txn,valenv.begin() as valtxn:
        train_cursor = txn.cursor(); val_cursor = valtxn.cursor()
        nTrain = txn.stat()['entries']
        with tf.Session() as sess:

            outfilename = os.path.join(conf.cachedir,conf.outname)
            traindatafilename = os.path.join(conf.cachedir,conf.databasename)
            latest_ckpt = tf.train.get_checkpoint_state(conf.cachedir,
                                                latest_filename = conf.ckptbasename)
            if not latest_ckpt or not resume:
                sess.run(init)
                startat = 0
            else:
                saver.restore(latest_ckpt.model_checkpoint_path)
                matchObj = re.match(outfilename + '-(\d*)',ckpt.model_checkpoint_path)
                startat = int(matchObj.group(1)+1)
                tdfile = open(traindatafilename,'rb')
                trainData = pickle.load(tdfile)
                tdfile.close()
                
            read_time = 0.; proc_time = 0.; opt_time = 0.
            # Keep training until reach max iterations
            for step in range(startat,training_iters):
                excount = step*batch_size
                cur_lr = learning_rate *                         conf.gamma**math.floor(old_div(excount,conf.step_size))
                
                r_start = time.clock()
                batch_xs, locs = multiPawTools.readLMDB(train_cursor,
                                        batch_size,imsz,multiResData)
                r_end = time.clock()
                
                locs = multiResData.sanitize_locs(locs)
                
                x0_in,x1_in,x2_in = multiPawTools.multiScaleImages(
                    batch_xs.transpose([0,2,3,1]),rescale,scale)
                
                labelims = multiPawTools.createLabelImages(locs,
                                   conf.imsz,conf.pool_scale*conf.rescale,
                                   conf.label_blur_rad) 
                p_end = time.clock()
                sess.run(optimizer, 
                         feed_dict={x0: x0_in, x1: x1_in,
                                    x2: x2_in, y: labelims, 
                                    keep_prob: dropout, learning_rate_ph:cur_lr})
                o_end = time.clock()
                
                read_time += r_end-r_start
                proc_time += p_end-r_end
                opt_time += o_end-p_end

                if step % display_step == 0:
                    train_loss = sess.run(cost, feed_dict={x0:x0_in,
                                                     x1:x1_in,
                                                     x2:x2_in,
                                               y: labelims, keep_prob: 1.})
                    train_loss /= batch_size
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
                    loss = old_div((old_div(loss,numrep)),batch_size)
                    print("Iter " + str(step))
                    print("  Training Loss= " + "{:.6f}".format(train_loss) +                          ", Minibatch Loss= " + "{:.6f}".format(loss)) 
                    print("  Read Time:" + "{:.2f}, ".format(old_div(read_time,(step+1))) +                           "Proc Time:" + "{:.2f}, ".format(old_div(proc_time,(step+1))) +                           "Opt Time:" + "{:.2f}".format(old_div(opt_time,(step+1)))) 
                    trainData['train_err'].append(train_loss)        
                    trainData['val_err'].append(loss)        
                    trainData['step_no'].append(step)        
                    
                if step % conf.save_step == 0:
                    saver.save(sess,outfilename,global_step=step,
                               latest_filename = conf.ckptbasename)
                    print('Saved state to %s-%d' %(outfilename,step))
                    tdfile = open(traindatafilename,'wb')
                    pickle.dump(trainData,tdfile)
                    tdfile.close()
                    
                step += 1
            print("Optimization Finished!")
            saver.save(sess,outfilename,global_step=step,
                       latest_filename = conf.ckptbasename)
            print('Saved state to %s-%d' %(outfilename,step))



# In[ ]:


def trainFine(conf,jointTrain=False,resume=True):
    # Parameters
    learning_rate = conf.fine_learning_rate;  
    batch_size = conf.fine_batch_size;        display_step = conf.display_step
    n_input = conf.psz; n_classes = conf.n_classes; dropout = conf.dropout 
    imsz = conf.imsz;   rescale = conf.rescale;     scale = conf.scale
    pool_scale = conf.pool_scale
    
    x0,x1,x2,y,keep_prob = createPlaceHolders(imsz,rescale,scale,pool_scale,n_classes)
    locs_ph = tf.placeholder(tf.float32,[conf.batch_size,n_classes,2])
    learning_rate_ph = tf.placeholder(tf.float32,shape=[])

    weights = initNetConvWeights(conf)
    pred_gradient,layers = net_multi_conv(x0,x1,x2, weights, keep_prob,
                          imsz,rescale,pool_scale)
    
    baseoutname = '%s_%d.ckpt'%(conf.outname,conf.base_training_iters)
    basemodelfile = os.path.join(conf.cachedir,baseoutname)

    sess = tf.Session()
    saver = tf.train.Saver()

    pred = tf.stop_gradient(pred_gradient)
    training_iters = conf.fine_training_iters
    outname = conf.fineoutname
    print("Restoring base model from:" + basemodelfile)
    saver.restore(sess, basemodelfile)
        
    # Construct fine model
    labelT  = multiPawTools.createFineLabelTensor(conf)
    layer1_1 = tf.stop_gradient(layers['base_dict_0']['conv1'])
    layer1_2 = tf.stop_gradient(layers['base_dict_0']['conv2'])
    layer2_1 = tf.stop_gradient(layers['base_dict_1']['conv1'])
    layer2_2 = tf.stop_gradient(layers['base_dict_1']['conv2'])
    curfine1_1 = extractPatches(layer1_1,pred,conf,1,4)
    curfine1_2 = extractPatches(layer1_2,pred,conf,2,2)
    curfine2_1 = extractPatches(layer2_1,pred,conf,2,2)
    curfine2_2 = extractPatches(layer2_2,pred,conf,4,1)
    curfine1_1u = tf.unpack(tf.transpose(curfine1_1,[1,0,2,3,4]))
    curfine1_2u = tf.unpack(tf.transpose(curfine1_2,[1,0,2,3,4]))
    curfine2_1u = tf.unpack(tf.transpose(curfine2_1,[1,0,2,3,4]))
    curfine2_2u = tf.unpack(tf.transpose(curfine2_2,[1,0,2,3,4]))
    finepred = fineOut(curfine1_1u,curfine1_2u,curfine2_1u,curfine2_2u,conf)    
    limgs = multiPawTools.createFineLabelImages(locs_ph,pred,conf,labelT)

    # training data stuff
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    env = lmdb.open(lmdbfilename, readonly = True)
    valenv = lmdb.open(vallmdbfilename, readonly = True)

    # Define loss and optimizer
    costFine = tf.reduce_mean(tf.nn.l2_loss(finepred- tf.to_float(limgs)))
    costBase =  tf.reduce_mean(tf.nn.l2_loss(pred- y))

    cost = costFine

    saver1 = tf.train.Saver(max_to_keep=conf.maxckpt)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph).minimize(cost)

    outfilename = os.path.join(conf.cachedir,conf.fineoutname)
    traindatafilename = os.path.join(conf.cachedir,conf.datafinename)
    latest_ckpt = tf.train.get_checkpoint_state(conf.cachedir,
                                        latest_filename = conf.ckptfinename)
    
    if not latest_ckpt or not resume:
        startat = 0
        trainData = {'train_err':[],'val_err':[],'step_no':[]}
        varlist = tf.all_variables()
        for var in varlist:
            try:
                sess.run(tf.assert_variables_initialized([var]))
            except tf.errors.FailedPreconditionError:
                sess.run(tf.initialize_variables([var]))

    else:
        saver1.restore(latest_ckpt.model_checkpoint_path)
        matchObj = re.match(outfilename + '-(\d*)',ckpt.model_checkpoint_path)
        startat = int(matchObj.group(1)+1)
        tdfile = open(traindatafilename,'rb')
        trainData = pickle.load(tdfile)
        tdfile.close()


#             print('Initializing variable %s'%var.name)
            
#     init = tf.initialize_all_variables()
#     sess.run(init)

    with env.begin() as txn,valenv.begin() as valtxn:
        train_cursor = txn.cursor(); val_cursor = valtxn.cursor()

        # Keep training until reach max iterations
        for step in range(startat,training_iters):
            excount = step*batch_size
            cur_lr = learning_rate *                     conf.gamma**math.floor(old_div(excount,conf.step_size))

            batch_xs, locs = multiPawTools.readLMDB(train_cursor,
                                    batch_size,imsz,multiResData)

            locs = multiResData.sanitize_locs(locs)

            x0_in,x1_in,x2_in = multiPawTools.iScaleImages(
                batch_xs.transpose([0,2,3,1]),rescale,scale)

            labelims = multiPawTools.createLabelImages(locs,
                               conf.imsz,conf.pool_scale*conf.rescale,
                               conf.label_blur_rad) 
            feed_dict={x0: x0_in,x1: x1_in,x2: x2_in,
                y: labelims, keep_prob: dropout,locs_ph:np.array(locs),
                learning_rate_ph:cur_lr}
            sess.run(optimizer, feed_dict = feed_dict)

            if step % display_step == 0:
                feed_dict={x0: x0_in,x1: x1_in,x2: x2_in,
                    y: labelims, keep_prob: 1.,locs_ph:np.array(locs)}
                train_loss = sess.run([cost,costBase], feed_dict=feed_dict)

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
                    feed_dict={x0: x0_in,x1: x1_in,x2: x2_in,
                        y: labelims, keep_prob:1.,locs_ph:np.array(locs)}
                    loss += sess.run(cost, feed_dict=feed_dict)
                loss = old_div((old_div(loss,numrep)),batch_size)
                print("Iter " + str(step) +                 "  Minibatch Loss= " + "{:.3f}".format(loss) +                  ", Training Loss= " + "{:.3f}".format(old_div(train_loss[0],batch_size)) +                  ", Base Training Loss= " + "{:.3f}".format(old_div(train_loss[1],batch_size)))
                trainData['train_err'].append(old_div(train_loss[0],batch_size))
                trainData['val_err'].append(loss)
                trainData['step_no'].append(step)

            if step % conf.save_step == 0:
                saver1.save(sess,outfilename,global_step=step,
                           latest_filename = conf.ckptfinename)
                print('Saved state to %s-%d' %(outfilename,step))
                tdfile = open(traindatafilename,'wb')
                pickle.dump(trainData,tdfile)
                tdfile.close()
#             if step % conf.save_step == 0:
#                 curoutname = '%s_%d.ckpt'% (outname,step)
#                 outfilename = os.path.join(conf.cachedir,curoutname)
#                 saver1.save(sess,outfilename)
#                 print('Saved state to %s' %(outfilename))

            step += 1
            
        print("Optimization Finished!")
        saver1.save(sess,outfilename,global_step=step,
                   latest_filename = conf.ckptfinename)
        print('Saved state to %s-%d' %(outfilename,step))
        tdfile = open(traindatafilename,'wb')
        pickle.dump(trainData,tdfile)
        tdfile.close()
    
    sess.close()


# In[1]:

def extractPatches(layer,out,conf,scale,outscale):
    hsz = conf.fine_sz/scale/2
    padsz = tf.constant([[0,0],[hsz, hsz],[hsz,hsz],[0,0]])
    patchsz = tf.to_int32([old_div(conf.fine_sz,scale),old_div(conf.fine_sz,scale),-1])

    patches = []
    maxloc = multiPawTools.argmax2d(out)*outscale
    padlayer = tf.pad(layer,padsz)
    for inum in range(conf.batch_size):
        curpatches = []
        for ndx in range(conf.n_classes):
            curloc = tf.concat(0,[tf.squeeze(maxloc[:,inum,ndx]),[0]])
            curpatches.append(tf.slice(padlayer[inum,:,:,:],curloc,patchsz))
        patches.append(tf.pack(curpatches))
    return tf.pack(patches)


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
    nframes = int(cap.get(cvc.FRAME_COUNT))
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

