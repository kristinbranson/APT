
# coding: utf-8

# In[ ]:

'''
Mayank Oct 10 2016
'''
from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from past.utils import old_div
import tensorflow as tf
import os,sys,shutil
import tempfile,copy,re
from enum import Enum
import localSetup

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math,cv2,scipy,time,pickle
import numpy as np

import PoseTools,myutils,multiResData

import convNetBase as CNB
from batch_norm import batch_norm
import PoseTrain


# In[ ]:

class PoseRegression(PoseTrain.PoseTrain):
    
    def createPH(self):
        super(PoseRegression,self).createPH()
        scale = self.conf.rescale*self.conf.pool_scale
        lsz0 = old_div(self.conf.imsz[0],scale)
        lsz1 = old_div(self.conf.imsz[1],scale)
        n_classes = self.conf.n_classes
        regx_ph = tf.placeholder(tf.float32, [None, lsz0,lsz1,n_classes],'regx')
        self.ph['regx'] = regx_ph
        regy_ph = tf.placeholder(tf.float32, [None, lsz0,lsz1,n_classes],'regy')
        self.ph['regy'] = regy_ph

    
    def updateFeedDict(self,dbtype,distort):
        super(PoseRegression,self).updateFeedDict(dbtype,distort)
        
        labelims,regimsx,regimsy = PoseTools.create_reg_label_images(self.locs,
                                                                     self.conf.imsz,
                                                                     self.conf.pool_scale * self.conf.rescale,
                                                                     self.conf.label_blur_rad)
        self.feed_dict[self.ph['y']] = labelims
        self.feed_dict[self.ph['regx']] = regimsx
        self.feed_dict[self.ph['regy']] = regimsy
        
    def createLoss(self):
        label_cost = tf.nn.l2_loss(self.baseregPred-self.ph['y']) 
        norm_label = tf.sqrt(tf.maximum((old_div(self.ph['y'],2))+0.5,0))
        # sqrt so that in l2_loss it becomes what we want
        xcost = tf.mul(self.regpredx-self.ph['regx'],norm_label)
        ycost = tf.mul(self.regpredy-self.ph['regy'],norm_label)

        rad = self.conf.label_blur_rad*2*self.conf.pool_scale
        # dividing xcost and ycost by rad so that if we are off by rad in regression
        # it is equivalent to complete misprediction on labels
        reg_cost = tf.nn.l2_loss(old_div(xcost,rad)) + tf.nn.l2_loss(old_div(ycost,rad))
        
        reg_lambda = self.conf.reg_lambda
        self.cost = label_cost + reg_lambda*(reg_cost)

    def createBaseReg(self,doBatchNorm):
        pred,predx,predy,layers = CNB.net_multi_conv_reg(self.ph['x0'],self.ph['x1'],
                                         self.ph['x2'],self.ph['keep_prob'],
                                         self.conf,doBatchNorm,
                                         self.ph['phase_train_base']
                                        )
        self.baseregPred = pred
        self.regpredx = predx
        self.regpredy = predy
        self.baseLayers = layers
        
    def createBaseRegSaver(self):
        self.baseregsaver = tf.train.Saver(var_list = PoseTools.get_vars('regbase'),
                                           max_to_keep=self.conf.maxckpt)
        
    def restoreBaseReg(self,sess,restore):
        outfilename = os.path.join(self.conf.cachedir,self.conf.baseregoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.baseregdataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename = self.conf.baseregckptname)
        if not latest_ckpt or not restore:
            self.baseregstartat = 0
            self.baseregtrainData = {'train_err':[], 'val_err':[], 'step_no':[],
                                  'train_dist':[], 'val_dist':[] }
            sess.run(tf.initialize_variables(PoseTools.get_vars('base')))
            print("Not loading base variables. Initializing them")
            return False
        else:
            self.baseregsaver.restore(sess,latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.baseregstartat = int(matchObj.group(1))+1
            with open(traindatafilename,'rb') as tdfile:
                inData = pickle.load(tdfile)
                if not isinstance(inData,dict):
                    self.baseregtrainData, loadconf = inData
                    print('Parameters that dont match for base:')
                    PoseTools.compare_conf(self.conf, loadconf)
                else:
                    print("No config was stored for base. Not comparing conf")
                    self.baseregtrainData = inData
            print("Loading base variables from %s"%latest_ckpt.model_checkpoint_path)
            return True
            
    def saveBaseReg(self,sess,step):
        outfilename = os.path.join(self.conf.cachedir,self.conf.baseregoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.baseregdataname)
        self.baseregsaver.save(sess,outfilename,global_step=step,
                   latest_filename = self.conf.baseregckptname)
        print('Saved state to %s-%d' %(outfilename,step))
        with open(traindatafilename,'wb') as tdfile:
            pickle.dump([self.baseregtrainData,self.conf],tdfile)
            
    def updateBaseRegLoss(self,step,train_loss,val_loss,trainDist,valDist):
        print("Iter " + str(step) +              ", Train = " + "{:.3f},{:.1f}".format(train_loss[0],trainDist[0]) +              ", Val = " + "{:.3f},{:.1f}".format(val_loss[0],valDist[0]))
#         nstep = step-self.basestartat
#         print "  Read Time:" + "{:.2f}, ".format(self.read_time/(nstep+1)) + \
#               "Opt Time:" + "{:.2f}".format(self.opt_time/(nstep+1)) 
        self.baseregtrainData['train_err'].append(train_loss[0])      
        self.baseregtrainData['val_err'].append(val_loss[0])        
        self.baseregtrainData['step_no'].append(step)        
        self.baseregtrainData['train_dist'].append(trainDist[0])        
        self.baseregtrainData['val_dist'].append(valDist[0])        

        
    def baseRegress(self, restore=True, trainPhase=True):
        self.createPH()
        self.createFeedDict()
        self.feed_dict[self.ph['phase_train_base']] = trainPhase
        self.feed_dict[self.ph['keep_prob']] = 0.5
        doBatchNorm = self.conf.doBatchNorm
        
        with tf.variable_scope('regbase'):
            self.createBaseReg(doBatchNorm)
        
        self.createLoss()
        self.openDBs()
        self.createOptimizer()
        self.createBaseRegSaver()

        with self.env.begin() as txn,                 self.valenv.begin() as valtxn,                 tf.Session() as sess:
                    
            self.createCursors(txn,valtxn)
            self.restoreBaseReg(sess,restore)
            self.initializeRemainingVars(sess)
            
            for step in range(self.baseregstartat,self.conf.basereg_training_iters+1):
                self.feed_dict[self.ph['keep_prob']] = 0.5
                self.doOpt(sess,step,self.conf.base_learning_rate)
                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train,distort=True)
                    self.feed_dict[self.ph['keep_prob']] = 1.
                    train_loss = self.computeLoss(sess,[self.cost])
                    tt1 = self.computePredDist(sess,self.baseregPred)
                    trainDist = [tt1.mean()]
                    numrep = int(old_div(self.conf.numTest,self.conf.batch_size))+1
                    val_loss = np.zeros([2,])
                    valDist = [0.]
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val,distort=False)
                        val_loss += np.array(self.computeLoss(sess,[self.cost]))
                        tt1 = self.computePredDist(sess,self.baseregPred)
                        valDist = [valDist[0]+tt1.mean()]
                    val_loss = old_div(val_loss,numrep)
                    valDist = [old_div(valDist[0],numrep)]
                    self.updateBaseRegLoss(step,train_loss,val_loss,trainDist,valDist)
                if step % self.conf.save_step == 0:
                    self.saveBaseReg(sess,step)
            print("Optimization Finished!")
            self.saveBaseReg(sess,step)

