from __future__ import division

# coding: utf-8

# In[ ]:

from builtins import object
from past.utils import old_div
import os
import re
import localSetup
import numpy as np

class myconfig(object):
    expname = 'janLeg'
    baseName = 'Base'
    fineName = 'Fine' #_resize'
    mrfName = 'MRF' #_identity'
    acName = 'AC'
    regName = 'Reg'
    evalName = 'eval'
    genName = 'gen'

    # ----- Network parameters

    scale = 2
    rescale = 1  # how much to downsize the base image.
    numscale = 3
    pool_scale = 4
    pool_size = 3
    pool_stride = 2
    # sel_sz determines the patch size used for the final decision
    # i.e., patch seen by the fc6 layer
    # ideally as large as possible but limited by
    # a) gpu memory size
    # b) overfitting due to large number of variables.
    sel_sz = 512/2/2
    psz = sel_sz/(scale**(numscale-1))/rescale/pool_scale
    dist2pos = 5
    label_blur_rad = 3 #1.5
    fine_label_blur_rad = 1.5
    n_classes = 4
    dropout = 0.5
    nfilt = 128
    nfcfilt = 512
    doBatchNorm = True
    useMRF = True
    useHoldout = False

    mrf_psz = 50
    # ----- Fine Network parameters
    fine_flt_sz = 5
    fine_nfilt = 48
    fine_sz = 48

    # ----- MRF Network Parameters
    baseIter4MRFTrain = 5000
    baseIter4ACTrain = 5000


    # ----- Learning parameters

    base_learning_rate = 0.0003
    mrf_learning_rate = 0.00001
    ac_learning_rate = 0.0003
    fine_learning_rate = 0.0003

    batch_size = 8
    mult_fac = old_div(16,batch_size)
    base_training_iters = 15000*mult_fac
    fine_training_iters = 5000*mult_fac
    mrf_training_iters = 3000*mult_fac
    ac_training_iters = 5000
    gamma = 0.1
    step_size = 100000
    display_step = 30
    numTest = 100

    # range for contrast, brightness and rotation adjustment
    horzFlip = False
    vertFlip = False
    brange = [-0.2,0.2]
    crange = [0.7,1.3]
    rrange = 30
    imax = 255.
    adjustContrast = True
    clahegridsize = 20
    normalize_mean_img = True

    # ----- Data parameters

    split = True
    view = 0
    l1_cropsz = 0
    imsz = (256,256)
    map_size = 100000*imsz[0]*imsz[1]*3
    cropLoc = {(256,256):[0,0]}
    selpts = np.arange(3,7)
    imgDim = 1

    cachedir = os.path.join(localSetup.bdir,'cachejanLeg/')
    labelfile = os.path.join(localSetup.bdir,'janLegTracking','160819_Dhyey_2_al_fixed.lbl')
    # this label file has more data and includes the correction for vertical flipping
    trainfilename = 'train_TF'
    fulltrainfilename = 'fullTrain_TF'
    valfilename = 'val_TF'
    holdouttrain = 'holdouttrain_lmdb'
    holdouttest = 'holdouttest_lmdb'
    valdatafilename = 'valdata'
    valratio = 0.3
    holdoutratio = 0.8


    # ----- Save parameters

    save_step = 500
    maxckpt = 20
    baseoutname = expname + baseName
    fineoutname = expname + fineName
    mrfoutname = expname + mrfName
    acoutname = expname + acName
    baseckptname = baseoutname + 'ckpt'
    fineckptname = fineoutname + 'ckpt'
    mrfckptname = mrfoutname + 'ckpt'
    acckptname = acoutname + 'ckpt'
    basedataname = baseoutname + 'traindata'
    finedataname = fineoutname + 'traindata'
    mrfdataname = mrfoutname + 'traindata'
    acdataname = acoutname + 'traindata'


    def getexpname(self,dirname):
        expname = os.path.basename(os.path.dirname(dirname))
        return expname

    def getexplist(self,L):
        # fname = 'vid{:d}files'.format(self.view+1)
        # return L[fname]

        return L['movieFilesAll'][self.view,:]


conf = myconfig()

