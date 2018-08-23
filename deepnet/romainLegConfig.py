from __future__ import division

# coding: utf-8

# In[ ]:

from builtins import range
from builtins import object
from past.utils import old_div
import os
import re
import localSetup
import numpy as np

class myconfig(object): 

    # ----- Names

    expname = 'romainLeg'
    baseName = 'Base'
    fineName = 'Fine' #_resize'
    mrfName = 'MRF' #_identity'
    acName = 'AC'
    regName = 'Reg'
    evalName = 'eval'
    eval2Name = 'eval2'
    genName = 'gen'
    shapeName = 'shape'

    # ----- Network parameters

    scale = 2
    rescale = 1  # how much to downsize the base image.
    eval_scale = 1
    numscale = 3
    pool_size = 3
    pool_stride = 2
    pool_scale = (pool_stride)**2
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
    n_classes = 18 # 
    dropout = 0.5 # Dropout, probability to keep units
    nfilt = 128
    nfcfilt = 256
    doBatchNorm = True
    useMRF = True
    useHoldout = False
    device = None
    reg_lambda = 0.5

    # ----- Fine Network parameters

    fine_flt_sz = 5
    fine_nfilt = 48
    fine_sz = 48

    # ----- MRF Network Parameters

    maxDPts = 400
    mrf_psz = old_div((old_div(maxDPts,rescale)),pool_scale)
    #Above should not be determined automatically
    # baseIter4MRFTrain = 4000 # without batch_norm
    baseIter4MRFTrain = 5000 # without batch_norm
    baseIter4ACTrain = 5000 # without batch_norm
    
    # ------ Pose Generation Network Params
    gen_minlen = 8

    # ------ Pose Eval 2
    poseEval2_psz = 128

    # ----- Learning parameters

    batch_size = 8
    base_learning_rate = 0.0003 #0.0001 --without batch norm
    mrf_learning_rate = 0.00001
    ac_learning_rate = 0.0003
    fine_learning_rate = 0.0003
    eval_learning_rate = 0.00001
    eval2_learning_rate = 0.01
    shape_learning_rate = 0.001
    eval_num_neg = 0
    N2move4neg = 3
    eval_minlen = 30
    
    mult_fac = old_div(16,batch_size)
    base_training_iters = 10000*mult_fac
    # with rescale = 1 performance keeps improving even at around 3000 iters.. because batch size has been halved.. duh..
    # -- March 31, 2016 Mayank
    
    # with batch normalization quite good performance is achieved within 2000 iters
    # -- March 30, 2016 Mayank
    # when run iwth batch size of 32, best validation loss is achieved at 8000 iters 
    # for FlyHeadStephenCuratedData.mat -- Feb 11, 2016 Mayank
    basereg_training_iters = 5000*mult_fac
    fine_training_iters = 3000*mult_fac
    mrf_training_iters = 3000*mult_fac
    ac_training_iters = 5000*mult_fac
    eval_training_iters = 2000*mult_fac
    eval2_training_iters = 5000*mult_fac
    shape_training_iters = 5000*mult_fac
    gen_training_iters = 4000*mult_fac
    gamma = 0.1
    step_size = 60000
    eval2_step_size = 40000
    display_step = 30
    numTest = 100
    
    # range for contrast, brightness and rotation adjustment
    horzFlip = False
    vertFlip = False
    brange = [0,0] #[-0.2,0.2] 
    crange = [0.9,1.1] #[0.7,1.3]
    rrange = 45
    imax = 255.
    adjustContrast = True
    clahegridsize = 20
    normalize_mean_img = True

    # fine_batch_size = 8

    # Shape Parameters
    shape_n_orts = 8 # number of orientation bins for shape output
    shape_n_rad = 1 # number of orientation bins for shape output
    shape_selpt1 = [0]
    shape_selpt2 = [list(range(18))]
    # for ndx in range(n_classes):
    #     shape_selpt2.append(range(n_classes))
    # shape_selpt2 = [[6,12]]
    shape_psz = 128

    # ----- Data parameters

    split = True
    view = 2
    l1_cropsz = 0
    imsz = (624,672)
    cropLoc = {(624,672):[0,0],(762,768):[85,0],(628,672):[0,0],(648,768):[0,0]} # for front view crop the central part of the image
    selpts = np.arange(0,18)
    imgDim = 1

    cachedir = os.path.join(localSetup.bdir,'cache','romainLegBottom')
    labelfile = os.path.join(localSetup.bdir,'RomainLeg','RomainCombined_fixed_fixedbabloo_20170410.lbl')
 
    trainfilename = 'train_TF'
    fulltrainfilename = 'fullTrain_TF'
    valfilename = 'val_TF'
    holdouttrain = 'holdouttrain_lmdb'
    holdouttest = 'holdouttest_lmdb'
    valdatafilename = 'valdata'
    valratio = 0.38
    holdoutratio = 0.8


    # ----- Save parameters
    save_step = 500
    maxckpt = 20
    baseoutname = expname + baseName
    fineoutname = expname + fineName
    mrfoutname = expname + mrfName
    evaloutname = expname + evalName
    eval2outname = expname + eval2Name
    genoutname = expname + genName
    baseregoutname = expname + regName
    baseckptname = baseoutname + 'ckpt'
    baseregckptname = baseregoutname + 'ckpt'
    fineckptname = fineoutname + 'ckpt'
    mrfckptname = mrfoutname + 'ckpt'
    evalckptname = evaloutname + 'ckpt'
    eval2ckptname = eval2outname + 'ckpt'
    genckptname = genoutname + 'ckpt'
    basedataname = baseoutname + 'traindata'
    baseregdataname = baseregoutname + 'traindata'
    finedataname = fineoutname + 'traindata'
    mrfdataname = mrfoutname + 'traindata'
    evaldataname = evaloutname + 'traindata'
    eval2dataname = eval2outname + 'traindata'
    gendataname = genoutname + 'traindata'
    shapeoutname = expname + shapeName
    shapeckptname = shapeoutname + 'ckpt'
    shapedataname = shapeoutname + 'traindata'


    # ----- project specific functions

    def getexpname(self,dirname):
        dirname = os.path.normpath(dirname)
        dir_parts = dirname.split(os.sep)
        expname = dir_parts[-1]
        return expname

    def getexplist(self,L):
        return L['movieFilesAll'][self.view,:]

bottomconf = myconfig()
bottomconf.brange = [0,0] 
bottomconf.crange = [0.9,1.1]
bottomconf.rrange = 45
# bottomconf.pool_size = 5
# bottomconf.pool_stride = 3
# bottomconf.pool_scale = (bottomconf.pool_stride) ** 2

# bottomconf.scale = 3
# bottomconf.psz = 576


side1conf = myconfig()
side1conf.cropLoc = {(592,288):[0,0],(672,256):[0,0],(672,320):[0,0]}
side1conf.view = 0  
side1conf.imsz = (592,256) 
# side1conf.labelfile = os.path.join(localSetup.bdir,'RomainLeg','Jun22Sep16Sep15Sep13Aug26Sep07Sep05_fixed.lbl')
side1conf.selpts = np.arange(0,18)
side1conf.cachedir = os.path.join(localSetup.bdir,'cache','romainLegSide1')
side1conf.eval_scale = 2
side1conf.brange = [-0.2,0.2] 
side1conf.crange = [0.7,1.3]
side1conf.rrange = 45



side2conf = myconfig()
side2conf.cropLoc = {(640,288):[0,0],(672,256):[0,0],(654,288):[0,0]}    
side2conf.view = 1
side2conf.imsz = (640,256) 
# side2conf.labelfile = os.path.join(localSetup.bdir,'RomainLeg','Jun22Sep16Sep15Sep13Aug26Sep07Sep05.lbl')
side2conf.selpts = np.arange(0,18)
side2conf.cachedir = os.path.join(localSetup.bdir,'cache','romainLegSide2')
side2conf.eval_scale = 2
side2conf.brange = [-0.2,0.2]
side2conf.crange = [0.7,1.3]
side2conf.rrange = 45

