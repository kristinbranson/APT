from __future__ import division

# coding: utf-8

# In[ ]:

from builtins import object
from past.utils import old_div
import os
import re
import localSetup
import numpy as np
import copy

class config(object):
    # ----- Names

    # baseName = 'Base'
    # fineName = 'Fine'  # _resize'
    # mrfName = 'MRF'  # _identity'

    # ----- Network parameters
    def __init__(self):
        self.scale = 2
        self.rescale = 1  # how much to downsize the base image.
        self.numscale = 3
        self.pool_scale = 4
        self.pool_size = 3
        self.pool_stride = 2
        # # sel_sz determines the patch size used for the final decision
        # # i.e., patch seen by the fc6 layer
        # # ideally as large as possible but limited by
        # # a) gpu memory size
        # # b) overfitting due to large number of variables.
        # dist2pos = 5
        self.label_blur_rad = 3  # 1.5
        # fine_label_blur_rad = 1.5
        # dropout = 0.5 # Dropout, probability to keep units
        # nfilt = 128
        # nfcfilt = 512
        # doBatchNorm = True
        # useMRF = True
        # useHoldout = False

        # ----- Fine Network parameters

        # fine_flt_sz = 5
        # fine_nfilt = 48
        # fine_sz = 48

        # ----- MRF Network Parameters

        # maxDPts = 104*2
        # mrf_psz = (maxDPts/rescale)/pool_scale
        # Above should not be determined automatically
        # baseIter4MRFTrain = 4000 # without batch_norm
        # baseIter4MRFTrain = 5000  # without batch_norm
        # baseIter4ACTrain = 5000  # without batch_norm

        # ----- Learning parameters

        # base_learning_rate = 0.0003  # 0.0001 --without batch norm
        # mrf_learning_rate = 0.00001
        # fine_learning_rate = 0.0003

        self.batch_size = 8
        self.view = 0
        # fine_batch_size = 8
        # mult_fac = old_div(16, batch_size)
        # base_training_iters = 10000 * mult_fac  # 15000
        # with rescale = 1 performance keeps improving even at around 3000 iters.. because batch size has been halved.. duh..
        # -- March 31, 2016 Mayank

        # with batch normalization quite good performance is achieved within 2000 iters
        # -- March 30, 2016 Mayank
        # when run iwth batch size of 32, best validation loss is achieved at 8000 iters
        # for FlyHeadStephenCuratedData.mat -- Feb 11, 2016 Mayank
        # basereg_training_iters = 5000 * mult_fac
        # fine_training_iters = 5000 * mult_fac
        # mrf_training_iters = 3000 * mult_fac
        self.gamma = 0.1
        self.step_size = 100000 # not used anymore
        self.display_step = 50
        self.numTest = 8
        self.dl_steps = 20000 # number of training iters
        self.cos_steps = 2 #number of times the learning rate is decayed
        self.n_steps = 2
        # rate will be reduced by gamma**n_steps by the end of the training.

        # range for contrast, brightness and rotation adjustment
        self.horzFlip = False
        self.vertFlip = False
        self.brange = [-0.2, 0.2]
        self.crange = [0.7, 1.3]
        self.rrange = 30
        self.trange = 10
        self.scale_range = 0.1
        self.imax = 255.
        self.adjustContrast = False
        self.clahegridsize = 20
        self.normalize_img_mean = True
        self.normalize_batch_mean = False
        self.perturb_color = False


        # ----- Data parameters
        # l1_cropsz = 0
        self.splitType = 'frame'
        self.trainfilename = 'train_TF'
        self.fulltrainfilename = 'fullTrain_TF'
        self.valfilename = 'val_TF'
        self.valdatafilename = 'valdata'
        self.valratio = 0.3
        self.holdoutratio = 0.8
        self.max_n_animals = 1
        self.flipud = False

        # ----- UNet params
        self.unet_rescale = 1
        #self.unet_steps = 20000
        self.unet_keep_prob = 1.0 # essentially don't use it.
        self.unet_use_leaky = False #will change it to True after testing.

        # ----- MDN params
        self.mdn_min_sigma = 3. # this should just be maybe twice the cell size??
        self.mdn_max_sigma = 4.
        self.mdn_logit_eps_training = 0.001
        self.mdn_extra_layers = 1

        # ----- OPEN POSE PARAMS
        self.op_label_scale = 8

        # ------ Leap params
        self.leap_net_name = "leap_cnn"

        # ----- Deep Lab Cut
        self.dlc_train_img_dir = 'deepcut_train'
        self.dlc_train_data_file = 'deepcut_data.mat'

        # ----- Time parameters
        self.time_window_size = 1
        self.do_time = False

        # ------ RNN Parameters
        self.rnn_before = 9
        self.rnn_after = 0

        # ------------ ATTention parameters
        self.att_hist = 128
        self.att_layers = [1] # use layer this far from the middle (top?) layers.

        # ----- Save parameters

        self.save_step = 2000
        self.save_td_step = 100
        self.maxckpt = 30

    def set_exp_name(self, exp_name):
        self.expname = exp_name
        # self.baseoutname = self.expname + self.baseName
        # self.baseckptname = self.baseoutname + 'ckpt'
        # self.basedataname = self.baseoutname + 'traindata'
        # self.fineoutname = self.expname + self.fineName
        # self.fineckptname = self.fineoutname + 'ckpt'
        # self.finedataname = self.fineoutname + 'traindata'
        # self.mrfoutname = self.expname + self.mrfName
        # self.mrfckptname = self.mrfoutname + 'ckpt'
        # self.mrfdataname = self.mrfoutname + 'traindata'


    def getexpname(self, dirname):
        return os.path.basename(os.path.dirname(dirname)) + '_' + os.path.splitext(os.path.basename(dirname))[0]

    def getexplist(self, L):
        return L['movieFilesAll'][self.view,:]


# -- alice fly --

aliceConfig = config()
aliceConfig.cachedir = os.path.join(localSetup.bdir, 'cache', 'alice')
#aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_20170925_cv.lbl')
# aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_20180107.lbl') # round1
# aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_expandedbehavior_20180425_local.lbl')
aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_expandedbehavior_20180425.lbl')
def alice_exp_name(dirname):
    return os.path.basename(os.path.dirname(dirname))

aliceConfig.getexpname = alice_exp_name
aliceConfig.has_trx_file = True
aliceConfig.imsz = (180, 180)
aliceConfig.selpts = np.arange(0, 17)
aliceConfig.imgDim = 1
aliceConfig.n_classes = len(aliceConfig.selpts)
aliceConfig.splitType = 'frame'
aliceConfig.set_exp_name('aliceFly')
aliceConfig.trange = 5
aliceConfig.nfcfilt = 128
aliceConfig.sel_sz = 144
aliceConfig.num_pools = 1
aliceConfig.dilation_rate = 2
aliceConfig.pool_scale = aliceConfig.pool_stride**aliceConfig.num_pools
aliceConfig.psz = aliceConfig.sel_sz / 4 / aliceConfig.pool_scale / aliceConfig.dilation_rate
aliceConfig.valratio = 0.25
# aliceConfig.mdn_min_sigma = 70.
# aliceConfig.mdn_max_sigma = 70.
aliceConfig.adjustContrast = False
aliceConfig.clahegridsize = 10
aliceConfig.brange = [0,0]
aliceConfig.crange = [1.,1.]
aliceConfig.mdn_extra_layers = 1
aliceConfig.normalize_img_mean = False
aliceConfig.mdn_groups = [range(17)]


aliceConfig_time = copy.deepcopy(aliceConfig)
aliceConfig_time.do_time = True
aliceConfig_time.cachedir = os.path.join(localSetup.bdir, 'cache','alice_time')


aliceConfig_rnn = copy.deepcopy(aliceConfig)
aliceConfig_rnn.cachedir = os.path.join(localSetup.bdir, 'cache','alice_rnn')
aliceConfig_rnn.batch_size = 2
# aliceConfig_rnn.trainfilename_rnn = 'train_rnn_TF'
# aliceConfig_rnn.fulltrainfilename_rnn = 'fullTrain_rnn_TF'
# aliceConfig_rnn.valfilename_rnn = 'val_rnn_TF'

# -- felipe bees --

felipeConfig = config()
felipeConfig.cachedir = os.path.join(localSetup.bdir, 'cache','felipe')
felipeConfig.labelfile = os.path.join(localSetup.bdir,'data','felipe','doesnt_exist.lbl')
def felipe_exp_name(dirname):
    return dirname

def felipe_get_exp_list(L):
    return 0

felipeConfig.getexpname = felipe_exp_name
felipeConfig.getexplist = felipe_get_exp_list
felipeConfig.view = 0
felipeConfig.imsz = (300, 300)
felipeConfig.selpts = np.arange(0, 5)
felipeConfig.imgDim = 3
felipeConfig.n_classes = len(felipeConfig.selpts)
felipeConfig.splitType = 'frame'
felipeConfig.set_exp_name('felipeBees')
felipeConfig.trange = 20
felipeConfig.nfcfilt = 128
felipeConfig.sel_sz = 144
felipeConfig.num_pools = 2
felipeConfig.dilation_rate = 1
felipeConfig.pool_scale = felipeConfig.pool_stride**felipeConfig.num_pools
felipeConfig.psz = felipeConfig.sel_sz / 4 / felipeConfig.pool_scale / felipeConfig.dilation_rate


##  -- felipe multi bees

# -- felipe bees --

felipe_config_multi = config()
felipe_config_multi.cachedir = os.path.join(localSetup.bdir, 'cache', 'felipe_m')
felipe_config_multi.labelfile = os.path.join(localSetup.bdir, 'data', 'felipe_m', 'doesnt_exist.lbl')
def felipe_exp_name(dirname):
    return dirname

def felipe_get_exp_list(L):
    return 0

felipe_config_multi.getexpname = felipe_exp_name
felipe_config_multi.getexplist = felipe_get_exp_list
felipe_config_multi.view = 0
felipe_config_multi.imsz = (360, 380)
felipe_config_multi.selpts = np.array([1, 3, 4])
felipe_config_multi.imgDim = 3
felipe_config_multi.n_classes = len(felipe_config_multi.selpts)
felipe_config_multi.splitType = 'frame'
felipe_config_multi.set_exp_name('felipeBeesMulti')
felipe_config_multi.trange = 20
felipe_config_multi.nfcfilt = 128
felipe_config_multi.sel_sz = 256
felipe_config_multi.num_pools = 2
felipe_config_multi.dilation_rate = 1
felipe_config_multi.pool_scale = felipe_config_multi.pool_stride ** felipe_config_multi.num_pools
felipe_config_multi.psz = felipe_config_multi.sel_sz / 4 / felipe_config_multi.pool_scale / felipe_config_multi.dilation_rate
felipe_config_multi.max_n_animals = 17
