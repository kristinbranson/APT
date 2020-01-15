from __future__ import print_function

from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, PReLU
from keras.layers import subtract as Subtract
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.callbacks import Callback
from keras.applications.vgg19 import VGG19
from scipy import stats
from keras.optimizers import Optimizer, Adam
from keras import backend as K
from keras.legacy import interfaces

import sys
import os
import re
import pickle
import math
import PoseTools
import os
import  numpy as np
import json
import tensorflow as tf
import keras.backend as K
import logging
from time import time
import cv2
from past.utils import old_div

import open_pose2 as op2
import open_pose_data as opdata
import heatmap

ISPY3 = sys.version_info >= (3, 0)

def relu(x): return Activation('relu')(x)

def prelu(x,nm):
    return PReLU(shared_axes=[1, 2],name=nm)(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def convblock(x0, nf, namebase, wd_kernel):
    '''
    Three 3x3 convs with PReLU and with results concatenated

    :param x0:
    :param nf:
    :param namebase:
    :param wd_kernel:
    :return:
    '''
    x1 = conv(x0, nf, 3, "cblock-{}-{}".format(namebase, 1), (wd_kernel, 0))
    x1 = prelu(x1, "cblock-{}-{}-prelu".format(namebase, 1))
    x2 = conv(x1, nf, 3, "cblock-{}-{}".format(namebase, 2), (wd_kernel, 0))
    x2 = prelu(x2, "cblock-{}-{}-prelu".format(namebase, 2))
    x3 = conv(x2, nf, 3, "cblock-{}-{}".format(namebase, 3), (wd_kernel, 0))
    x3 = prelu(x3, "cblock-{}-{}-prelu".format(namebase, 3))
    x = Concatenate(name="cblock-{}".format(namebase))([x1, x2, x3])
    return x

def stageCNN(x, nfout, stagety, stageidx, wd_kernel,
             nfconvblock=128, nconvblocks=5, nf1by1=128):
    # stagety: 'map' or 'paf'

    for iCB in range(nconvblocks):
        namebase = "{}-stg{}-cb{}".format(stagety, stageidx, iCB)
        x = convblock(x, nfconvblock, namebase, wd_kernel)
    x = conv(x, nf1by1, 1, "{}-stg{}-1by1-1".format(stagety, stageidx), (wd_kernel, 0))
    x = prelu(x, "{}-stg{}-1by1-1-prelu".format(stagety, stageidx))
    x = conv(x, nfout, 1, "{}-stg{}-1by1-2".format(stagety, stageidx), (wd_kernel, 0))
    x = prelu(x, "{}-stg{}-1by1-2-prelu".format(stagety, stageidx))
    return x

def stageCNNwithDeconv(x, nfout, stagety, stageidx, wd_kernel,
                       nfconvblock=128, nconvblocks=5, ndeconvs=2, nf1by1=128):
    '''
    Like stageCNN, but with ndeconvs Deconvolutions to increase imsz by 2**ndeconvs
    :param x:
    :param nfout:
    :param stagety:
    :param stageidx:
    :param wd_kernel:
    :param nfconvblock:
    :param nconvblocks:
    :param ndeconvs:
    :param nfdeconv:
    :param nf1by1:
    :return:
    '''

    for iCB in range(nconvblocks):
        namebase = "{}-stg{}-cb{}".format(stagety, stageidx, iCB)
        x = convblock(x, nfconvblock, namebase, wd_kernel)

    nfilt = x.shape.as_list()[-1]
    logging.info("Adding {} deconvs with nfilt={}".format(ndeconvs, nfilt))

    DCFILTSZ = 4
    for iDC in range(ndeconvs):
        dcname = "{}-stg{}-dc{}".format(stagety, stageidx, iDC)
        x = op2.deconv_2x_upsampleinit(x, nfilt, DCFILTSZ, dcname, None, 0)
        x = prelu(x, "{}-prelu".format(dcname))

    x = conv(x, nf1by1, 1, "{}-stg{}-1by1-1".format(stagety, stageidx), (wd_kernel, 0))
    x = prelu(x, "{}-stg{}-postDC-1by1-1-prelu".format(stagety, stageidx))
    x = conv(x, nfout, 1, "{}-stg{}-1by1-2".format(stagety, stageidx), (wd_kernel, 0))
    x = prelu(x, "{}-stg{}-postDC-1by1-2-prelu".format(stagety, stageidx))
    return x

'''
def stageTdeconv_block(x, num_p, stage, branch, weight_decay, weight_decay_dc, weight_decay_mode):
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    #x = deconv_2x(x, 128, 4, "Mdeconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = deconv_2x_upsampleinit(x, 128, 4, "Mdeconv3_stage%d_L%d" % (stage, branch),
                               (weight_decay_dc, 0), weight_decay_mode)
    x = relu(x)
    #x = deconv_2x(x, 128, 4, "Mdeconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = deconv_2x_upsampleinit(x, 128, 4, "Mdeconv4_stage%d_L%d" % (stage, branch),
                               (weight_decay_dc, 0), weight_decay_mode)
    x = relu(x)
    #x = deconv_2x(x, 128, 4, "Mdeconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = deconv_2x_upsampleinit(x, 128, 4, "Mdeconv5_stage%d_L%d" % (stage, branch),
                               (weight_decay_dc, 0), weight_decay_mode)
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    return x
'''

def apply_mask(x, mask, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    w = Multiply(name=w_name)([x, mask])  # vec_weight
    return w

def get_training_model(imszuse, wd_kernel, nPAFstg=5, nMAPstg=1, nlimbsT2=38, npts=19, doDC=True, nDC=2):
    '''

    :param imszuse: (imnr, imnc) raw image size, possibly adjusted to be 0 mod 8
    :param wd_kernel: weight decay for l2 reg (applied only to weights not biases)
    :param nlimbsT2:
    :param npts:
    :return: Model.
        Inputs: [img]
        Outputs: [paf_1, ... paf_nPAFstg, map_1, ... map_nMAPstg]
    '''

    imnruse, imncuse = imszuse
    assert imnruse % 8 == 0, "Image size must be divisible by 8"
    assert imncuse % 8 == 0, "Image size must be divisible by 8"

    paf_input_shape_hires = imszuse + (nlimbsT2,)
    map_input_shape_hires = imszuse + (npts,)

    imszvgg = (imnruse/8, imncuse/8)  # imsz post VGG ftrs
    paf_input_shape = imszvgg + (nlimbsT2,)
    map_input_shape = imszvgg + (npts,)

    inputs = []

    # This is hardcoded to dim=3 due to VGG pretrained weights
    img_input = Input(shape=imszuse + (3,), name='input_img')

    # paf_weight_input = Input(shape=paf_input_shape,
    #                          name='input_paf_mask')
    # map_weight_input = Input(shape=map_input_shape,
    #                          name='input_part_mask')
    # paf_weight_input_hires = Input(shape=paf_input_shape_hires,
    #                                name='input_paf_mask_hires')
    # map_weight_input_hires = Input(shape=map_input_shape_hires,
    #                                name='input_part_mask_hires')
    inputs.append(img_input)
    # inputs.append(paf_weight_input)
    # inputs.append(map_weight_input)
    # inputs.append(paf_weight_input_hires)
    # inputs.append(map_weight_input_hires)

    img_normalized = Lambda(lambda z: z / 256 - 0.5)(img_input)  # [-0.5, 0.5] Isn't this really [-0.5, 0.496]

    # VGG
    vggF = op2.vgg_block(img_normalized, wd_kernel)
    # sz should be (bsize, imszvgg[0], imszvgg[1], nchans)
    print(vggF.shape.as_list()[1:])
    assert vggF.shape.as_list()[1:] == list(imszvgg + (128,))

    # PAF 1..nPAFstg
    xpaflist = []
    xstagein = vggF
    for iPAFstg in range(nPAFstg):
        xstageout = stageCNN(xstagein, nlimbsT2, 'paf', iPAFstg, wd_kernel)
        xpaflist.append(xstageout)
        xstagein = Concatenate(name="paf-stg{}".format(iPAFstg))([vggF, xstageout])

    # MAP
    xmaplist = []
    for iMAPstg in range(nMAPstg):
        xstageout = stageCNN(xstagein, npts, 'map', iMAPstg, wd_kernel)
        xmaplist.append(xstageout)
        xstagein = Concatenate(name="map-stg{}".format(iMAPstg))([vggF, xpaflist[-1], xstageout])

    xmaplistDC = []
    if doDC:
        # xstagein is ready/good from MAP loop
        xstageout = stageCNNwithDeconv(xstagein, npts, 'map', nMAPstg, wd_kernel, ndeconvs=nDC)
        xmaplistDC.append(xstageout)

    assert len(xpaflist) == nPAFstg
    assert len(xmaplist) == nMAPstg
    outputs = xpaflist + xmaplist + xmaplistDC

    # w1 = apply_mask(stage1_branch1_out, paf_weight_input, 1, 1)
    # w2 = apply_mask(stage1_branch2_out, map_weight_input, 1, 2)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def configure_losses(model, bsize, dc_on=True, dcNum=None, dc_blur_rad_ratio=None, dc_wtfac=None):
    '''
    
    :param model: 
    :param bsize: 
    :param dc_on: True if deconv/hires is on
    :param dcNum: number of 2x deconvs applied 
    :param dc_blur_rad_ratio: The ratio blur_rad_hires/blur_rad_lores
    :param dc_wtfac: Weighting factor for hi-res
    
    :return: losses, loss_weights. both dicts whose keys are .names of model.outputs
    '''

    def eucl_loss(x, y):
        return K.sum(K.square(x - y)) / bsize / 2.  # not sure why norm by bsize nec

    losses = {}
    loss_weights = {}
    loss_weights_vec = []

    outs = model.outputs
    lyrs = model.layers
    for o in outs:
        # Not sure how to get from output Tensor to its layer. Using
        # output Tensor name doesn't work with model.compile
        olyrname = [l.name for l in lyrs if l.output == o]
        assert len(olyrname) == 1, "Found multiple layers for output."
        key = olyrname[0]
        losses[key] = eucl_loss

        if "postDC" in key:
            assert dc_on, "Found post-deconv layer"
            # left alone, L2 loss will be ~dc_blur_rad_ratio**2 larger for hi-res wrt lo-res
            loss_weights[key] = float(dc_wtfac) / float(dc_blur_rad_ratio)**2
        else:
            loss_weights[key] = 1.0

        logging.info('Configured loss for output name {}, loss_weight={}'.format(key, loss_weights[key]))
        loss_weights_vec.append(loss_weights[key])

    return losses, loss_weights, loss_weights_vec

def get_testing_model(imszuse, nPAFstg=5, nMAPstg=1, nlimbsT2=38, npts=19, doDC=True, nDC=2, fullpred=False):
    '''
    See get_training_model
    :param imszuse:
    :param nPAFstg:
    :param nMAPstg:
    :param nlimbsT2:
    :param npts:
    :param fullpred:
    :return:
    '''

    imnruse, imncuse = imszuse
    assert imnruse % 8 == 0, "Image size must be divisible by 8"
    assert imncuse % 8 == 0, "Image size must be divisible by 8"
    imszvgg = (imnruse/8, imncuse/8)  # imsz post VGG ftrs

    img_input = Input(shape=imszuse + (3,), name='input_img')

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    vggF = op2.vgg_block(img_normalized, None)
    # sz should be (bsize, imszvgg[0], imszvgg[1], nchans)
    print(vggF.shape.as_list()[1:])
    assert vggF.shape.as_list()[1:] == list(imszvgg + (128,))

    # PAF 1..nPAFstg
    xpaflist = []
    xstagein = vggF
    for iPAFstg in range(nPAFstg):
        # Using None for wd_kernel is nonsensical but shouldn't hurt in test mode
        xstageout = stageCNN(xstagein, nlimbsT2, 'paf', iPAFstg, None)
        xpaflist.append(xstageout)
        xstagein = Concatenate(name="paf-stg{}".format(iPAFstg))([vggF, xstageout])

    # MAP
    xmaplist = []
    for iMAPstg in range(nMAPstg):
        # Using None for wd_kernel is nonsensical but shouldn't hurt in test mode
        xstageout = stageCNN(xstagein, npts, 'map', iMAPstg, None)
        xmaplist.append(xstageout)
        xstagein = Concatenate(name="map-stg{}".format(iMAPstg))([vggF, xpaflist[-1], xstageout])

    xmaplistDC = []
    if doDC:
        # xstagein is ready/good from MAP loop
        xstageout = stageCNNwithDeconv(xstagein, npts, 'map', nMAPstg, None, ndeconvs=nDC)
        xmaplistDC.append(xstageout)

    assert len(xpaflist) == nPAFstg
    assert len(xmaplist) == nMAPstg

    if fullpred:
        outputs = xpaflist + xmaplist + xmaplistDC
    elif doDC:
        outputs = [xpaflist[-1], xmaplistDC[-1], ]
    else:
        outputs = [xpaflist[-1], xmaplist[-1], ]

    model = Model(inputs=[img_input], outputs=outputs)

    return model




# ---------------------
# -- Training ---------
#----------------------

def set_openpose_defaults(conf):
    conf.label_blur_rad = 5
    conf.rrange = 5
    conf.display_step = 50 # this is same as batches per epoch
    conf.dl_steps = 600000
    conf.batch_size = 10
    conf.n_steps = 4.41
    conf.gamma = 0.333


def massage_conf(conf):
    assert conf.dl_steps >= conf.display_step, \
        "Number of dl steps must be greater than or equal to the display step"
    div,mod = divmod(conf.dl_steps,conf.display_step)
    if mod != 0:
        conf.dl_steps = (div+1) * conf.display_step
        logging.info("Openpose requires the number of dl steps to be an even multiple of the display step. Increasing dl steps to {}".format(conf.dl_steps))

    assert conf.save_step >= conf.display_step, \
        "dl save step must be greater than or equal to the display step"
    div,mod = divmod(conf.save_step,conf.display_step)
    if mod != 0:
        conf.save_step = (div+1) * conf.display_step
        logging.info("Openpose requires the save step to be an even multiple of the display step. Increasing save step to {}".format(conf.save_step))


def imszcheckcrop(sz, dimname):
    szm8 = sz % 8
    szuse = sz - szm8
    if szm8 != 0:
        warnstr = 'Image {} dimension ({}) is not a multiple of 8. Image will be cropped slightly.'.format(dimname, sz)
        logging.warning(warnstr)
    return szuse

# AL losses, resolutions, blurs
# Calling "loss0" the loss vs an all-zero array of the right size.
# - For hmap, changing resolutions does not change loss0
# - For hmap, increasing blur_rad increases loss0 by ~blur_rad^2 as expected.
#    (blur_rad 1->3 ~ loss0 9.5->85)
# - For paf, changing resolutions does change loss0 very roughly linearly as the
#   limb length is linear in img sz. (increase res 5x => loss0 5x)
# - For paf, changing the blur_rad also changes loss0 roughly linearly as the
#   limb width ~ linear in blur_rad.

def dot(K, L):
   assert len(K) == len(L), 'lens do not match: {} vs {}'.format(len(K), len(L))
   return sum(i[0] * i[1] for i in zip(K, L))


def training(conf, name='deepnet'):

    base_lr = conf.op_base_lr
    batch_size = conf.batch_size  # Gines 10
    gamma = conf.gamma  # Gines 1/2
    stepsize = int(conf.decay_steps)  # after each stepsize iterations update learning rate: lr=lr*gamma
      # Gines much larger: 200k, 300k, then every 60k
    iterations_per_epoch = conf.display_step
    max_iter = conf.dl_steps/iterations_per_epoch
    last_epoch = 0

    (imnr, imnc) = conf.imsz
    imnr_use = imszcheckcrop(imnr, 'row')
    imnc_use = imszcheckcrop(imnc, 'column')
    imszuse = (imnr_use, imnc_use)
    conf.imszuse = imszuse

    assert conf.dl_steps % iterations_per_epoch == 0, 'For open-pose dl steps must be a multiple of display steps'
    assert conf.save_step % iterations_per_epoch == 0, 'For open-pose save steps must be a multiple of display steps'

    # need this to set default
    save_time = conf.get('save_time', None)

    train_data_file = os.path.join(conf.cachedir, 'traindata')
    with open(train_data_file, 'wb') as td_file:
        pickle.dump(conf, td_file, protocol=2)
    logging.info('Saved config to {}'.format(train_data_file))

    #model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{epoch:d}')
    model = get_training_model(imszuse,
                               conf.op_weight_decay_kernel,
                               nPAFstg=conf.op_paf_nstage,
                               nMAPstg=conf.op_map_nstage,
                               nlimbsT2=len(conf.op_affinity_graph) * 2,
                               npts=conf.n_classes,
                               doDC=conf.op_hires,
                               nDC=conf.op_hires_ndeconv)

    logging.info("Loading vgg19 weights...")
    from_vgg = op2.from_vgg
    vgg_model = VGG19(include_top=False, weights='imagenet')
    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            logging.info("Loaded VGG19 layer: {}->{}".format(layer.name, vgg_layer_name))

    # prepare generators
    train_di = opdata.DataIteratorTF(conf, 'train', True, True)
    train_di2 = opdata.DataIteratorTF(conf, 'train', True, True)
    val_di = opdata.DataIteratorTF(conf, 'train', False, False)

    assert conf.op_label_scale == 8
    logging.info("Your label_blur_rad is {}".format(conf.label_blur_rad))
    losses, loss_weights, loss_weights_vec = \
        configure_losses(model, batch_size,
                         dc_on=conf.op_hires,
                         dc_blur_rad_ratio=conf.op_map_hires_blur_rad / conf.op_map_lores_blur_rad,
                         dc_wtfac=2.5)

    def lr_decay(epoch):  # epoch is 0-based
        initial_lrate = base_lr
        steps = (epoch+1) * iterations_per_epoch
        lrate = initial_lrate * math.pow(gamma, math.floor(steps / stepsize))
        return lrate

    # Callback to do writing pring stuff.
    class OutputObserver(Callback):
        def __init__(self, conf, dis):
            self.train_di, self.val_di = dis
            self.train_info = {}
            self.train_info['step'] = []
            self.train_info['train_dist'] = []
            self.train_info['train_loss'] = []  # scalar loss (dotted with weightvec)
            self.train_info['train_loss_K'] = []  # scalar loss as reported by K
            self.train_info['train_loss_full'] = []  # full loss, layer by layer
            self.train_info['val_dist'] = []
            self.train_info['val_loss'] = []  # etc
            self.train_info['val_loss_K'] = []
            self.train_info['val_loss_full'] = []
            self.train_info['lr'] = []
            self.config = conf
            self.force = False
            self.save_start = time()

        def on_epoch_end(self, epoch, logs={}):
            step = (epoch+1) * iterations_per_epoch
            val_x, val_y = self.val_di.next()
            val_out = self.model.predict(val_x, batch_size=batch_size)
            val_loss_full = self.model.evaluate(val_x, val_y, batch_size=batch_size, verbose=0)
            val_loss_K = val_loss_full[0]  # want Py 3 unpack
            val_loss_full = val_loss_full[1:]
            val_loss = dot(val_loss_full, loss_weights_vec)
            #val_loss = np.nan
            train_x, train_y = self.train_di.next()
            train_out = self.model.predict(train_x, batch_size=batch_size)
            train_loss_full = self.model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
            train_loss_K = train_loss_full[0]  # want Py 3 unpack
            train_loss_full = train_loss_full[1:]
            train_loss = dot(train_loss_full, loss_weights_vec)
            #train_loss = np.nan
            lr = K.eval(self.model.optimizer.lr)

            # dist only for last MAP layer (will be hi-res if deconv is on)
            predhmval = val_out[-1]
            predhmval = clip_heatmap_with_warn(predhmval)
            # (bsize, npts, 2), (x,y), 0-based
            predlocsval = heatmap.get_weighted_centroids(predhmval,
                                                         floor=self.config.op_hmpp_floor,
                                                         nclustermax=self.config.op_hmpp_nclustermax)
            gtlocs = heatmap.get_weighted_centroids(val_y[-1],
                                                    floor=self.config.op_hmpp_floor,
                                                    nclustermax=self.config.op_hmpp_nclustermax)
            tt1 = predlocsval - gtlocs
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))  # [bsize x ncls]
            val_dist = np.nanmean(tt1)  # this dist is in op_scale-downsampled space
                                        # *self.config.op_label_scale

            # NOTE train_dist uses argmax
            tt1 = PoseTools.get_pred_locs(train_out[-1]) - \
                  PoseTools.get_pred_locs(train_y[-1])
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            train_dist = np.nanmean(tt1) # *self.config.op_label_scale
            self.train_info['val_dist'].append(val_dist)
            self.train_info['val_loss'].append(val_loss)
            self.train_info['val_loss_K'].append(val_loss_K)
            self.train_info['val_loss_full'].append(val_loss_full)
            self.train_info['train_dist'].append(train_dist)
            self.train_info['train_loss'].append(train_loss)
            self.train_info['train_loss_K'].append(train_loss_K)
            self.train_info['train_loss_full'].append(train_loss_full)
            self.train_info['step'].append(int(step))
            self.train_info['lr'].append(lr)

            p_str = ''
            for k in self.train_info.keys():
                lastval = self.train_info[k][-1]
                if k == 'lr':
                    p_str += '{:s}:{:.4g} '.format(k, lastval)
                elif isinstance(lastval, list):
                    p_str += '{:s}:<list {} els>'.format(k, len(lastval))
                else:
                    p_str += '{:s}:{:.2f} '.format(k, lastval)
            logging.info(p_str)

            train_data_file = os.path.join(self.config.cachedir, 'traindata')

            json_data = {}
            for x in self.train_info.keys():
                json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
            with open(train_data_file + '.json', 'w') as json_file:
                json.dump(json_data, json_file)
            with open(train_data_file, 'wb') as td:
                pickle.dump([self.train_info, conf], td, protocol=2)

            if conf.save_time is None:
                if step % conf.save_step == 0:
                    model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(int(step)))))
            else:
                if time() - self.save_start > conf.save_time*60:
                    self.save_start = time()
                    model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(int(step)))))


    # configure callbacks
    lrate = LearningRateScheduler(lr_decay)
    # checkpoint = ModelCheckpoint(val_di
    #     model_file, monitor='loss', verbose=0, save_best_only=False,
    #     save_weights_only=True, mode='min', period=conf.save_step)
    obs = OutputObserver(conf, [train_di2, val_di])
    callbacks_list = [lrate, obs]  #checkpoint,

    # optimizer = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)#, clipnorm=1.)
    # Mayank 20190423 - Adding clipnorm so that the loss doesn't go to zero.
    optimizer = Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

    logging.info("Your model.metrics_names are {}".format(model.metrics_names))

    # save initial model
    model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(0))))

    model.fit_generator(train_di,
                        steps_per_epoch=iterations_per_epoch,
                        epochs=max_iter-1,
                        callbacks=callbacks_list,
                        verbose=0,
                        initial_epoch=last_epoch
                        )
                        # validation_data=val_di,
                        # validation_steps=val_samples // batch_size,
#                        use_multiprocessing=True,
#                        workers=4,

    # force saving in case the max iter doesn't match the save step.
    model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(int(max_iter*iterations_per_epoch)))))
    obs.on_epoch_end(max_iter-1)

def clip_heatmap_with_warn(predhm):
    '''

    :param predhm:
    :return: clipped predhm; could be same array as predhm if no change
    '''

    if np.any(predhm < 0.0):
        PTILES = [1, 5, 10, 50, 99]
        ptls = np.percentile(predhm, PTILES)
        warnstr = 'Prediction heatmap has negative els! PTILES {}: {}'.format(PTILES, ptls)
        logging.warning(warnstr)

        predhm_clip = predhm.copy()
        predhm_clip[predhm_clip < 0.0] = 0.0
    else:
        predhm_clip = predhm

    return predhm_clip

def get_pred_fn(conf, model_file=None, name='deepnet', rawpred=False):
    (imnr, imnc) = conf.imsz
    imnr_use = imszcheckcrop(imnr, 'row')
    imnc_use = imszcheckcrop(imnc, 'column')
    imszuse = (imnr_use, imnc_use)
    conf.imszuse = imszuse

    model = get_testing_model(imszuse,
                              nPAFstg=conf.op_paf_nstage,
                              nMAPstg=conf.op_map_nstage,
                              nlimbsT2=len(conf.op_affinity_graph) * 2,
                              npts=conf.n_classes,
                              doDC=conf.op_hires,
                              nDC=conf.op_hires_ndeconv,
                              fullpred=rawpred)
    if model_file is None:
        latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    else:
        latest_model_file = model_file
    logging.info("Loading the weights from {}.. ".format(latest_model_file))
    model.load_weights(latest_model_file)
    # thre1 = conf.get('op_param_hmap_thres',0.1)
    # thre2 = conf.get('op_param_paf_thres',0.05)

    def pred_fn(all_f):

        assert conf.op_rescale == 1  # for now
        assert all_f.shape[0] == conf.batch_size

        locs_sz = (conf.batch_size, conf.n_classes, 2)

        # mirror open_pose_data/DataIteratorTF

        ims, _ = PoseTools.preprocess_ims(
            all_f,
            in_locs=np.zeros(locs_sz),
            conf=conf,
            distort=False,
            scale=conf.op_rescale)

        ims = ims[:, 0:imnr_use, 0:imnc_use, :]

        assert conf.img_dim == ims.shape[-1]
        if conf.img_dim == 1:
            ims = np.tile(ims, 3)

        model_preds = model.predict(ims)
        # all_infered = []
        # for ex in range(xs.shape[0]):
        #     infered = do_inference(model_preds[-1][ex,...],model_preds[-2][ex,...],conf, thre1, thre2)
        #     all_infered.append(infered)
        predhm = model_preds[-1]  # this is always the last/final MAP hmap
        predhm_clip = clip_heatmap_with_warn(predhm)

        # (bsize, npts, 2), (x,y), 0-based
        predlocs_argmax = PoseTools.get_pred_locs(predhm)
        predlocs_wgtcnt = heatmap.get_weighted_centroids(predhm_clip,
                                                         floor=conf.op_hmpp_floor,
                                                         nclustermax=conf.op_hmpp_nclustermax)
        assert predlocs_argmax.shape == locs_sz
        assert predlocs_wgtcnt.shape == locs_sz
        print("HMAP POSTPROC, floor={}, nclustermax={}".format(conf.op_hmpp_floor, conf.op_hmpp_nclustermax))

        unscalefac = conf.op_label_scale
        if conf.op_hires:
            unscalefac = unscalefac / 2**conf.op_hires_ndeconv
        assert predhm_clip.shape[1] == imnr_use / unscalefac
        assert predhm_clip.shape[2] == imnc_use / unscalefac
        predlocs_argmax_hires = opdata.unscale_points(predlocs_argmax, unscalefac)
        predlocs_wgtcnt_hires = opdata.unscale_points(predlocs_wgtcnt, unscalefac)

        assert conf.op_rescale == 1  # we are not rescaling by this

        # base_locs = np.array(all_infered)*conf.op_rescale
        # nanidx = np.isnan(base_locs)
        # base_locs[nanidx] = raw_locs[nanidx]

        ret_dict = {}
        ret_dict['locs'] = predlocs_wgtcnt_hires
        ret_dict['locs_mdn'] = predlocs_argmax_hires  # XXX hack for now
        ret_dict['locs_unet'] = predlocs_argmax_hires  # XXX hack for now
        ret_dict['conf'] = np.max(predhm, axis=(1, 2))
        ret_dict['conf_unet'] = np.max(predhm, axis=(1, 2)) # XXX hack
        return ret_dict

    # def pred_fn_rawmaps(all_f):
    #     all_f = all_f[:, 0:imnr_use, 0:imnc_use, :]
    #
    #     if all_f.shape[3] == 1:
    #         all_f = np.tile(all_f,[1,1,1,3])
    #     # tiling beforehand a little weird as preprocess_ims->normalizexyxy branches on
    #     # if img is color
    #     xs, _ = PoseTools.preprocess_ims(
    #         all_f, in_locs=np.zeros([conf.batch_size, conf.n_classes, 2]), conf=conf,
    #         distort=False, scale=conf.op_rescale)
    #     model_preds = model.predict(xs)
    #     # all_infered = []
    #     # for ex in range(xs.shape[0]):
    #     #     infered = do_inference(model_preds[-1][ex,...],model_preds[-2][ex,...],conf, thre1, thre2)
    #     #     all_infered.append(infered)
    #     return model_preds


    def close_fn():
        K.clear_session()
        # gc.collect()
        # del model

    if rawpred:
        assert False, "unsupported"
        return pred_fn_rawmaps, close_fn, latest_model_file
    else:
        return pred_fn, close_fn, latest_model_file


def do_inference(hmap, paf, conf, thre1, thre2):
    all_peaks = []
    peak_counter = 0
    limb_seq = conf.op_affinity_graph
    hmap = cv2.resize(hmap,(0,0),fx=conf.op_label_scale, fy=conf.op_label_scale,interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf,(0,0),fx=conf.op_label_scale, fy=conf.op_label_scale,interpolation=cv2.INTER_CUBIC)

    for part in range(hmap.shape[-1]):
        map_ori = hmap[:, :, part]
        map = map_ori
        # map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse

        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        # if len(peaks_with_score)>2:
        #    peaks_with_score = sorted(peaks_with_score,key=lambda x:x[2],reverse=True)[:2]
        #    peaks = peaks_with_score #  taking the first two peaks
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 8

    for k in range(len(limb_seq)):
        x_paf = paf[:,:,k*2]
        y_paf = paf[:,:,k*2+1]
        # score_mid = paf[:, :, [x for x in limb_seq[k]]]
        candA = all_peaks[limb_seq[k][0]]
        candB = all_peaks[limb_seq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limb_seq[k]
        if  nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    if norm > max(conf.imsz):
                        continue
                    # if limbSeq[k][0]==0 and limbSeq[k][1]==1 and norm < 150:
                    #   continue

                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [x_paf[int(round(startend[I][1])), int(round(startend[I][0]))] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [y_paf[int(round(startend[I][1])), int(round(startend[I][0]))] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) # + min( 0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len( score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, conf.n_classes + 2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(limb_seq)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limb_seq[k])

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif subset[j][indexA] != partAs[i]:
                        subset[j][indexA] = partAs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partAs[i].astype(int), 2] + connection_all[k][i][2]

                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 2:
                    row = -1 * np.ones(conf.n_classes+2)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < conf.n_classes or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # canvas = cv2.imread(input_image)  # B,G,R order
    detections = []
    parts = {}
    for i in range(conf.n_classes):  # 17
        parts[i] = []
        for j in range(len(all_peaks[i])):
            a = int(all_peaks[i][j][0])
            b = int(all_peaks[i][j][1])
            c = all_peaks[i][j][2]
            detections.append((a, b, c))
            parts[i].append((a, b, c))
    #        cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    # stickwidth = 10 #4
    # print()
    mappings = np.ones([conf.n_classes,2])*np.nan
    if subset.shape[0] < 1:
        return mappings

    subset = subset[np.argsort(subset[:,-2])] # sort by highest scoring one.
    for n in range(1):
        for i in range(conf.n_classes):
            index = subset[-n-1][i]
            if index < 0:
                mappings[i,:] = np.nan
            else:
                mappings[i,:] = candidate[index.astype(int), :2]
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
            # 360, 1)
            # cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            # canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    mappings -= conf.op_label_scale/4 # For some reason, cv2.imresize maps (x,y) to (8*x+4,8*y+4). sigh.
    return mappings


def model_files(conf, name):
    latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    if latest_model_file is None:
        return None
    traindata_file = PoseTools.get_train_data_file(conf, name)
    return [latest_model_file, traindata_file + '.json']

