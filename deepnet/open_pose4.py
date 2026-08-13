import os
import sys
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
K = tf.keras.backend
# Pylance complains about all these imports b/c apparently TF uses dynamic behavior to import keras -- ALT, 2023-03-31
#   See: https://github.com/microsoft/pylance-release/issues/3387
# from tf.keras.models import Model
# from tf.keras.layers import Concatenate
# from tf.keras.layers import Activation, Input, Lambda, PReLU
# from tf.keras.layers import Conv2DTranspose
# from tf.keras.layers import Multiply
# from tf.keras.regularizers import l2
# from tf.keras.initializers import random_normal,constant
# from tf.keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
# from tf.keras.callbacks import Callback
# from tf.keras.applications.vgg19 import VGG19

import imagenet_resnet
#from scipy import stats
import re
import pickle
import math
import PoseTools
import numpy as np
import json
import logging
from time import time
import cv2

import tfdatagen
import heatmap
import util

import vgg_cpm
from vgg_cpm import conv
import multiprocessing

#sys.stderr = stderr

'''
Adapted from:

* OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
  Zhe Cao, Gines Hidalgo, Tomas Simon, Shin-En Wei, Yaser Sheikh 
  https://arxiv.org/pdf/1812.08008.pdf

* OpenPose: Whole-Body Pose Estimation
  Gines Hidalgo
  CMU-RI-TR-19-015
  https://www.ri.cmu.edu/publications/openpose-whole-body-pose-estimation/
  
'''



ISPY3 = sys.version_info >= (3, 0)

def prelu(x,nm):
    return tf.keras.layers.PReLU(shared_axes=[1, 2],name=nm)(x)

def upsample_filt(alg='nn', dtype=None):
    if alg == 'nn':
        x = np.array([[0., 0., 0., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 0., 0., 0.]], dtype=dtype.as_numpy_dtype)
    elif alg == 'bl':
        x = np.array(
            [[0.0625, 0.1875, 0.1875, 0.0625],
             [0.1875, 0.5625, 0.5625, 0.1875],
             [0.1875, 0.5625, 0.5625, 0.1875],
             [0.0625, 0.1875, 0.1875, 0.0625]], dtype=dtype.as_numpy_dtype)
    else:
        assert False
    return x

def upsample_init_value(shape, alg='nn', dtype=None):
    # Return numpy array for initialization value

    print("upsample initializer desired shape, type: {}. {}".format(shape,dtype))
    f = upsample_filt(alg, dtype)

    filtnr, filtnc, kout, kin = shape
    assert kout == kin  # for now require equality
    if kin > kout:
        wstr = "upsample filter has more inputs ({}) than outputs ({}). Using truncated identity".format(kin, kout)
        logging.warning(wstr)

    xinit = np.zeros(shape)
    for i in range(kout):
        xinit[:, :, i, i] = f

    return xinit

def upsample_initializer(shape, alg='nn', dtype=None):
    xinit = upsample_init_value(shape, alg, dtype)
    if ISPY3:
        return K.constant(xinit, dtype=dtype)  # using K.variable causes strange init/async/race(?) errs in Py3+tf1.14
    else:
        return K.variable(value=xinit, dtype=dtype)  # this worked OK in Py2+tf1.13

# could use functools.partial etc
def upsamp_init_nn(shape, dtype=None, partition_info=None):
    return upsample_initializer(shape, 'nn', dtype)
def upsamp_init_bl(shape, dtype=None, partition_info=None):
    return upsample_initializer(shape, 'bl', dtype)

def make_kernel_regularizer(kinit, kweightdecay):
    # kinit: numpy array with initial value of tensor

    k0 = K.constant(kinit)

    def reg(wmat):
        assert k0.shape.as_list() == wmat.shape.as_list()
        return kweightdecay * K.sum(K.square(Subtract([k0, wmat])))

    return reg

def deconv_2x_upsampleinit(x, nf, ks, name, wd, wdmode):
    # init around upsampling
    #
    # wd: None, or [2]: kernel, then bias weight decay
    # wdmode: 0 'aroundzero' or  1 'aroundinit'

    assert ks == 4, "Filtersize must be 4, using upsamp_init_bl"
    # nf must also equal number of channels in x

    if wdmode == 0:  # 'aroundzero':
        kernel_reg = tf.keras.regularizers.l2(wd[0]) if wd else None
        bias_reg = tf.keras.regularizers.l2(wd[1]) if wd else None
        logging.info("Deconv: regularization around zero with weights {}".format(wd))
    elif wdmode == 1:  # 'aroundinit'
        kshape = (ks, ks, nf, nf)
        kinit = upsample_init_value(kshape, 'bl')
        kernel_reg = make_kernel_regularizer(kinit, wd[0])
        bias_reg = tf.keras.regularizers.l2(wd[1]) if wd else None
        logging.info("Deconv: regulization around init with weights {}".format(wd))
    else:
        assert False

    x = tf.keras.layers.Conv2DTranspose(nf, (ks, ks), strides=2,
                        padding='same', name=name,
                        kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg,
                        kernel_initializer=upsamp_init_bl,
                        bias_initializer=tf.keras.initializers.constant(0.0))(x)
    logging.info("Using 2xdeconv w/init around upsample, wdmode={}, wd={}.".format(wdmode, wd))

    return x

def convblock(x0, nf, namebase, kernel_reg):
    '''
    Three 3x3 convs with tf.keras.layers.PReLU and with results concatenated

    :param x0:
    :param nf:
    :param namebase:
    :param kernel_reg:
    :return:
    '''
    x1 = conv(x0, nf, 3, kernel_reg, name="cblock-{}-{}".format(namebase, 1))
    x1 = prelu(x1, "cblock-{}-{}-prelu".format(namebase, 1))
    x2 = conv(x1, nf, 3, kernel_reg, name="cblock-{}-{}".format(namebase, 2))
    x2 = prelu(x2, "cblock-{}-{}-prelu".format(namebase, 2))
    x3 = conv(x2, nf, 3, kernel_reg, name="cblock-{}-{}".format(namebase, 3))
    x3 = prelu(x3, "cblock-{}-{}-prelu".format(namebase, 3))
    x = tf.keras.layers.Concatenate(name="cblock-{}".format(namebase))([x1, x2, x3])
    return x

def stageCNN(x, nfout, stagety, stageidx, kernel_reg,
             nfconvblock=128, nconvblocks=5, nf1by1=128):
    # stagety: 'map' or 'paf'

    for iCB in range(nconvblocks):
        namebase = "{}-stg{}-cb{}".format(stagety, stageidx, iCB)
        x = convblock(x, nfconvblock, namebase, kernel_reg)
    x = conv(x, nf1by1, 1, kernel_reg, name="{}-stg{}-1by1-1".format(stagety, stageidx))
    x = prelu(x, "{}-stg{}-1by1-1-prelu".format(stagety, stageidx))
    x = conv(x, nfout, 1, kernel_reg, name="{}-stg{}-1by1-2".format(stagety, stageidx))
    # x = prelu(x, "{}-stg{}-1by1-2-prelu".format(stagety, stageidx))
    return x

def stageCNNwithDeconv(x, nfout, stagety, stageidx, kernel_reg,
                       nfconvblock=128, nconvblocks=5, ndeconvs=2, nf1by1=128):
    '''
    Like stageCNN, but with ndeconvs Deconvolutions to increase imsz by 2**ndeconvs
    :param x:
    :param nfout:
    :param stagety:
    :param stageidx:
    :param kernel_reg:
    :param nfconvblock:
    :param nconvblocks:
    :param ndeconvs:
    :param nfdeconv:
    :param nf1by1:
    :return:
    '''

    for iCB in range(nconvblocks):
        namebase = "{}-stg{}-cb{}".format(stagety, stageidx, iCB)
        x = convblock(x, nfconvblock, namebase, kernel_reg)

    nfilt = x.shape.as_list()[-1]
    logging.info("Adding {} deconvs with nfilt={}".format(ndeconvs, nfilt))

    DCFILTSZ = 4
    for iDC in range(ndeconvs):
        dcname = "{}-stg{}-dc{}".format(stagety, stageidx, iDC)
        x = deconv_2x_upsampleinit(x, nfilt, DCFILTSZ, dcname, None, 0)
        x = prelu(x, "{}-prelu".format(dcname))

    x = conv(x, nf1by1, 1, kernel_reg, name="{}-stg{}-1by1-1".format(stagety, stageidx))
    x = prelu(x, "{}-stg{}-postDC-1by1-1-prelu".format(stagety, stageidx))
    x = conv(x, nfout, 1, kernel_reg, name="{}-stg{}-postDC-1by1-2".format(stagety, stageidx))
    # x = conv(x, nfout, 1, kernel_reg, name="{}-stg{}-1by1-2".format(stagety, stageidx))
    # x = prelu(x, "{}-stg{}-postDC-1by1-2-prelu".format(stagety, stageidx))
    return x

def model_train(imszuse, kernel_reg, backbone='resnet50_8px', backbone_weights=None,
                       nPAFstg=5, nMAPstg=1,
                       nlimbsT2=38, npts=19, doDC=True, nDC=2,conf=None):
    '''

    :param imszuse: (imnr, imnc) raw image size, possibly adjusted to be 0 mod 8
    :param kernel_reg:
    :param nlimbsT2:
    :param npts:
    :return: tf.keras.models.Model.
        Inputs: [img]
        Outputs: [paf_1, ... paf_nPAFstg, map_1, ... map_nMAPstg]
    '''

    imnruse, imncuse = imszuse
    assert imnruse % 8 == 0, "Image size must be divisible by 8"
    assert imncuse % 8 == 0, "Image size must be divisible by 8"

    #paf_input_shape_hires = imszuse + (nlimbsT2,)
    #map_input_shape_hires = imszuse + (npts,)

    inputs = []

    # This is hardcoded to dim=3 due to VGG pretrained weights
    img_input = tf.keras.layers.Input(shape=imszuse + (3,), name='input_img')
    mask_input = tf.keras.layers.Input(shape=imszuse, name='input_mask')

    # paf_weight_input = tf.keras.layers.Input(shape=paf_input_shape,
    #                          name='input_paf_mask')
    # map_weight_input = tf.keras.layers.Input(shape=map_input_shape,
    #                          name='input_part_mask')
    # paf_weight_input_hires = tf.keras.layers.Input(shape=paf_input_shape_hires,
    #                                name='input_paf_mask_hires')
    # map_weight_input_hires = tf.keras.layers.Input(shape=map_input_shape_hires,
    #                                name='input_part_mask_hires')
    inputs.append(img_input)
    inputs.append(mask_input)
    # inputs.append(paf_weight_input)
    # inputs.append(map_weight_input)
    # inputs.append(paf_weight_input_hires)
    # inputs.append(map_weight_input_hires)

    img_normalized = tf.keras.layers.Lambda(lambda z: z / 256. - 0.5)(img_input)  # [-0.5, 0.5] Isn't this really [-0.5, 0.496]
    # sub mean?

    # backbone
    if backbone == 'vgg':
        imszBB = (imnruse // 8, imncuse // 8)  # imsz post backbone
        #paf_input_shape = imszvgg + (nlimbsT2,)
        #map_input_shape = imszvgg + (npts,)
        backboneF = vgg_cpm.vgg19_truncated(img_normalized, kernel_reg)
        # sz should be (bsize, imszvgg[0], imszvgg[1], nchans)
        print(backboneF.shape.as_list()[1:])
        assert backboneF.shape.as_list()[1:] == list(imszBB + (128,))
    elif backbone == 'resnet50_8px':
        #imszBB = (imnruse / 8, imncuse / 8)
        #inputshape = img_normalized.shape.as_list()[1:]
        backboneMdl = imagenet_resnet.ResNet50_8px(include_top=False, weights=backbone_weights, input_tensor=img_normalized, pooling=None)
        backboneF = backboneMdl.output
    else:
        assert False, "Unrecognized backbone: {}".format(backbone)

    print("BackBone {} with weights {} instantiated; output shape is {}".format(
        backbone, backbone_weights, backboneF.shape.as_list()[1:]))

    # PAF 1..nPAFstg
    xpaflist = []
    xstagein = backboneF
    for iPAFstg in range(nPAFstg):
        xstageout = stageCNN(xstagein, nlimbsT2, 'paf', iPAFstg, kernel_reg)
        xpaflist.append(xstageout)
        xstagein = tf.keras.layers.Concatenate(name="paf-stg{}".format(iPAFstg))([backboneF, xstageout])

    # MAP
    xmaplist = []
    for iMAPstg in range(nMAPstg):
        xstageout = stageCNN(xstagein, npts, 'map', iMAPstg, kernel_reg)
        xmaplist.append(xstageout)
        xstagein = tf.keras.layers.Concatenate(name="map-stg{}".format(iMAPstg))([backboneF, xpaflist[-1], xstageout])

    xmaplistDC = []
    if doDC:
        # xstagein is ready/good from MAP loop
        xstageout = stageCNNwithDeconv(xstagein, npts, 'map', nMAPstg, kernel_reg, ndeconvs=nDC)
        xmaplistDC.append(xstageout)
        dc_scale = 2**conf.op_hires_ndeconv
    else:
        dc_scale = 1

    assert len(xpaflist) == nPAFstg
    assert len(xmaplist) == nMAPstg

    scale_hires = conf.op_label_scale // dc_scale  # downsample scale of hires (postDC) relative to network input
    mask_low_res = mask_input[:,::conf.op_label_scale,::conf.op_label_scale,None]
    mask_high_res = mask_input[:,::scale_hires,::scale_hires,None]
    # for ndx, xp in enumerate(xpaflist):
    #     xpaflist[ndx] = xp*mask_input[:,::conf.op_label_scale,::conf.op_label_scale,None]
    # for ndx, xp in enumerate(xmaplist):
    #     xmaplist[ndx] = xp*mask_input[:,::conf.op_label_scale,::conf.op_label_scale,None]
    # for ndx, xp in enumerate(xmaplistDC):
    #     xmaplistDC[ndx] = xp*mask_input[:,::scale_hires,::scale_hires,None]

    outputs = xpaflist + xmaplist + xmaplistDC +[mask_low_res,mask_high_res]

    # w1 = apply_mask(stage1_branch1_out, paf_weight_input, 1, 1)
    # w2 = apply_mask(stage1_branch2_out, map_weight_input, 1, 2)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def configure_losses(model, dc_on=True, dcNum=None, dc_blur_rad_ratio=None, dc_wtfac=None,use_mask=True):
    '''
    
    :param model: 
    :param bsize: 
    :param dc_on: True if deconv/hires is on
    :param dcNum: number of 2x deconvs applied 
    :param dc_blur_rad_ratio: The ratio blur_rad_hires/blur_rad_lores
    :param dc_wtfac: Weighting factor for hi-res
    
    :return: losses, loss_weights. both dicts whose keys are .names of model.outputs
    '''

    def eucl_loss(x, y, mask = None, use_mask=True):
        diff = K.square(x - y)
        if use_mask and (mask is not None):
            loss = K.sum(diff*mask) /  2.
        else:
            loss = K.sum(diff) / 2.
        return loss

    losses = {}
    loss_weights = {}
    loss_weights_vec = []

    outputs = model.outputs[:-2]
    masks = model.outputs[-2:]
    # this is fantastically ugly.
    layers = model.layers
    for output in outputs:
        # Not sure how to get from output Tensor to its layer. Using
        # output Tensor name doesn't work with model.compile
        
        layers_matching_output = [layer for layer in layers if layer.output is output]
        assert len(layers_matching_output) == 1, "Found multiple layers for output."
        layer_matching_output = layers_matching_output[0]
        output_layer_name = layer_matching_output.name

        #output_layer_names = [layer.name for layer in layers if layer.output == output]
        #assert len(output_layer_names) == 1, "Found multiple layers for output."
        #output_layer_name = output_layer_names[0]

        #output_layer_name = output.node.layer.name  # ALT 2023-03-30: Maybe this will work?

        print('output_layer_name: ', output_layer_name)

        if "postDC" in output_layer_name:
            losses[output_layer_name] = lambda x, y: eucl_loss(x, y, masks[1],use_mask=use_mask)
            assert dc_on, "Found post-deconv layer"
            # left alone, L2 loss will be ~dc_blur_rad_ratio**2 larger for hi-res wrt lo-res
            loss_weights[output_layer_name] = float(dc_wtfac) / float(dc_blur_rad_ratio)**2
        else:
            losses[output_layer_name] = lambda x, y: eucl_loss(x, y, masks[0],use_mask=use_mask)
            loss_weights[output_layer_name] = 1.0

        logging.info('Configured loss for output name {}, loss_weight={}'.format(output_layer_name, loss_weights[output_layer_name]))
        loss_weights_vec.append(loss_weights[output_layer_name])

    return losses, loss_weights, loss_weights_vec

def model_test(imszuse, kernel_reg,
                      backbone='resnet50_8px',
                      nPAFstg=5, nMAPstg=1, nlimbsT2=38, npts=19,
                      doDC=True, nDC=2, fullpred=False):
    '''
    See model_train
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
    imszvgg = (imnruse//8, imncuse//8)  # imsz post VGG ftrs

    img_input = tf.keras.layers.Input(shape=imszuse + (3,), name='input_img')

    img_normalized = tf.keras.layers.Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # backbone
    if backbone == 'vgg':
        imszBB = (imnruse // 8, imncuse // 8)  # imsz post backbone
        #paf_input_shape = imszvgg + (nlimbsT2,)
        #map_input_shape = imszvgg + (npts,)
        backboneF = vgg_cpm.vgg19_truncated(img_normalized, 0.)
        # sz should be (bsize, imszvgg[0], imszvgg[1], nchans)
        print(backboneF.shape.as_list()[1:])
        assert backboneF.shape.as_list()[1:] == list(imszBB + (128,))
    elif backbone == 'resnet50_8px':
        #imszBB = (imnruse / 8, imncuse / 8)
        #inputshape = img_normalized.shape.as_list()[1:]
        backboneMdl = imagenet_resnet.ResNet50_8px(include_top=False,
                                                   weights=None,
                                                   input_tensor=img_normalized,
                                                   pooling=None)
        backboneF = backboneMdl.output

    else:
        assert False, "Unrecognized backbone: {}".format(backbone)

    print("BackBone {} instantiated; output shape is {}".format(
        backbone, backboneF.shape.as_list()[1:]))

    # PAF 1..nPAFstg
    xpaflist = []
    xstagein = backboneF
    for iPAFstg in range(nPAFstg):
        # Using None for kernel_reg is nonsensical but shouldn't hurt in test mode
        xstageout = stageCNN(xstagein, nlimbsT2, 'paf', iPAFstg, kernel_reg)
        xpaflist.append(xstageout)
        xstagein = tf.keras.layers.Concatenate(name="paf-stg{}".format(iPAFstg))([backboneF, xstageout])

    # MAP
    xmaplist = []
    for iMAPstg in range(nMAPstg):
        # Using None for kernel_reg is nonsensical but shouldn't hurt in test mode
        xstageout = stageCNN(xstagein, npts, 'map', iMAPstg, kernel_reg)
        xmaplist.append(xstageout)
        xstagein = tf.keras.layers.Concatenate(name="map-stg{}".format(iMAPstg))([backboneF, xpaflist[-1], xstageout])

    xmaplistDC = []
    if doDC:
        # xstagein is ready/good from MAP loop
        xstageout = stageCNNwithDeconv(xstagein, npts, 'map', nMAPstg, kernel_reg, ndeconvs=nDC)
        xmaplistDC.append(xstageout)

    assert len(xpaflist) == nPAFstg
    assert len(xmaplist) == nMAPstg

    if fullpred:
        outputs = xpaflist + xmaplist + xmaplistDC
    elif doDC:
        outputs = [xpaflist[-1], xmaplistDC[-1], ]
    else:
        outputs = [xpaflist[-1], xmaplist[-1], ]

    model = tf.keras.models.Model(inputs=[img_input], outputs=outputs)

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

def compute_padding_imsz_net(imsz, rescale, bb_ds_scale, ndeconv):
    '''
    :param imsz: [2] raw im size
    :param rescale: float, desired rescale.
    :return:
    '''

    # in tfdatagen, the input pipeline is read->pad->rescale/distort->ready_for_network

    # in the network, it goes inputres->bbres(downsamp currently 8x)->deconvres(2x upsamp ndeconv times)

    # we set the initial padding so all scaling is 'perfect' ie the initial rescale, then bbdownsample etc.

    def checkint(x, name):
        assert isinstance(x, int) or x.is_integer(), "Expect {} to be integral value".format(name)

    # checkint(rescale, 'rescale')
    checkint(bb_ds_scale, 'bb_ds_scale')
    checkint(ndeconv, 'ndeconv')

    dc_scale = 2**ndeconv
    scale_hires = bb_ds_scale / dc_scale  # downsample scale of hires (postDC) relative to network input
    assert scale_hires.is_integer(), "scale_hires is non-integral: {}".format(scale_hires)

    imsz_pad_should_be_divisible_by = int(rescale * bb_ds_scale)
    dsfac = imsz_pad_should_be_divisible_by
    def roundupeven(x):
        return int(np.ceil(x/dsfac)) * dsfac

    imsz_pad = (roundupeven(imsz[0]), roundupeven(imsz[1]))
    padx = imsz_pad[1] - imsz[1]
    pady = imsz_pad[0] - imsz[0]
    imsz_net = (int(imsz_pad[0]/rescale), int(imsz_pad[1]/rescale))
    imsz_bb = (int(imsz_pad[0]/rescale/bb_ds_scale), int(imsz_pad[1]/rescale/bb_ds_scale))
    imsz_dc = (int(imsz_pad[0]/rescale/scale_hires), int(imsz_pad[1]/rescale/scale_hires))

    netinoutscale = scale_hires

    return padx, pady, imsz_pad, imsz_net, imsz_bb, imsz_dc, netinoutscale

def update_conf(conf):
    '''
    Update conf in-place
    :param conf:
    :return:
    '''

    if not conf.op_hires:
        assert conf.op_hires_ndeconv == 0
        # stuff should work with ndeconv=0

    (conf.op_im_padx, conf.op_im_pady, conf.op_imsz_pad, conf.op_imsz_net,
     conf.op_imsz_lores, conf.op_imsz_hires, conf.op_net_inout_scale) = \
        compute_padding_imsz_net(conf.imsz, conf.rescale,
                                 conf.op_label_scale,
                                 conf.op_hires_ndeconv)
    logging.info("OP size stuff: imsz={}, imsz_pad={}, imsz_net={}, imsz_lores={}, imsz_hires={}, rescale={}".format(
        conf.imsz, conf.op_imsz_pad, conf.op_imsz_net, conf.op_imsz_lores, conf.op_imsz_hires,
        conf.rescale))

    if conf.normalize_img_mean:
        conf.normalize_img_mean = False
        logging.warning("Turning off normalize_img_mean. Openpose does its own normalization.")

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

def get_train_data_filename(conf, name):
    tdf = 'traindata' if name == 'deepnet' else conf.expname + '_' + name + '_traindata'
    train_data_file = os.path.join(conf.cachedir, tdf)
    return train_data_file

def update_op_graph(op_graph):
    # rearranges so that the new edges are always connected to previous ones.
    assert  len(op_graph) > 1, 'Openpose affinity graph is empty'
    new_graph = [op_graph[0]]
    remaining = []            # edges that remain
    done = list(op_graph[0])        # nodes that are connected
    for k in op_graph[1:]:
        if k[0] in done:
            done.append(k[1])
            new_graph.append(k)
        elif k[1] in done:
            done.append(k[0])
            new_graph.append(k)
        else:
            remaining.append(k)

        for idx in reversed(range(len(remaining))):
            j = remaining[idx]
            if j[0] in done:
                done.append(j[1])
                new_graph.append(j)
                remaining.pop(idx)
            elif j[1] in done:
                done.append(j[0])
                new_graph.append(j)
                remaining.pop(idx)
    return  new_graph


def training(conf, name='deepnet',restore=False, model_file=None):
    # base_lr = conf.op_base_lr
    base_lr = conf.get('op_base_lr',4e-5) * conf.get('learning_rate_multiplier',1.)
    batch_size = conf.batch_size  # Gines 10
    gamma = conf.gamma  # Gines 1/2
    stepsize = int(conf.decay_steps)  # after each stepsize iterations update learning rate: lr=lr*gamma
      # Gines much larger: 200k, 300k, then every 60k
    iterations_per_epoch = conf.display_step
    max_iter = conf.dl_steps//iterations_per_epoch
    last_epoch = 0

    assert conf.dl_steps % iterations_per_epoch == 0, 'For open-pose dl steps must be a multiple of display steps'
    assert conf.save_step % iterations_per_epoch == 0, 'For open-pose save steps must be a multiple of display steps'

    # need this to set default
    #save_time = conf.get('save_time', None)

    if name == 'deepnet':
        train_data_file = os.path.join(conf.cachedir, 'traindata')
    else:
        train_data_file = os.path.join(conf.cachedir,conf.expname + '_' + name + '_traindata')

    with open(train_data_file, 'wb') as td_file:
        pickle.dump([{},conf], td_file, protocol=2)
    logging.info('Saved config to {}'.format(train_data_file))

    #model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{epoch:d}')
    assert not conf.normalize_img_mean, "OP currently performs its own img input norm"
    assert not conf.normalize_batch_mean, "OP currently performs its own img input norm"
    model = model_train(conf.op_imsz_net,
                               conf.op_weight_decay_kernel,
                               backbone=conf.op_backbone,
                               backbone_weights=conf.op_backbone_weights,
                               nPAFstg=conf.op_paf_nstage,
                               nMAPstg=conf.op_map_nstage,
                               nlimbsT2=len(conf.op_affinity_graph) * 2,
                               npts=conf.n_classes,
                               doDC=conf.op_hires,
                               nDC=conf.op_hires_ndeconv,conf=conf)

    if conf.op_backbone=='vgg' and conf.op_backbone_weights=='imagenet':
        logging.info("Loading vgg19 weights...")
        vgg_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        for layer in model.layers:
            if layer.name in vgg_cpm.from_vgg:
                vgg_layer_name = vgg_cpm.from_vgg[layer.name]
                layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                logging.info("Loaded VGG19 layer: {}->{}".format(layer.name, vgg_layer_name))

    if model_file is not None:
        try:
            logging.info("Loading the weights from {}.. ".format(model_file))
            model.load_weights(model_file)
        except Exception as e:
            logging.info(f'Could not initialize model weights from {model_file}')
            logging.info(e)


    elif restore:
        latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
        logging.info("Loading the weights from {}.. ".format(latest_model_file))
        model.load_weights(latest_model_file)
        last_iter = re.search(f'{name}-(\d*)$',latest_model_file).groups()[0]
        last_epoch = int(last_iter)//iterations_per_epoch

    # prepare generators
    PREPROCFN = 'ims_locs_preprocess_openpose'
    trntfr = os.path.join(conf.cachedir, conf.trainfilename) + PoseTools.dbformat_to_extension(conf.db_format)
    train_di = tfdatagen.make_data_generator(trntfr, conf, True, True, PREPROCFN)
    train_di2 = tfdatagen.make_data_generator(trntfr, conf, True, True, PREPROCFN)
    val_di = tfdatagen.make_data_generator(trntfr, conf, False, False, PREPROCFN)

    # For debugging data gen pipeline
    # kk = tfdatagen.make_data_generator(trntfr, conf, True, True, PREPROCFN, debug=True)
    # while True:
    #     a = next(kk)

    assert conf.op_label_scale == 8
    logging.info("Your label_blur_rads (hi/lo)res are {}/{}".format(
        conf.op_map_hires_blur_rad, conf.op_map_lores_blur_rad))
    losses, loss_weights, loss_weights_vec = \
        configure_losses(model, dc_on=conf.op_hires,
                         dc_blur_rad_ratio=conf.op_map_hires_blur_rad / conf.op_map_lores_blur_rad,
                         dc_wtfac=2.5,use_mask=conf.is_multi&conf.multi_loss_mask)

    def lr_decay(epoch):  # epoch is 0-based
        initial_lrate = base_lr
        steps = (epoch+1) * iterations_per_epoch
        lrate = initial_lrate * math.pow(gamma, math.floor(steps / stepsize))
        return lrate

    # Callback to do writing pring stuff.
    # See apt_dpk_callbacks/TrainDataLogger, future refactor
    class OutputObserver(tf.keras.callbacks.Callback):
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
            self.save_start = time()
            self.force = False
            self.prev_models = []

        def on_epoch_end(self, epoch, logs={}):
            step = (epoch+1) * iterations_per_epoch
            val_x, val_y = next(self.val_di)
            val_out = self.model.predict(val_x, batch_size=batch_size)
            val_loss_full = self.model.evaluate(val_x, val_y, batch_size=batch_size, verbose=0)
            val_loss_K = val_loss_full[0]  # want Py 3 unpack
            val_loss_full = val_loss_full[1:]
            val_loss = dot(val_loss_full, loss_weights_vec)
            #val_loss = np.nan
            train_x, train_y = next(self.train_di)
            train_out = self.model.predict(train_x, batch_size=batch_size)
            train_loss_full = self.model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
            train_loss_K = train_loss_full[0]  # want Py 3 unpack
            train_loss_full = train_loss_full[1:]
            train_loss = dot(train_loss_full, loss_weights_vec)
            #train_loss = np.nan
            lr = K.eval(self.model.optimizer.lr)

            # dist only for last MAP layer (will be hi-res if deconv is on)
            predhmval = val_out[-3]
            predhmval = clip_heatmap_with_warn(predhmval)
            # (bsize, npts, 2), (x,y), 0-based

            if self.config.is_multi:
                nclustermax = self.config.max_n_animals
            else:
                nclustermax = self.config.op_hmpp_nclustermax
            predlocsval, _ = \
                heatmap.get_weighted_centroids_with_argmax(predhmval, floor=self.config.op_hmpp_floor, nclustermax=nclustermax,sz=self.config.op_map_lores_blur_rad*2,is_multi=self.config.is_multi)
            gtlocs, _ = heatmap.get_weighted_centroids_with_argmax(val_y[-1], floor=self.config.op_hmpp_floor, nclustermax=nclustermax,sz=self.config.op_map_lores_blur_rad*2,is_multi=self.config.is_multi)

            tt1 = predlocsval - gtlocs
            tt1 = np.sqrt(np.sum(tt1 ** 2, -1))  # [bsize x ncls]
            val_dist = np.nanmean(tt1)  # this dist is in op_scale-downsampled space
                                        # *self.config.op_label_scale

            # NOTE train_dist uses argmax
            tt1 = PoseTools.get_pred_locs(train_out[-3]) - \
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

            train_data_file = get_train_data_filename(self.config, name)

            json_data = {}
            for x in self.train_info.keys():
                json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
            with open(train_data_file + '.json', 'w') as json_file:
                json.dump(json_data, json_file)
            with open(train_data_file, 'wb') as td:
                pickle.dump([self.train_info, conf], td, protocol=2)

            m_file = str(os.path.join(conf.cachedir, name + '-{}'.format(int(step))))
            if conf.save_time is None:
                if step % conf.save_step == 0:
                    model.save(m_file)
                    self.prev_models.append(m_file)
            else:
                if time() - self.save_start > conf.save_time*60:
                    self.save_start = time()
                    model.save(m_file)
                    self.prev_models.append(m_file)

            if len(self.prev_models) > conf.maxckpt:
                for curm in self.prev_models[:-conf.maxckpt]:
                    if os.path.exists(curm):
                        os.remove(curm)
                _ = self.prev_models.pop(0)


    # configure callbacks
    lrate = tf.keras.callbacks.LearningRateScheduler(lr_decay)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(val_di
    #     model_file, monitor='loss', verbose=0, save_best_only=False,
    #     save_weights_only=True, mode='min', period=conf.save_step)
    obs = OutputObserver(conf, [train_di2, val_di])
    callbacks_list = [lrate, obs]  #checkpoint,

    # optimizer = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)#, clipnorm=1.)
    # Mayank 20190423 - Adding clipnorm so that the loss doesn't go to zero.
    # Epsilon: could just leave un-speced, None leads to default in tf1.14 at least
    # Decay: 0.0 bc lr schedule handled above by callback/LRScheduler
    #optimizer = Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # ALT 2023-03-29 --- Dropped "decay" keyword argument b/c no longer supported
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=base_lr, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
      # Have to use "legacy" version b/c without ".legacy" you get an "experimental" version, which tries to use .numpy(), 
      # which doesn't work b/c we've disabled eager execution via tf.compat.v1.disable_v2_behavior() in APT_interface.main().  -- ALT, 2023-03-31

    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
    #model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer, run_eagerly=True)  # for debugging

    logging.info("Your model.metrics_names are {}".format(model.metrics_names))

    # save initial model
    # model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(0))))

    model.fit(train_di,
              steps_per_epoch=iterations_per_epoch,
              epochs=max_iter-1,
              callbacks=callbacks_list,
              verbose=0,
              initial_epoch=last_epoch
              )

    # force saving in case the max iter doesn't match the save step.
    model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(int(max_iter*iterations_per_epoch)))))
    obs.on_epoch_end(max_iter-1)

def clip_heatmap_with_warn(predhm):
    '''

    :param predhm:
    :return: clipped predhm; always is a copy
    '''

    if np.any(predhm < 0.0):
        PTILES = [1, 5, 10, 50, 99]
        ptls = np.percentile(predhm, PTILES)
        warnstr = 'Prediction heatmap has negative els! PTILES {}: {}'.format(PTILES, ptls)
        logging.warning(warnstr)

        predhm_clip = predhm.copy()
        predhm_clip[predhm_clip < 0.0] = 0.0
    else:
        predhm_clip = predhm.copy()

    return predhm_clip

def compare_conf_traindata(conf, name):
    '''
    Compare given conf to one on disk
    :param conf:
    :return:
    '''

    #cdir = conf.cachedir
    tdfile = get_train_data_filename(conf, name)
    if os.path.exists(tdfile):
        with open(tdfile, 'rb') as f:
            td = pickle.load(f, encoding='latin1')
        conftrain = td[1]
        logging.info("Comparing prediction config to training config within {}...".format(tdfile))
        d0 = vars(conf)
        d1 = vars(conftrain)
        util.dictdiff(d0, d1, logging.info)
        logging.info("... done comparing configs")
    else:
        wstr = "Cannot find traindata file {}. Not checking predict vs train config.".format(tdfile)
        logging.warning(wstr)


def get_pred_fn(conf, model_file=None, name='deepnet', edge_ignore=0):
    '''

    :param conf:
    :param model_file:
    :param name:
    :param edge_ignore: ignore pad + this
    :return:
    '''
    #(imnr, imnc) = conf.imsz
    #imnr_use = imszcheckcrop(imnr, 'row')
    #imnc_use = imszcheckcrop(imnc, 'column')
    #imszuse = (imnr_use, imnc_use)
    #conf.imszuse = imszuse

    compare_conf_traindata(conf, name)

    # TODO: Theoretically probably should deep-copy conf since it is used in the returned fn

    assert not conf.normalize_img_mean, "OP currently performs its own img input norm"
    assert not conf.normalize_batch_mean, "OP currently performs its own img input norm"
    model = model_test(conf.op_imsz_net,
                       conf.op_weight_decay_kernel,
                              backbone=conf.op_backbone,
                              nPAFstg=conf.op_paf_nstage,
                              nMAPstg=conf.op_map_nstage,
                              nlimbsT2=len(conf.op_affinity_graph) * 2,
                              npts=conf.n_classes,
                              doDC=conf.op_hires,
                              nDC=conf.op_hires_ndeconv,
                              fullpred=conf.op_pred_raw)
    if model_file is None:
        latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    else:
        latest_model_file = model_file
    logging.info("Loading the weights from {}.. ".format(latest_model_file))
    model.load_weights(latest_model_file)
    # thre2 = conf.get('op_param_paf_thres',0.05)
    thre_hm = conf.get('op_param_hmap_thres',0.1)
    thre_paf = conf.get('op_param_paf_thres',0.05)

    op_pred_simple = conf.get('op_pred_simple', False)
    op_inference_old = conf.get('op_inference_old', False)
    if not op_pred_simple:
        parpool = multiprocessing.Pool(conf.batch_size)
    else:
        parpool = None

    def pred_fn(all_f, retrawpred=conf.op_pred_raw):
        '''

        :param all_f: must have precisely 3 chans (if b/w, already tiled)
        :param rawpred: bool flag
        :return:
        '''

        assert all_f.shape[0] == conf.batch_size
        locs_sz = (conf.batch_size, conf.n_classes, 2)
        locs_dummy = np.zeros(locs_sz)
        ret = tfdatagen.ims_locs_preprocess_openpose(all_f, locs_dummy, conf, False, gen_target_hmaps=False,mask=np.ones_like(all_f[...,0])>0)
        ims = ret[0]

        model_preds = []
        for ix in range(ims.shape[0]):
            model_preds.append(model.predict(ims[ix:ix+1,...]))

        # all_infered = []
        # for ex in range(xs.shape[0]):
        #     infered = do_inference(model_preds[-1][ex,...],model_preds[-2][ex,...],conf, thre1, thre2)
        #     all_infered.append(infered)

        if op_pred_simple:
            predhm = np.array([m[-1][0,...] for m in model_preds])  # this is always the last/final MAP hmap
            ret_dict = pred_simple(predhm,conf,edge_ignore,retrawpred,ims,model_preds)
        else:
            in_args = [[mm[-1][0,...],mm[-2][0,...],conf,thre_hm,thre_paf] for mm in model_preds]
            fn = do_inference_old if op_inference_old else do_inference
            if len(in_args)>1:
                cur_locs = parpool.starmap(fn,in_args)
            else:
                cur_locs = [fn(*in_args[0])]
            locs = np.array(cur_locs).copy()

            # undo rescale
            locs = PoseTools.unscale_points(locs, conf.rescale, conf.rescale)

            # undo padding
            locs[..., 0] -= conf.op_im_padx // 2
            locs[..., 1] -= conf.op_im_pady // 2

            if not conf.is_multi:
                locs = locs[:,0,...]
            ret_dict = {}
            ret_dict['locs'] = locs  # * conf.rescale
            if retrawpred:
                ret_dict['pred_hmaps'] = model_preds
                ret_dict['ims'] = ims

        return ret_dict

    def close_fn():
        K.clear_session()

    return pred_fn, close_fn, latest_model_file


def pred_simple(predhm,conf,edge_ignore,retrawpred,ims,model_preds):
    predhm_clip = clip_heatmap_with_warn(predhm)
    locs_sz = (conf.batch_size, conf.n_classes, 2)
    thre_hm = conf.get('op_param_hmap_thres',0.1)

    # ignore edges, mod prehm_clip in-place
    if edge_ignore > 0:
        padx0 = conf.op_im_padx // 2
        padx1 = conf.op_im_padx - padx0
        pady0 = conf.op_im_pady // 2
        pady1 = conf.op_im_pady - pady0
        igx0 = padx0 + edge_ignore
        igx1 = padx1 + edge_ignore
        igy0 = pady0 + edge_ignore
        igy1 = pady1 + edge_ignore
        bsize = predhm_clip.shape[0]
        npts = predhm_clip.shape[3]
        for ib in range(bsize):
            for ipt in range(npts):
                minval = predhm_clip[ib, :, :, ipt].min()
                predhm_clip[ib, :igy0, :, ipt] = minval
                predhm_clip[ib, :, :igx0, ipt] = minval
                predhm_clip[ib, -igy1:, :, ipt] = minval
                predhm_clip[ib, :, -igx1:, ipt] = minval

    # (bsize, npts, 2), (x,y), 0-based
    # predlocs_argmax = PoseTools.get_pred_locs(predhm_clip)
    predlocs_wgtcnt, predlocs_argmax = \
        heatmap.get_weighted_centroids_with_argmax(predhm_clip, floor=conf.op_hmpp_floor, nclustermax=conf.op_hmpp_nclustermax)
    # predlocs_wgtcnt0 = heatmap.get_weighted_centroids_with_argmax(predhm_clip,
    #                                                  floor=0,
    #                                                  nclustermax=1)
    assert predlocs_argmax.shape == locs_sz
    assert predlocs_wgtcnt.shape == locs_sz
    print("HMAP POSTPROC, floor={}, nclustermax={}".format(conf.op_hmpp_floor, conf.op_hmpp_nclustermax))

    # OTHER OUTPUTS
    # simple diagnostic in lieu of do_inf for now
    NCLUSTER_MAX = 5
    num_peaks = np.zeros((conf.batch_size, conf.n_classes))
    pks_with_score = np.zeros((conf.batch_size, conf.n_classes,
                               NCLUSTER_MAX, 3))  # last D: x, y, score
    pks_with_score[:] = np.nan
    pks_with_score_cmpt = pks_with_score.copy()
    for ib in range(conf.batch_size):
        for ipt in range(conf.n_classes):
            hmthis = predhm_clip[ib, :, :, ipt]
            pks_with_score_this = heatmap.find_peaks(hmthis, thre_hm)
            if len(pks_with_score_this) > 0:
                pks_with_score_this = np.stack(pks_with_score_this)  # npk x 3
                irows = min(NCLUSTER_MAX, pks_with_score_this.shape[0])
                pks_with_score[ib, ipt, :irows, :] = pks_with_score_this[:irows, :]

            a, mu, sig, _ = heatmap.compactify_hmap(hmthis, floor=0.1, nclustermax=NCLUSTER_MAX)
            mu = mu[::-1, :] - 1.0  # now x,y and 0b
            #mu -= 1.0
            a_mu = np.vstack((mu, a)).T
            assert a_mu.shape == (NCLUSTER_MAX, 3)
            pks_with_score_cmpt[ib, ipt, :, :] = a_mu

            num_peaks[ib, ipt] = len(pks_with_score_this)

    netscalefac = conf.op_net_inout_scale
    imnr_net, imnc_net = conf.op_imsz_net
    assert predhm_clip.shape[1] == imnr_net / netscalefac  # rhs should be integral
    assert predhm_clip.shape[2] == imnc_net / netscalefac  # "

    totscalefac = netscalefac * conf.rescale  # rescale to net input, then undo rescale
    predlocs_argmax_hires = PoseTools.unscale_points(predlocs_argmax, totscalefac, totscalefac)
    predlocs_wgtcnt_hires = PoseTools.unscale_points(predlocs_wgtcnt, totscalefac, totscalefac)
    pks_with_score[..., :2] = PoseTools.unscale_points(pks_with_score[..., :2], totscalefac, totscalefac)
    pks_with_score_cmpt[..., :2] = PoseTools.unscale_points(pks_with_score_cmpt[..., :2],
                                                            totscalefac, totscalefac)

    # undo padding
    predlocs_argmax_hires[..., 0] -= conf.op_im_padx // 2
    predlocs_argmax_hires[..., 1] -= conf.op_im_pady // 2
    predlocs_wgtcnt_hires[..., 0] -= conf.op_im_padx // 2
    predlocs_wgtcnt_hires[..., 1] -= conf.op_im_pady // 2
    # predlocs_wgtcnt0_hires[..., 0] -= conf.op_im_padx // 2
    # predlocs_wgtcnt0_hires[..., 1] -= conf.op_im_pady // 2
    pks_with_score[..., 0] -= conf.op_im_padx // 2
    pks_with_score[..., 1] -= conf.op_im_pady // 2
    pks_with_score_cmpt[..., 0] -= conf.op_im_padx // 2
    pks_with_score_cmpt[..., 1] -= conf.op_im_pady // 2

    # base_locs = np.array(all_infered)*conf.op_rescale
    # nanidx = np.isnan(base_locs)
    # base_locs[nanidx] = raw_locs[nanidx]
    ret_dict = {}

    if retrawpred:
        ret_dict['locs'] = predlocs_wgtcnt_hires
        ret_dict['locs_argmax'] = predlocs_argmax_hires
        # ret_dict['locs_wgtcnt0'] = predlocs_wgtcnt0_hires
        ret_dict['pred_hmaps'] = model_preds
        ret_dict['ims'] = ims
    else:
        # all return args should be [bsize x n_classes x ...]
        ret_dict['locs'] = predlocs_wgtcnt_hires
        ret_dict['locs_wgtcnt_hires'] = predlocs_wgtcnt_hires
        ret_dict['locs_argmax_hires'] = predlocs_argmax_hires
        ret_dict['conf'] = np.max(predhm, axis=(1, 2))
        ret_dict['num_hm_peaks'] = num_peaks
        ret_dict['pks_with_score'] = pks_with_score
        ret_dict['pks_with_score_cmpt'] = pks_with_score_cmpt

    return  ret_dict


def is_paf_conn(x_paf,y_paf,pt1,pt2,conf,thre_paf):
    vec = np.subtract(pt2[:2], pt1[:2])
    norm = np.linalg.norm(vec)
    mid_num = 8

    # failure case when 2 body parts overlaps
    if norm == 0:
        return False, 0
    if norm > max(conf.op_imsz_pad):
        return False, 0

    vec = np.divide(vec, norm)

    x_list = np.linspace(pt1[0], pt2[0], num=mid_num)
    y_list = np.linspace(pt1[1], pt2[1], num=mid_num)
    # list of mid_num (xm,ym) pts evenly spaced along line seg
    # from A to B

    # could interpolate paf vals if x_list, y_list are subpx/non-integral
    vec_x = np.array([x_paf[int(round(y_list[ss])), int(round(x_list[ss]))]  for ss in range(mid_num)])
    vec_y = np.array([y_paf[int(round(y_list[ss])), int(round(x_list[ss]))]  for ss in range(mid_num)])

    paf_scores = vec_x*vec[0] + vec_y*vec[1]
    scores_mean = sum(paf_scores) / len(paf_scores)
    # mean of dot(paf, vec) over mid_num pts -- "line integral"

    if norm < 3:
        # MK 20200803: pts that are very close, use small default values because PAFs would be weird.
        paf_scores = np.ones_like(paf_scores) * (thre_paf + 0.05)
        scores_mean = 0.05

    # average dot prod of paf along line seg
    cond1 = len(np.nonzero(paf_scores > thre_paf)[0]) > 0.6 * len(paf_scores)  # 60% of dot prods exceed thre_paf
    cond2 = scores_mean > 0
    return (cond1 and cond2), scores_mean

def do_inference(hmap, paf, conf,thre_hm,thre_paf):
    '''

    :param hmap: hmnr x hmnc x npt
    :param paf: hmnr x hmnc x nlimb
    :param conf:
    :param thre_hm: scalar float
    :param thre_paf: scalar float
    :return:
    '''

    all_preds = []
    af_graph = conf.op_affinity_graph

    # upscale fac from net output to padded raw image
    hmapscalefac = conf.op_net_inout_scale
    pafscalefac = hmapscalefac * (2**conf.op_hires_ndeconv)

    # work at the network input resolution
    hmap = cv2.resize(hmap, (0,0), fx=hmapscalefac, fy=hmapscalefac, interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (0,0), fx=pafscalefac, fy=pafscalefac, interpolation=cv2.INTER_CUBIC)

    npts = hmap.shape[-1]
    for part in range(npts):
        map_ori = hmap[:, :, part]
        map = map_ori
        peaks_with_score = heatmap.find_peaks(map, thre_hm)
        if len(peaks_with_score) ==0 and not conf.is_multi:
            ss = PoseTools.get_pred_locs(map[np.newaxis,:,:,np.newaxis]).tolist()[0][0]
            sc  = map[int(ss[1]),int(ss[0])]
            peaks_with_score = [[ss[0],ss[1],sc]]
        all_preds.append(peaks_with_score)

    # all_preds[part] is list of peaks (x, y, score)
    assert len(all_preds) == npts

    connection_all = []
    special_k = []

    n_edges = len(af_graph)
    assert paf.shape[-1] == n_edges*2
    score_to_pts = [] # keep track of all PAF scores and its info
    for k in range(n_edges):
        x_paf = paf[:,:,k*2]
        y_paf = paf[:,:,k*2+1]
        pt1s = all_preds[af_graph[k][0]]
        pt2s = all_preds[af_graph[k][1]]
        np1 = len(pt1s)
        np2 = len(pt2s)
        if np1 != 0 and np2 != 0:
            is_conn = []
            for i in range(np1):
                for j in range(np2):
                    conn, score_paf = is_paf_conn(x_paf,y_paf,pt1s[i],pt2s[j],conf,thre_paf)
                    if conn:
                        is_conn.append([i, j, score_paf, score_paf+pt1s[i][2] + pt2s[j][2]])
            # last entry is the total score

            # sort by tot_score
            is_conn = sorted(is_conn, key=lambda x: x[3], reverse=True)
            # cols are (i, j, paf_score)
            sel_conn = np.zeros((0, 3))
            for c in range(len(is_conn)):
                i, j, s, ts = is_conn[c]
                # each part candidate (eg i or j) can only participate in one row/connection.
                # greedy match, match highest scores first
                if i not in sel_conn[:, 0] and j not in sel_conn[:, 1]:
                    sel_conn = np.vstack([sel_conn, [i, j, ts]])
                    score_to_pts.append([ts,k,sel_conn.shape[0]-1])
                    if len(sel_conn) >= min(np1, np2):
                        break

            connection_all.append(sel_conn)
        else:
            special_k.append(k)
            connection_all.append([])

    assert len(connection_all) == n_edges

    targets = np.ones((0, npts)) * np.nan
    scores = np.zeros([0,2])
    # XXX what are the invariants for subset if any?

    score_to_pts = sorted(score_to_pts, key=lambda x:x[0], reverse=True)
    peaks_done = [[] for i in range(npts)]

    for cur_x in score_to_pts:
        k = cur_x[1]
        cndx = cur_x[2]
        i = int(connection_all[k][cndx][0])
        j = int(connection_all[k][cndx][1])
        p1,p2 = af_graph[k]

        if i in peaks_done[p1] and j in peaks_done[p2]:
            cur_t1 = np.where(targets[:,p1]==i)[0][0]
            cur_t2 = np.where(targets[:,p2]==j)[0][0]
            if cur_t1 == cur_t2:
                # Both belong to same target, nothing to do
                continue

            elif np.all(np.isnan(targets[cur_t1,:]) | np.isnan(targets[cur_t2,:])):
                # No overlap. Merge
                cur_t = min(cur_t1,cur_t2)
                targets[cur_t,:] = np.where(np.isnan(targets[cur_t1,:]),targets[cur_t2,:],targets[cur_t1,:])
                to_del = cur_t1 + cur_t2 - cur_t # this is smart,isn't it :)
                targets = np.delete(targets,to_del,0)

        elif i in peaks_done[p1]:
            cur_t = np.where(targets[:,p1]==i)[0]
            if not np.isnan(targets[cur_t,p2]):
                continue
            targets[cur_t,p2] = j
            peaks_done[p2].append(j)
        elif j in peaks_done[p2]:
            cur_t = np.where(targets[:,p2]==j)[0]
            if not np.isnan(targets[cur_t,p1]):
                continue
            targets[cur_t,p1] = i
            peaks_done[p1].append(i)
        else:
            # Both belong to none. So create new
            cur_r = np.ones([1,npts]) *np.nan
            cur_r[0,p1] = i
            cur_r[0,p2] = j
            peaks_done[p1].append(i)
            peaks_done[p2].append(j)
            targets = np.vstack([targets,cur_r])

    # delete if the less than 1/2 pts.
    deleteIdx = np.where(np.sum(~np.isnan(targets),axis=1)<npts/2)[0]
    targets = np.delete(targets, deleteIdx, axis=0)

    targets_locs = np.ones([conf.max_n_animals, npts, 2]) * np.nan
    if conf.is_multi:
        for n_out in range(min(targets.shape[0],conf.max_n_animals)):
            for p in range(npts):
                if np.isnan(targets[n_out,p]):
                    continue
                cur_i = int(targets[n_out,p])
                targets_locs[n_out,p,:] = all_preds[p][cur_i][:2]

    else:
        if targets.shape[0] != 1:
            # special case for single animal
            for p in range(npts):
                if len(all_preds[p]) < 1:
                    continue
                kk = sorted(all_preds[p], key=lambda x:x[2], reverse=True)
                targets_locs[0,p,:] = kk[0][:2]
        else:
            for p in range(npts):
                if np.isnan(targets[0,p]):
                    kk = sorted(all_preds[p], key=lambda x: x[2], reverse=True)
                    targets_locs[0, p, :] = kk[0][:2]
                else:
                    cur_i = int(targets[0,p])
                    targets_locs[0,p,:] = all_preds[p][cur_i][:2]

    return targets_locs

def do_inference_old(hmap, paf, conf,thre_hm,thre_paf):
    '''

    :param hmap: hmnr x hmnc x npt
    :param paf: hmnr x hmnc x nlimb
    :param conf:
    :param thre_hm: scalar float
    :param thre_paf: scalar float
    :return:
    '''

    all_peaks = []
    peak_counter = 0
    limb_seq = conf.op_affinity_graph

    # upscale fac from net output to padded raw image
    hmapscalefac = conf.op_net_inout_scale
    pafscalefac = hmapscalefac * (2**conf.op_hires_ndeconv)

    # work at raw input res
    hmap = cv2.resize(hmap, (0,0), fx=hmapscalefac, fy=hmapscalefac,
                      interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (0,0), fx=pafscalefac, fy=pafscalefac,
                     interpolation=cv2.INTER_CUBIC)

    npts = hmap.shape[-1]
    for part in range(npts):
        map_ori = hmap[:, :, part]
        map = map_ori
        # map = gaussian_filter(map_ori, sigma=3)

        peaks_with_score = heatmap.find_peaks(map, thre_hm)

        # if len(peaks_with_score)>2:
        #    peaks_with_score = sorted(peaks_with_score,key=lambda x:x[2],reverse=True)[:2]
        #    peaks = peaks_with_score #  taking the first two peaks
        id = range(peak_counter, peak_counter + len(peaks_with_score))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        # list of (x, y, score, id)

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks_with_score)

    # all_peaks[ipt] is list of (qualifying) hmap peaks found: (x, y, score, id)
    assert len(all_peaks) == npts

    connection_all = []
    special_k = []
    mid_num = 8

    nlimb = len(limb_seq)
    assert paf.shape[-1] == nlimb*2
    for k in range(nlimb):
        x_paf = paf[:,:,k*2]
        y_paf = paf[:,:,k*2+1]
        # score_mid = paf[:, :, [x for x in limb_seq[k]]]
        candA = all_peaks[limb_seq[k][0]]  # list of (x,y,score,id) for first pt in limb
        candB = all_peaks[limb_seq[k][1]]  # " second pt
        nA = len(candA)
        nB = len(candB)
        # indexA, indexB = limb_seq[k]
        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])  # (x,y) vec from A->B
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    if norm > max(conf.op_imsz_pad):
                        continue

                    # if limbSeq[k][0]==0 and limbSeq[k][1]==1 and norm < 150:
                    #   continue

                    vec = np.divide(vec, norm)  # now unit vec from A->B

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),  np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                    # list of mid_num (xm,ym) pts evenly spaced along line seg
                    # from A to B

                    vec_x = np.array(
                        [x_paf[int(round(startend[I][1])), int(round(startend[I][0]))] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [y_paf[int(round(startend[I][1])), int(round(startend[I][0]))] \
                         for I in range(len(startend))])
                    # np vectors of length mid_num containing paf (component) vals

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) # + min( 0.5 * oriImg.shape[0] / norm - 1, 0)

                    if norm < 3:
                        # MK 20200803: pts that are very close, use small default values because PAFs would be weird.
                        score_midpts = np.ones_like(score_midpts)*(thre_paf + 0.05)
                        score_with_dist_prior = 0.05

                    # average dot prod of paf along line seg
                    criterion1 = len(np.nonzero(score_midpts > thre_paf)[0]) > 0.8 * len(score_midpts)  # 80% of dot prods exceed thre_paf
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
            # connection_candidate is list of (i, j, average_paf_dotprod, totscore) of
            # accepted candidates where i/j index candA/B resp. connection_candidate could
            # be empty.
            #
            # scale of score_with_dist_prior is paf scale; candA/B scores are hmap scale
            # so adding them seems ok

            # currently sort by paf-dotprod only (instead of totscore)
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            # cols are (pkid_i, pkid_j, paf_dotprod, i, j)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                # each part candidate (eg i or j) can only participate in one row/connection.
                # greedy match, match highest scores first
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    assert len(connection_all) == nlimb
    # connection_all[k] is a connection array [nconn_cands x 5] where cols are
    #       (pkid_i, pkid_j, paf_dotprod, i, j)
    #   where i/j index all_peaks[limb_seq[k][0]]/all_peaks[limb_seq[k][1]], resp.
    #   and rows are avail connections for limb k, in ~ descending order of match quality
    #   connection_all[k] should not have any repeated indices in cols 0, 1, 3, 4
    #
    # connection_all[k] can be [0x5] or [] if no candidate connections; in the latter
    # case special_k will contain k

    # subset. rows are "people"
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    # For first n_classes els, subset[isub,ipt] is -1 if unset, otherwise is
    # pkID for that part
    subset = -1 * np.ones((0, conf.n_classes + 2))
    # XXX what are the invariants for subset if any?

    candidate = np.array([item for sublist in all_peaks for item in sublist])
    # candidate is [npeakstot x 4] array with cols (x, y, hmapscore, pkid)
    assert np.array_equal(candidate[:,-1], range(len(candidate)))

    for k in range(len(limb_seq)):
        if k not in special_k:  # connection_all[k] could still be empty [0x5]
            partAs = connection_all[k][:, 0]  # peakids
            partBs = connection_all[k][:, 1]  # peakids
            assert len(np.unique(partAs)) == len(partAs)
            assert len(np.unique(partBs)) == len(partBs)
            indexA, indexB = np.array(limb_seq[k])

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                # (loop over)/check all avail conns for this limb

                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    # found person where one part already matches. fill in other part
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
                    # membership[ipt] is number of people in which landmark ipt is assigned:
                    # can be 0, 1, or 2
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge; no overlap
                        subset[j1][:-2] += (subset[j2][:-2] + 1)  # tricky
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        # XXX dont understand why this unilaterally assigns partBs[i] to j1
                        # and leaves j2 unchanged; and doesnt consider partAs[i]
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found: # and k < 2:
                    # XXX what if k>=2!?!
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
            a = int(all_peaks[i][j][0])  # x
            b = int(all_peaks[i][j][1])  # y
            c = all_peaks[i][j][2]  # hmap score
            detections.append((a, b, c))
            parts[i].append((a, b, c))
    #        cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    # stickwidth = 10 #4
    # print()
    mappings = np.ones([conf.max_n_animals,conf.n_classes,2])*np.nan
    if subset.shape[0] < 1:
        return mappings

    subset = subset[np.argsort(subset[:,-2])] # sort by highest scoring one in ASCENDING order
    for n in range(subset.shape[0]):
        for i in range(conf.n_classes):
            index = subset[-n-1][i]
            if index < 0:
                mappings[n,i,:] = np.nan
            else:
                mappings[n,i,:] = candidate[index.astype(int), :2]
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
            # 360, 1)
            # cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            # canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    # XXXX commented out but check this
    # mappings -= conf.op_label_scale/4 # For some reason,
    # cv2.imresize maps (x,y) to (8*x+4,8*y+4). sigh.
    # XXX NOTE this output is returned relative to padded im
    return mappings

def model_files(conf, name):
    latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    if latest_model_file is None:
        return None
    traindata_file = PoseTools.get_train_data_file(conf, name)
    return [latest_model_file, traindata_file + '.json']

def pafhm_prog_viz(pafhm,
                   theta_mag=False,
                   ilimb=0,
                   ibatch=0,
                   figsz=(1400,1800),
                   figfaceclr=(0.5, 0.5, 0.5)):
    '''

    :param pafhm: list of PAF heatmaps
    :param theta_mag: if True, plot theta/magnitude instead of x/y
    :param ilimb: limb index
    :param ibatch: batch index to show
    :param figsz:
    :param figfaceclr:
    :return:
    '''

    import matplotlib.pyplot as plt

    nstg = len(pafhm)

    f, ax = plt.subplots(2, nstg, squeeze=False)
    m = plt.get_current_fig_manager()
    m.resize(*figsz)
    f.set_facecolor(figfaceclr)

    for istg in range(nstg):
        ilimbx = 2*ilimb
        ilimby = 2*ilimb+1
        hmx = pafhm[istg][ibatch, :, :, ilimbx]
        hmy = pafhm[istg][ibatch, :, :, ilimby]
        if theta_mag:
            hm1 = np.arctan2(hmy, hmx)
            hm1 = hm1/np.pi*180.
            hm2 = np.sqrt(hmx**2 + hmy**2)
        else:
            hm1 = hmx
            hm2 = hmy

        plt.axes(ax[0, istg])
        plt.cla()
        plt.imshow(hm1)
        plt.colorbar()
        plt.axes(ax[1, istg])
        plt.cla()
        plt.imshow(hm2)
        plt.colorbar()
