from __future__ import print_function
from __future__ import division

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation, Input, Lambda, PReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import random_normal, constant
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.callbacks import Callback
import imagenet_resnet
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras import backend as K
#from keras.legacy import interfaces

import sys
import pickle
import math
import PoseTools
import os
import  numpy as np
import json
import tensorflow as tf
import tensorflow.keras.backend as K
import logging
from time import time

import tfdatagen as opdata
import heatmap
import open_pose4 as op
import util

ISPY3 = sys.version_info >= (3, 0)


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

    print("upsample initializer desired shape, type: {}, {}".format(shape,dtype))
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
    return K.variable(value=xinit, dtype=dtype)
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
        kernel_reg = l2(wd[0]) if wd else None
        bias_reg = l2(wd[1]) if wd else None
        logging.info("Deconv: regularization around zero with weights {}".format(wd))
    elif wdmode == 1:  # 'aroundinit'
        kshape = (ks, ks, nf, nf)
        kinit = upsample_init_value(kshape, 'bl')
        kernel_reg = make_kernel_regularizer(kinit, wd[0])
        bias_reg = l2(wd[1]) if wd else None
        logging.info("Deconv: regularization around init with weights {}".format(wd))
    else:
        assert False

    x = Conv2DTranspose(nf, (ks, ks), strides=2,
                        padding='same', name=name,
                        kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg,
                        kernel_initializer=upsamp_init_bl,
                        bias_initializer=constant(0.0))(x)
    logging.info("Using 2xdeconv w/init around upsample, wdmode={}, wd={}.".format(wdmode, wd))

    return x

def get_training_model(imszuse,
                       wd_kernel,
                       nDC=3,
                       dc_num_filt=256,
                       npts=19,
                       backbone='ResNet50_8px',
                       backbone_weights=None,
                       upsamp_chan_handling='direct_deconv'):

    '''

    :param imszuse: (imnr, imnc) image size (typically adjusted/padded) for input into net
    :param wd_kernel: weight decay for l2 reg (applied only to weights not biases)
    :param npts:
    :return: Model.
        Inputs: [img]
        Outputs: [hmap (res determined by backbone+nDC)]
    '''

    imnruse, imncuse = imszuse
    assert imnruse % 32 == 0, "Image size must be divisible by 32"
    assert imncuse % 32 == 0, "Image size must be divisible by 32"

    img_input = Input(shape=imszuse + (3,), name='input_img')
    inputs = [img_input, ]

    img_normalized = Lambda(lambda z: z / 256. - 0.5)(img_input)  # [-0.5, 0.5] Isn't this really [-0.5, 0.496]
    # sub mean?

    # backbone
    bb_model_fcn = getattr(imagenet_resnet, backbone)
    bb_model = bb_model_fcn(include_top=False,
                            weights=backbone_weights,
                            input_tensor=img_normalized,
                            pooling=None)
    backboneF = bb_model.output
    logging.info("BackBone {} with weights {} instantiated; output shape is {}".format(
        backbone, backbone_weights, backboneF.shape.as_list()[1:]))

    x = backboneF

    DCFILTSZ = 4
    if upsamp_chan_handling == 'reduce_first':
        REDUCEFILTSZ = 1
        x = conv(x, dc_num_filt, REDUCEFILTSZ, 'bb_reduce', (wd_kernel, 0.0))
        for iDC in range(nDC):
            dcname = "dc-{}".format(iDC)
            x = deconv_2x_upsampleinit(x, dc_num_filt, DCFILTSZ, dcname, None, 0)
            x = prelu(x, "{}-prelu".format(dcname))

        logging.info("Deconvs. Reduce first with filtsz={}. Added {} deconvs with filtsz={}, nfilt={}".format(REDUCEFILTSZ, nDC, DCFILTSZ, dc_num_filt))
    elif upsamp_chan_handling == 'direct_deconv':
        for iDC in range(nDC):
            init_zero = tf.keras.initializers.Zeros()
            init_trunc_norm = \
                tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
            dcname = "dc-{}".format(iDC)
            x = Conv2DTranspose(dc_num_filt, (DCFILTSZ, DCFILTSZ),
                                strides=2,
                                padding='same',
                                name=dcname,
                                kernel_initializer=init_trunc_norm,
                                bias_initializer=init_zero)(x)
            x = prelu(x, "{}-prelu".format(dcname))

        logging.info("Deconvs. direct deconv. Added {} deconvs with filtsz={}, nfilt={}".format(nDC, DCFILTSZ, dc_num_filt))
    else:
         assert False

    #nfilt = x.shape.as_list()[-1]
    #x = conv(x, npts nf1by1, 1, "{}-stg{}-1by1-1".format(stagety, stageidx), (wd_kernel, 0))
    #x = prelu(x, "{}-stg{}-postDC-1by1-1-prelu".format(stagety, stageidx))

    x = Conv2D(npts, (1, 1),
               padding='same',
               name="out-1by1",
               kernel_regularizer=l2(wd_kernel),
               bias_regularizer=None,
               kernel_initializer=tf.keras.initializers.VarianceScaling(),
               bias_initializer=constant(0.0))(x)
    #x = conv(x, npts, 1, "out-1by1", (wd_kernel, 0.0))
    x = prelu(x, "out-1by1-prelu")

    outputs = [x,]

    model = Model(inputs=inputs, outputs=outputs)
    return model

def configure_losses(model, bsize):
    '''
    
    :param model: 
    :param bsize: 

    :return: losses, loss_weights. both dicts whose keys are .names of model.outputs
    '''

    def eucl_loss(x, y):
        return K.sum(K.square(x - y)) / bsize / 2.0  # not sure why norm by bsize nec

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
        loss_weights[key] = 1.0
        loss_weights_vec.append(loss_weights[key])

    return losses, loss_weights, loss_weights_vec

def get_testing_model(imszuse,
                      nDC=3,
                      dc_num_filt=256,
                      npts=19,
                      backbone='Resnet50_8px',
                      upsamp_chan_handling='direct_deconv'):
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
    assert imnruse % 32 == 0, "Image size must be divisible by 32"
    assert imncuse % 32 == 0, "Image size must be divisible by 32"

    img_input = Input(shape=imszuse + (3,), name='input_img')
    inputs = [img_input, ]

    img_normalized = Lambda(lambda x: x / 256. - 0.5)(img_input) # [-0.5, 0.5]

    # backbone
    bb_model_fcn = getattr(imagenet_resnet, backbone)
    bb_model = bb_model_fcn(include_top=False,
                            weights=None,
                            input_tensor=img_normalized,
                            pooling=None)
    backboneF = bb_model.output
    print("BackBone {} instantiated; output shape is {}".format(
        backbone, backboneF.shape.as_list()[1:]))

    x = backboneF

    DCFILTSZ = 4
    if upsamp_chan_handling == 'reduce_first':
        REDUCEFILTSZ = 1
        x = conv(x, dc_num_filt, REDUCEFILTSZ, 'bb_reduce', None)
        for iDC in range(nDC):
            dcname = "dc-{}".format(iDC)
            x = deconv_2x_upsampleinit(x, dc_num_filt, DCFILTSZ, dcname, None, 0)
            x = prelu(x, "{}-prelu".format(dcname))

        logging.info("Deconvs. Reduce first with filtsz={}. Added {} deconvs with filtsz={}, nfilt={}".format(REDUCEFILTSZ, nDC, DCFILTSZ, dc_num_filt))

    elif upsamp_chan_handling == 'direct_deconv':
        for iDC in range(nDC):
            dcname = "dc-{}".format(iDC)
            x = Conv2DTranspose(dc_num_filt, (DCFILTSZ, DCFILTSZ),
                                strides=2,
                                padding='same',
                                name=dcname)(x)
            x = prelu(x, "{}-prelu".format(dcname))

        logging.info("Deconvs. direct deconv. Added {} deconvs with filtsz={}, nfilt={}".format(nDC, DCFILTSZ, dc_num_filt))
    else:
        assert False

    #nfilt = x.shape.as_list()[-1]
    #x = conv(x, npts nf1by1, 1, "{}-stg{}-1by1-1".format(stagety, stageidx), (wd_kernel, 0))
    #x = prelu(x, "{}-stg{}-postDC-1by1-1-prelu".format(stagety, stageidx))

    x = Conv2D(npts, (1, 1),
               padding='same',
               name="out-1by1")(x)
    # x = conv(x, npts, 1, "out-1by1", (wd_kernel, 0.0))
    x = prelu(x, "out-1by1-prelu")

    outputs = [x,]

    model = Model(inputs=inputs, outputs=outputs)
    return model




# ---------------------
# -- Training ---------
#----------------------

'''
def set_openpose_defaults(conf):
    conf.label_blur_rad = 5
    conf.rrange = 5
    conf.display_step = 50 # this is same as batches per epoch
    conf.dl_steps = 600000
    conf.batch_size = 10
    conf.n_steps = 4.41
    conf.gamma = 0.333
'''

'''
def get_im_pad(sz, dimname):
    BASE = 32
    szmod = sz % BASE
    if szmod == 0:
        pad = 0
        szuse = sz
    else:
        pad = BASE - szmod
        szuse = sz + pad
        warnstr = 'Image {} dimension ({}) is not a multiple of {}. Image will be padded'.format(dimname, sz, BASE)
        logging.warning(warnstr)

    return pad, szuse
'''

def compute_padding_imsz_net(imsz, rescale, n_transition_max):
    '''
    From the raw image size, desired rescale, and desired n_transition_min,
    compute the necessary padding and resulting imsz_net (input-to-network-size)

    n_transition_min is a term from from dpk but here it is just the downsample scale of
    network output:input (as an exponent with base 2)

    :param imsz: [2] raw im size
    :param rescale: float, desired rescale.
    :param n_transition_max: log_2(network_input_sz/network_output_sz)
    :return: padx, pady, imsz_pad, imsz_net, imsz_net_out_min_supported
    '''

    # in tfdatagen, the input pipeline is read->pad->rescale/distort->ready_for_network

    # we set the padding so the rescale is 'perfect' ie the desired rescale is the one precisely
    # used ie the imsz-after-pad is precisely divisible by rescale

    assert isinstance(rescale, int) or rescale.is_integer(), "Expect rescale to be integral value"

    net_in_out_ratio_max = 2 ** n_transition_max
    imsz_pad_should_be_divisible_by = int(rescale * net_in_out_ratio_max)
    dsfac = imsz_pad_should_be_divisible_by
    roundupeven = lambda x: int(np.ceil(x/dsfac)) * dsfac

    imsz_pad = (roundupeven(imsz[0]), roundupeven(imsz[1]))
    padx = imsz_pad[1] - imsz[1]
    pady = imsz_pad[0] - imsz[0]
    imsz_net = (int(imsz_pad[0]/rescale), int(imsz_pad[1]/rescale))
    imsz_net_out_min_supported = (int(imsz_pad[0]/rescale/net_in_out_ratio_max),
                                  int(imsz_pad[1]/rescale/net_in_out_ratio_max))

    return padx, pady, imsz_pad, imsz_net, imsz_net_out_min_supported

def update_conf(conf):
    '''
    Update conf in-place
    :param conf:
    :return:
    '''

    conf.sb_im_padx, conf.sb_im_pady, conf.sb_imsz_pad, conf.sb_imsz_net, _ = \
        compute_padding_imsz_net(conf.imsz, conf.rescale, conf.sb_n_transition_supported)
    logging.info("SB size stuff: imsz={}, imsz_pad={}, imsz_net={}, rescale={}".format(
        conf.imsz, conf.sb_imsz_pad, conf.sb_imsz_net, conf.rescale))


def dot(K, L):
   assert len(K) == len(L), 'lens do not match: {} vs {}'.format(len(K), len(L))
   return sum(i[0] * i[1] for i in zip(K, L))

def get_output_scale(model):
    # returns downsample scale (always >= 1; input assumed to be larger than out)
    innr, innc = model.input.shape.as_list()[1:3]
    otnr, otnc = model.output.shape.as_list()[1:3]

    assert innr >= otnr and innc >= otnc  # in theory doesnt need to hold
    assert innr % otnr == innc % otnc == 0
    scale = innr//otnr
    assert innc//otnc == scale

    return scale

def training(conf, name='deepnet'):

    base_lr = conf.sb_base_lr * conf.get('learning_rate_multiplier',1.)
    batch_size = conf.batch_size
    gamma = conf.gamma
    stepsize = int(conf.decay_steps)
    # after each stepsize iterations update learning rate: lr=lr*gamma
    iterations_per_epoch = conf.display_step
    max_iter = math.ceil(conf.dl_steps/iterations_per_epoch)
    last_epoch = 0

    #(imnr, imnc) = conf.imsz
    #conf.sb_im_pady, imnr_use = get_im_pad(imnr, 'row')
    #conf.sb_im_padx, imnc_use = get_im_pad(imnc, 'column')

    assert conf.dl_steps % iterations_per_epoch == 0, 'dl steps must be a multiple of display steps'
    assert conf.save_step % iterations_per_epoch == 0, ' save steps must be a multiple of display steps'

    # need this to set default
    # _ = conf.get('save_time', None)

    train_data_file = os.path.join(conf.cachedir, 'traindata')
    with open(train_data_file, 'wb') as td_file:
        pickle.dump(conf, td_file, protocol=2)
    logging.info('Saved config to {}'.format(train_data_file))

    #model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{epoch:d}')
    assert not conf.normalize_img_mean, "SB currently performs its own img input norm"
    assert not conf.normalize_batch_mean, "SB currently performs its own img input norm"
    model = get_training_model(conf.sb_imsz_net,
                               conf.sb_weight_decay_kernel,
                               nDC=conf.sb_num_deconv,
                               dc_num_filt=conf.sb_deconv_num_filt,
                               npts=conf.n_classes,
                               backbone=conf.sb_backbone,
                               backbone_weights=conf.sb_backbone_weights,
                               upsamp_chan_handling=conf.sb_upsamp_chan_handling)
    conf.sb_output_scale = get_output_scale(model)
    conf.sb_blur_rad_output_res = \
        max(1.0, conf.sb_blur_rad_input_res / float(conf.sb_output_scale))
    logging.info('Model output scale is {}, blurrad_input/output is {}/{}'.format(conf.sb_output_scale, conf.sb_blur_rad_input_res, conf.sb_blur_rad_output_res))

    # prepare generators
    PREPROCFN = 'ims_locs_preprocess_sb'
    trntfr = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
    train_di = opdata.make_data_generator(trntfr, conf, True, True, PREPROCFN)
    train_di2 = opdata.make_data_generator(trntfr, conf, True, True, PREPROCFN)
    val_di = opdata.make_data_generator(trntfr, conf, False, False, PREPROCFN)

    losses, loss_weights, loss_weights_vec = configure_losses(model, batch_size)

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
            val_x, val_y = next(self.val_di)
            val_out = self.model.predict(val_x, batch_size=batch_size)
            val_out = [val_out,] # single output apparently not in list
            val_loss_full = self.model.evaluate(val_x, val_y, batch_size=batch_size, verbose=0)
            #val_loss_K = val_loss_full  # scalar loss  # want Py 3 unpack
            #val_loss_full = val_loss_full[1:]
            #val_loss = dot(val_loss_full, loss_weights_vec)
            #val_loss = np.nan
            train_x, train_y = next(self.train_di)
            train_out = self.model.predict(train_x, batch_size=batch_size)
            train_out = [train_out,]
            train_loss_full = self.model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
            #train_loss_K = train_loss_full[0]  # want Py 3 unpack
            #train_loss_full = train_loss_full[1:]
            #train_loss = dot(train_loss_full, loss_weights_vec)
            #train_loss = np.nan
            lr = K.eval(self.model.optimizer.lr)

            # dist only for last MAP layer (will be hi-res if deconv is on)
            predhmval = val_out[-1]
            predhmval = clip_heatmap_with_warn(predhmval)
            # (bsize, npts, 2), (x,y), 0-based
            predlocsval, _ = heatmap.get_weighted_centroids_with_argmax(predhmval,
                                                                        floor=self.config.sb_hmpp_floor,
                                                                        nclustermax=self.config.sb_hmpp_nclustermax)
            gtlocs, _ = heatmap.get_weighted_centroids_with_argmax(val_y[-1],
                                                                   floor=self.config.sb_hmpp_floor,
                                                                   nclustermax=self.config.sb_hmpp_nclustermax)
            tt1 = predlocsval - gtlocs  # computed in the output res/space
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))  # [bsize x ncls]
            val_dist = np.nanmean(tt1)  # av over all batches/pts; in output res/space

            # NOTE train_dist uses argmax
            tt1 = PoseTools.get_pred_locs(train_out[-1]) - \
                  PoseTools.get_pred_locs(train_y[-1])
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            train_dist = np.nanmean(tt1)  # etc
            self.train_info['val_dist'].append(val_dist)
            self.train_info['val_loss'].append(0.0)
            self.train_info['val_loss_K'].append(0.0)
            self.train_info['val_loss_full'].append(val_loss_full)
            self.train_info['train_dist'].append(train_dist)
            self.train_info['train_loss'].append(0.0)
            self.train_info['train_loss_K'].append(0.0)
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

    # Epsilon: could just leave un-speced, None leads to default in tf1.14 at least
    # Decay: 0.0 bc lr schedule handled above by callback/LRScheduler
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

def get_pred_fn(conf, model_file=None, name='deepnet', retrawpred=False, **kwargs):
    #(imnr, imnc) = conf.imsz
    #conf.sb_im_pady, imnr_use = get_im_pad(imnr, 'row') # xxxxxxx
    #conf.sb_im_padx, imnc_use = get_im_pad(imnc, 'column')
    #imszuse = (imnr_use, imnc_use)
    #conf.imszuse = imszuse
    #logging.info('pady/padx = {}/{}, imszuse = {}'.format(
    #    conf.sb_im_pady, conf.sb_im_padx, imszuse))

    assert not conf.normalize_img_mean, "SB currently performs its own img input norm"
    assert not conf.normalize_batch_mean, "SB currently performs its own img input norm"
    model = get_testing_model(conf.sb_imsz_net,
                              nDC=conf.sb_num_deconv,
                              dc_num_filt=conf.sb_deconv_num_filt,
                              npts=conf.n_classes,
                              backbone=conf.sb_backbone,
                              upsamp_chan_handling=conf.sb_upsamp_chan_handling)
    conf.sb_output_scale = get_output_scale(model)
    conf.sb_blur_rad_output_res = \
        max(1.0, conf.sb_blur_rad_input_res / float(conf.sb_output_scale))
    logging.info('Model output scale is {}, blurrad_input/output is {}/{}'.format(conf.sb_output_scale, conf.sb_blur_rad_input_res, conf.sb_blur_rad_output_res))

    op.compare_conf_traindata(conf, name)

    # TODO: Theoretically probably should deep-copy conf since it is used in the returned fn

    if model_file is None:
        latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    else:
        latest_model_file = model_file
    logging.info("Loading the weights from {}.. ".format(latest_model_file))
    model.load_weights(latest_model_file)
    # thre1 = conf.get('op_param_hmap_thres',0.1)
    # thre2 = conf.get('op_param_paf_thres',0.05)

    def pred_fn(all_f, retrawpred=retrawpred):
        '''

        :param all_f: raw images NHWC
        :return:
        '''

        assert all_f.shape[0] == conf.batch_size
        locs_sz = (conf.batch_size, conf.n_classes, 2)
        locs_dummy = np.zeros(locs_sz)
        ims, _ = opdata.ims_locs_preprocess_sb(all_f, locs_dummy, conf, False,
                                               gen_target_hmaps=False)
        predhm = model.predict(ims)  # model with single output apparently not a list

        # all_infered = []
        # for ex in range(xs.shape[0]):
        #     infered = do_inference(model_preds[-1][ex,...],model_preds[-2][ex,...],conf, thre1, thre2)
        #     all_infered.append(infered)
        predhm_clip = clip_heatmap_with_warn(predhm)

        # (bsize, npts, 2), (x,y), 0-based
        #predlocs_argmax = PoseTools.get_pred_locs(predhm)
        predlocs_wgtcnt, predlocs_argmax = \
            heatmap.get_weighted_centroids_with_argmax(predhm_clip,
                                                       floor=conf.sb_hmpp_floor,
                                                       nclustermax=conf.sb_hmpp_nclustermax)
        #predlocs_wgtcnt0 = heatmap.get_weighted_centroids(predhm_clip,
        #                                                  floor=0,
        #                                                  nclustermax=1)
        assert predlocs_argmax.shape == locs_sz
        assert predlocs_wgtcnt.shape == locs_sz
        print("HMAP POSTPROC, floor={}, nclustermax={}".format(conf.sb_hmpp_floor, conf.sb_hmpp_nclustermax))

        netscalefac = conf.sb_output_scale
        imnr_net, imnc_net = conf.sb_imsz_net
        assert predhm_clip.shape[1] == imnr_net / netscalefac
        assert predhm_clip.shape[2] == imnc_net / netscalefac
        #predlocs_argmax_hires = PoseTools.unscale_points(predlocs_argmax, scalefac, scalefac)
        #predlocs_wgtcnt_hires = PoseTools.unscale_points(predlocs_wgtcnt, scalefac, scalefac)

        totscalefac = netscalefac * conf.rescale
        predlocs_argmax_hires = PoseTools.unscale_points(predlocs_argmax, totscalefac, totscalefac)
        predlocs_wgtcnt_hires = PoseTools.unscale_points(predlocs_wgtcnt, totscalefac, totscalefac)
        #predlocs_wgtcnt0_hires = PoseTools.unscale_points(predlocs_wgtcnt0, totscalefac, totscalefac)

        # undo padding
        predlocs_argmax_hires[..., 0] -= conf.sb_im_padx//2
        predlocs_argmax_hires[..., 1] -= conf.sb_im_pady//2
        predlocs_wgtcnt_hires[..., 0] -= conf.sb_im_padx//2
        predlocs_wgtcnt_hires[..., 1] -= conf.sb_im_pady//2
        #predlocs_wgtcnt0_hires[..., 0] -= conf.sb_im_padx//2
        #predlocs_wgtcnt0_hires[..., 1] -= conf.sb_im_pady//2

        ret_dict = {}
        if retrawpred:
            ret_dict['locs'] = predlocs_wgtcnt_hires
            ret_dict['locs_argmax'] = predlocs_argmax_hires
            #ret_dict['locs_wgtcnt0'] = predlocs_wgtcnt0_hires
            ret_dict['pred_hmaps'] = predhm_clip
            ret_dict['ims'] = ims
        else:
            # all return args should be [bsize x n_classes x ...]
            ret_dict['locs'] = predlocs_wgtcnt_hires
            ret_dict['locs_mdn'] = predlocs_argmax_hires  # XXX hack for now
            ret_dict['locs_unet'] = predlocs_argmax_hires  # XXX hack for now
            ret_dict['conf'] = np.max(predhm, axis=(1, 2))
            ret_dict['conf_unet'] = np.max(predhm, axis=(1, 2))  # XXX hack

        return ret_dict

    def close_fn():
        K.clear_session()

    return pred_fn, close_fn, latest_model_file

def model_files(conf, name):
    latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    if latest_model_file is None:
        return None
    traindata_file = PoseTools.get_train_data_file(conf, name)
    return [latest_model_file, traindata_file + '.json']
