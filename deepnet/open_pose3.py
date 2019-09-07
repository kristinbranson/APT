from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
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

import open_pose2 as op2

ISPY3 = sys.version_info >= (3, 0)

def relu(x): return Activation('relu')(x)

def prelu(x):
    return keras.layers.PReLU(shared_axes=[1, 2])(x)

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
    x1 = conv(x0, nf, 3, "convblock_{}_{}".format(namebase, 1), (wd_kernel, 0))
    x1 = prelu(x1)
    x2 = conv(x1, nf, 3, "convblock_{}_{}".format(namebase, 2), (wd_kernel, 0))
    x2 = prelu(x2)
    x3 = conv(x2, nf, 3, "convblock_{}_{}".format(namebase, 3), (wd_kernel, 0))
    x3 = prelu(x3)
    x = Concatenate()([x1, x2, x3])
    return x

def stageCNN(x, nfout, stagety, stageidx, wd_kernel,
             nfconvblock=128, nconvblocks=5, nf1by1=128):
    # stagety: 'map' or 'paf'

    for iCB in range(nconvblocks):
        namebase = "{}_stg{}_cb{}".format(stagety, stageidx, iCB)
        x = convblock(x, nfconvblock, namebase, wd_kernel)
    x = conv(x, nf1by1, 1, "{}_stg{}_1by1_1".format(stagety, stageidx), (wd_kernel, 0))
    x = prelu(x)
    x = conv(x, nfout, 1, "{}_stg{}_1by1_2".format(stagety, stageidx), (wd_kernel, 0))
    x = prelu(x)
    return x

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

def apply_mask(x, mask, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    w = Multiply(name=w_name)([x, mask])  # vec_weight
    return w

def get_training_model(imszuse, wd_kernel, nPAFstg=5, nMAPstg=1, nlimbsT2=38, npts=19):
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

    img_input_shape = imszuse + (3,)
    paf_input_shape_hires = imszuse + (nlimbsT2,)
    map_input_shape_hires = imszuse + (npts,)

    imszvgg = (imnruse/8, imncuse/8)  # imsz post VGG ftrs
    paf_input_shape = imszvgg + (nlimbsT2,)
    map_input_shape = imszvgg + (npts,)

    inputs = []

    img_input = Input(shape=img_input_shape, name='input_img')
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
    assert vggF.shape.as_list()[1:] == imszvgg + (128,)

    # PAF 1..nPAFstg
    xpaflist = []
    xstagein = vggF
    for iPAFstg in range(nPAFstg):
        xstageout = stageCNN(xstagein, nlimbsT2, 'paf', iPAFstg, wd_kernel)
        xpaflist.append(xstageout)
        xstagein = Concatenate()([vggF, xstageout])

    # MAP
    xmaplist = []
    for iMAPstg in range(nMAPstg):
        xstageout = stageCNN(xstagein, npts, 'map', iMAPstg, wd_kernel)
        xmaplist.append(xstageout)
        xstagein = Concatenate()[vggF, xpaflist[-1], xstageout]

    assert len(xpaflist) == nPAFstg
    assert len(xmaplist) == nMAPstg

    outputs = xpaflist + xmaplist

    # w1 = apply_mask(stage1_branch1_out, paf_weight_input, 1, 1)
    # w2 = apply_mask(stage1_branch2_out, map_weight_input, 1, 2)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_testing_model(imszuse, nlimb=38, npts=19, fullpred=False):
    stages = 6

    imnruse, imncuse = imszuse
    assert imnruse % 8 == 0, "Image size must be divisible by 8"
    assert imncuse % 8 == 0, "Image size must be divisible by 8"
    img_input_shape = imszuse + (3,)

    img_input = Input(shape=img_input_shape, name='input_img')

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    outputsfull = []

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, nlimb, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, npts, 2, None)

    outputsfull.append(stage1_branch1_out)
    outputsfull.append(stage1_branch2_out)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages):
        stageT_branch1_out = stageT_block(x, nlimb, sn, 1, None)
        stageT_branch2_out = stageT_block(x, npts, sn, 2, None)
        outputsfull.append(stageT_branch1_out)
        outputsfull.append(stageT_branch2_out)
        x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    # stage sn=stages
    # AL: passing None into weight_decay(s) doesn't make sense; since it's the test model it's prob ok
    stageT_branch1_out = stageTdeconv_block(x, nlimb, stages, 1, None, None, 0)
    stageT_branch2_out = stageTdeconv_block(x, npts, stages, 2, None, None, 0)

    outputsfull.append(stageT_branch1_out)
    outputsfull.append(stageT_branch2_out)

    if fullpred:
        model = Model(inputs=[img_input], outputs=outputsfull)
    else:
        model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model


# ---------------------
# -- Data Generator ---
#----------------------


def create_affinity_labels(locs, imsz, graph, scale=1):
    """
    Create/return part affinity fields

    locs: (nbatch x npts x 2)
    imsz: (nr, nc) size of affinity maps to create/return
    graph: (nlimb) array of 2-element tuples; connectivity/skeleton
    scale: width of "tube" around limb.

    returns (nbatch x imsz[0] x imsz[1] x nlimb*2) paf hmaps.
        4th dim ordering: limb1x, limb1y, limb2x, limb2y, ...
    """

    n_out = len(graph)
    n_ex = locs.shape[0]
    out = np.zeros([n_ex,imsz[0],imsz[1],n_out*2])
    n_steps = 2*max(imsz)

    for cur in range(n_ex):
        for ndx, e in enumerate(graph):
            start_x, start_y = locs[cur,e[0],:]
            end_x, end_y = locs[cur,e[1],:]
            ll = np.sqrt( (start_x-end_x)**2 + (start_y-end_y)**2)

            if ll==0:
                # Can occur if start/end labels identical
                # Don't update out/PAF
                continue

            dx = (end_x - start_x)/ll/2
            dy = (end_y - start_y)/ll/2
            zz = None
            for delta in np.arange(-scale,scale,0.25):  # delta indicates perpendicular displacement from line/limb segment (in px)
                # xx = np.round(np.linspace(start_x,end_x,6000))
                # yy = np.round(np.linspace(start_y,end_y,6000))
                # zz = np.stack([xx,yy])
                xx = np.round(np.linspace(start_x+delta*dy,end_x+delta*dy,n_steps))
                yy = np.round(np.linspace(start_y-delta*dx,end_y-delta*dx,n_steps))
                if zz is None:
                    zz = np.stack([xx,yy])
                else:
                    zz = np.concatenate([zz,np.stack([xx,yy])],axis=1)
                # xx = np.round(np.linspace(start_x-dy,end_x-dy,6000))
                # yy = np.round(np.linspace(start_y+dx,end_y+dx,6000))
                # zz = np.concatenate([zz,np.stack([xx,yy])],axis=1)
            # zz now has all the pixels that are along the line.
            # or "tube" of width scale around limb
            zz = np.unique(zz,axis=1)
            # zz now has all the unique pixels that are along the line with thickness==scale.
            dx = (end_x - start_x) / ll
            dy = (end_y - start_y) / ll
            for x,y in zz.T:
                xint = int(round(x))
                yint = int(round(y))
                if xint < 0 or xint >= out.shape[2] or yint < 0 or yint >= out.shape[1]:
                    continue
                out[cur,yint,xint,ndx*2] = dx
                out[cur,yint,xint,ndx*2+1] = dy

    return out

def create_label_images(locs, imsz, scale=1):
    """
    Create/return target hmap for parts

    This is a 2d isotropic gaussian with sigma=scale with tails clipped to 0. everywhere below 0.05

    hmap min is 0., max is 1.

    locs: (nbatch x npts x 2) part locs
    """

    n_out = locs.shape[1]
    n_ex = locs.shape[0]
    out = np.zeros([n_ex,imsz[0],imsz[1],n_out])
    for cur in range(n_ex):
        for ndx in range(n_out):
            x,y = np.meshgrid(range(imsz[1]),range(imsz[0]))
            x = x - locs[cur,ndx,0]
            y = y - locs[cur,ndx,1]
            dd = np.sqrt(x**2+y**2)
            out[cur,:,:,ndx] = stats.norm.pdf(dd,scale=scale)/stats.norm.pdf(0,scale=scale)
    out[out<0.05] = 0.
    return out

class DataIteratorTF(object):


    def __init__(self, conf, db_type, distort, shuffle):
        self.conf = conf
        if db_type == 'val':
            filename = os.path.join(self.conf.cachedir, self.conf.valfilename) + '.tfrecords'
        elif db_type == 'train':
            filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + '.tfrecords'
        else:
            raise IOError('Unspecified DB Type') # KB 20190424 - py3
        self.file = filename
        self.iterator = None
        self.distort = distort
        self.shuffle = shuffle
        self.batch_size = self.conf.batch_size
        self.vec_num = len(conf.op_affinity_graph)
        self.heat_num = self.conf.n_classes
        self.N = PoseTools.count_records(filename)


    def reset(self):
        if self.iterator:
            self.iterator.close()
        self.iterator = tf.python_io.tf_record_iterator(self.file)
        # print('========= Resetting ==========')


    def read_next(self):
        if not self.iterator:
            self.iterator = tf.python_io.tf_record_iterator(self.file)
        try:
            if ISPY3:
                record = next(self.iterator)
            else:
                record = self.iterator.next()
        except StopIteration:
            self.reset()
            if ISPY3:
                record = next(self.iterator)
            else:
                record = self.iterator.next()

        return record

    def next(self):

        all_ims = []
        all_locs = []
        for b_ndx in range(self.batch_size):
            # AL: this 'shuffle' seems weird
            n_skip = np.random.randint(30) if self.shuffle else 0
            for _ in range(n_skip+1):
                record = self.read_next()

            example = tf.train.Example()
            example.ParseFromString(record)
            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            depth = int(example.features.feature['depth'].int64_list.value[0])
            expid = int(example.features.feature['expndx'].float_list.value[0]),
            t = int(example.features.feature['ts'].float_list.value[0]),
            img_string = example.features.feature['image_raw'].bytes_list.value[0]
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((height, width, depth))
            locs = np.array(example.features.feature['locs'].float_list.value)
            locs = locs.reshape([self.conf.n_classes, 2])
            if 'trx_ndx' in example.features.feature.keys():
                trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
            else:
                trx_ndx = 0
            all_ims.append(reconstructed_img)
            all_locs.append(locs)

        ims = np.stack(all_ims)  # [bsize x height x width x depth]
        locs = np.stack(all_locs)  # [bsize x ncls x 2]

        imszuse = self.conf.imszuse
        (imnr_use, imnc_use) = imszuse
        ims = ims[:, 0:imnr_use, 0:imnc_use, :]

        if self.conf.img_dim == 1:
            assert ims.shape[-1] == 1, "Expected image depth of 1"
            ims = np.tile(ims, 3)

        assert self.conf.op_rescale == 1, "op_rescale not sure if we are okay"
        mask_sz = [int(x/self.conf.op_label_scale/self.conf.op_rescale) for x in imszuse]
        mask_sz1 = [self.batch_size,] + mask_sz + [2*self.vec_num]
        mask_sz2 = [self.batch_size,] + mask_sz + [self.heat_num]
        mask_im1 = np.ones(mask_sz1)
        mask_im2 = np.ones(mask_sz2)
        mask_sz_origres = [int(x/self.conf.op_rescale) for x in imszuse]
        mask_sz1_origres = [self.batch_size,] + mask_sz_origres + [2*self.vec_num]
        mask_sz2_origres = [self.batch_size,] + mask_sz_origres + [self.heat_num]
        mask_im1_origres = np.ones(mask_sz1_origres)
        mask_im2_origres = np.ones(mask_sz2_origres)

        ims, locs = PoseTools.preprocess_ims(ims, locs, self.conf,
                                             self.distort, self.conf.op_rescale)
        # locs has been rescaled per op_rescale (but not op_label_scale)

        label_ims = create_label_images(locs/self.conf.op_label_scale, mask_sz, 1) #self.conf.label_blur_rad)
#        label_ims = PoseTools.create_label_images(locs/self.conf.op_label_scale, mask_sz,1,2)
        label_ims = np.clip(label_ims, 0, 1)  # AL: possibly unnec?

        label_ims_origres = create_label_images(locs, mask_sz_origres,
                                                self.conf.label_blur_rad)
        label_ims_origres = np.clip(label_ims_origres, 0, 1) # AL: possibly unnec?

        affinity_ims = create_affinity_labels(locs/self.conf.op_label_scale,
                                              mask_sz, self.conf.op_affinity_graph, 1) #self.conf.label_blur_rad)

        affinity_ims_origres = create_affinity_labels(locs,
                                                      mask_sz_origres,
                                                      self.conf.op_affinity_graph,
                                                      self.conf.label_blur_rad)

        return [ims, mask_im1, mask_im2, mask_im1_origres, mask_im2_origres], \
               [affinity_ims, label_ims,
                affinity_ims, label_ims,
                affinity_ims, label_ims,
                affinity_ims, label_ims,
                affinity_ims, label_ims,
                affinity_ims_origres, label_ims_origres]
        # (inputs, targets)


    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


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

def configure_lr_multipliers(model):
    # setup lr multipliers for conv layers

    lr_mult = dict()
    for layer in model.layers:
        # AL: second clause here unnec as Conv2DTranspose appears to be a subclass.
        # Just for clarity
        if isinstance(layer, Conv2D) or isinstance(layer, Conv2DTranspose):
            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name) or \
               re.match("Mdeconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name) or \
                 re.match("Mdeconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

    return lr_mult


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

def configure_loss_functions(batch_size, stg6_blur_rad, stg6_resfac,
                             stg6_wtfac_paf, stg6_wtfac_prt):
    def eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    loss_weights = {}
    loss_weights_vec = []  # yes this is dumb. loss_weights in order S1L1 S1L2 S2L1 ...
    for stage in range(1, 7):
        for lvl in range(1, 3):
            key = 'weight_stage{}_L{}'.format(stage, lvl)
            losses[key] = eucl_loss
            if stage == 6:
                # stage6 hmap and paf maps are at
                # 1. stg6_resfac (relative) upsampled resolution
                # 2. stg6_blur_rad blur_rad

                ispaf = lvl == 1
                if ispaf:
                    # 1. increased resolution increases natural scale of loss by
                    #    stg6_resfac
                    # 2. increased blur_rad increases " by stg6_blur_rad
                    #       (relative to blur_rad of 1)
                    loss_weights[key] = stg6_wtfac_paf / stg6_resfac / stg6_blur_rad
                    logging.info('Stage 6 paf loss_weight: {}'.format(loss_weights[key]))
                else:
                    # 1. has no effect
                    # 2. increases the natural scale of the loss by stg6_blur_rad**2
                    #     (assuming earlier stages have blur_rad==1)
                    loss_weights[key] = stg6_wtfac_prt / stg6_blur_rad**2
                    logging.info('Stage 6 hmap loss_weight: {}'.format(loss_weights[key]))

                # loss_weights[key] is the end-of-the-day weighting factor passed to K.compile.
                # Empirically 201906 on bub, blur_rad of 3 (and resfac of 8):
                #  mean(val_loss_full_paf_ratio)~25 and
                #  mean(val_loss_full_prt_ratio)~37
                # ie typical raw loss of stg6/(others) is that number; figured we want optimizer to
                # upweight stg6 more in the [1,3] range although very unclear it makes any diff.
                # Note the ratios above are flattish over the convergence/training (with some jumps
                # at eg LR steps); it is not that the stg6 loss is offset to be higher with small
                # reduction as convergence occurs. It is delta-loss that is relevant after all
            else:
                loss_weights[key] = 1.0
            loss_weights_vec.append(loss_weights[key])

    return losses, loss_weights, loss_weights_vec

def dot(K, L):
   assert len(K) == len(L), 'lens do not match: {} vs {}'.format(len(K), len(L))
   return sum(i[0] * i[1] for i in zip(K, L))

def training(conf,name='deepnet'):

    #AL 20190327 For now we massage on the App side so the App knows what to expect for
    # training outputs
    #massage_conf(conf)

    base_lr = 4e-5  # 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    lr_policy = "step"
    batch_size = conf.batch_size
    gamma = conf.gamma
    stepsize = int(conf.decay_steps)
    # stepsize = 68053  # 136106 #   // after each stepsize iterations update learning rate: lr=lr*gamma
    iterations_per_epoch = conf.display_step
    max_iter = conf.dl_steps/iterations_per_epoch
    restart = True
    last_epoch = 0

    (imnr, imnc) = conf.imsz
    imnr_use = imszcheckcrop(imnr, 'row')
    imnc_use = imszcheckcrop(imnc, 'column')
    imszuse = (imnr_use, imnc_use)
    conf.imszuse = imszuse

    assert conf.dl_steps % iterations_per_epoch == 0, 'For open-pose dl steps must be a multiple of display steps'
    assert conf.save_step % iterations_per_epoch == 0, 'For open-pose save steps must be a multiple of display steps'

    train_data_file = os.path.join(conf.cachedir, 'traindata')
    with open(train_data_file, 'wb') as td_file:
        pickle.dump(conf, td_file, protocol=2)
    logging.info('Saved config to {}'.format(train_data_file))

    model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{epoch:d}')
    model = get_training_model(imszuse,
                               weight_decay,
                               conf.weight_decay_kernel_dc,
                               conf.weight_decay_dc_mode,
                               nlimbsT2=len(conf.op_affinity_graph) * 2,
                               npts=conf.n_classes)

    # load previous weights or vgg19 if this is the first run
    from_vgg = dict()
    for blk in range(1,5):
        for lvl in range(1,3):
            from_vgg['conv{}_{}'.format(blk,lvl)] = 'block{}_conv{}'.format(blk,lvl)
    logging.info("Loading vgg19 weights...")
    vgg_model = VGG19(include_top=False, weights='imagenet')
    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            logging.info("Loaded VGG19 layer: " + vgg_layer_name)

    # prepare generators
    train_di = DataIteratorTF(conf, 'train', True, True)
    train_di2 = DataIteratorTF(conf, 'train', True, True)
    val_di = DataIteratorTF(conf, 'train', False, False)

    # AL: looks like lr_mults not used anymore with Adam
    # configure_lr_multipliers(model)
    # logging.info('Configured layer learning rate mulitpliers')

    assert conf.op_label_scale == 8
    logging.info("Your label_blur_rad is {}".format(conf.label_blur_rad))
    losses, loss_weights, loss_weights_vec = \
        configure_loss_functions(batch_size, conf.label_blur_rad, conf.op_label_scale,
                                 conf.op_hires_wtfac_paf, conf.op_hires_wtfac_prt)

    save_time = conf.get('save_time',None)
    # lr decay.
    def step_decay(epoch):
        initial_lrate = base_lr
        steps = epoch * iterations_per_epoch
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
            step = (epoch+1) * conf.display_step
            val_x, val_y = self.val_di.next()
            val_out = self.model.predict(val_x)
            val_loss_full = self.model.evaluate(val_x, val_y, verbose=0)
            val_loss_K = val_loss_full[0]  # want Py 3 unpack
            val_loss_full = val_loss_full[1:]
            val_loss = dot(val_loss_full, loss_weights_vec)
            train_x, train_y = self.train_di.next()
            train_out = self.model.predict(train_x)
            train_loss_full = self.model.evaluate(train_x, train_y, verbose=0)
            train_loss_K = train_loss_full[0]  # want Py 3 unpack
            train_loss_full = train_loss_full[1:]
            train_loss = dot(train_loss_full, loss_weights_vec)
            lr = K.eval(self.model.optimizer.lr)

            # dist only for last layer
            tt1 = PoseTools.get_pred_locs(val_out[-1]) - \
                  PoseTools.get_pred_locs(val_y[-1])
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))  # [bsize x ncls]
            val_dist = np.nanmean(tt1)  # this dist is in op_scale-downsampled space
                                        # *self.config.op_label_scale
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
    lrate = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint(
        model_file, monitor='loss', verbose=0, save_best_only=False,
        save_weights_only=True, mode='min', period=conf.save_step)
    obs = OutputObserver(conf, [train_di2, val_di])
    callbacks_list = [lrate, obs] #checkpoint,

    # sgd optimizer with lr multipliers
    # optimizer = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)#, clipnorm=1.)
    # Mayank 20190423 - Adding clipnorm so that the loss doesn't go to zero.
    optimizer = Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # start training
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

    logging.info("Your model.metrics_names are {}".format(model.metrics_names))

    #save initial model
    model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(0))))

    # training
    model.fit_generator(train_di,
                        steps_per_epoch=conf.display_step,
                        epochs=max_iter-1,
                        callbacks=callbacks_list,
                        verbose=0,
                        # validation_data=val_di,
                        # validation_steps=val_samples // batch_size,
#                        use_multiprocessing=True,
#                        workers=4,
                        initial_epoch=last_epoch
                        )

    # force saving in case the max iter doesn't match the save step.
    model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(int(max_iter*iterations_per_epoch)))))
    obs.on_epoch_end(max_iter-1)


def get_pred_fn(conf, model_file=None, name='deepnet', rawpred=False):
    (imnr, imnc) = conf.imsz
    imnr_use = imszcheckcrop(imnr, 'row')
    imnc_use = imszcheckcrop(imnc, 'column')
    imszuse = (imnr_use, imnc_use)
    conf.imszuse = imszuse

    model = get_testing_model(imszuse,
                              nlimb=len(conf.op_affinity_graph) * 2,
                              npts=conf.n_classes,
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
        all_f = all_f[:, 0:imnr_use, 0:imnc_use, :]

        if all_f.shape[3] == 1:
            all_f = np.tile(all_f,[1,1,1,3])
        # tiling beforehand a little weird as preprocess_ims->normalizexyxy branches on
        # if img is color
        xs, _ = PoseTools.preprocess_ims(
            all_f, in_locs=np.zeros([conf.batch_size, conf.n_classes, 2]), conf=conf,
            distort=False, scale=conf.op_rescale)
        model_preds = model.predict(xs)
        # all_infered = []
        # for ex in range(xs.shape[0]):
        #     infered = do_inference(model_preds[-1][ex,...],model_preds[-2][ex,...],conf, thre1, thre2)
        #     all_infered.append(infered)
        pred = model_preds[-1]
        raw_locs = PoseTools.get_pred_locs(pred)
        raw_locs = raw_locs * conf.op_rescale  # * conf.op_label_scale
        # base_locs = np.array(all_infered)*conf.op_rescale
        # nanidx = np.isnan(base_locs)
        # base_locs[nanidx] = raw_locs[nanidx]
        base_locs = raw_locs
        ret_dict = {}
        ret_dict['locs'] = base_locs
        ret_dict['hmaps'] = pred
        ret_dict['conf'] = np.max(pred, axis=(1, 2))
        return ret_dict

    def pred_fn_rawmaps(all_f):
        all_f = all_f[:, 0:imnr_use, 0:imnc_use, :]

        if all_f.shape[3] == 1:
            all_f = np.tile(all_f,[1,1,1,3])
        # tiling beforehand a little weird as preprocess_ims->normalizexyxy branches on
        # if img is color
        xs, _ = PoseTools.preprocess_ims(
            all_f, in_locs=np.zeros([conf.batch_size, conf.n_classes, 2]), conf=conf,
            distort=False, scale=conf.op_rescale)
        model_preds = model.predict(xs)
        # all_infered = []
        # for ex in range(xs.shape[0]):
        #     infered = do_inference(model_preds[-1][ex,...],model_preds[-2][ex,...],conf, thre1, thre2)
        #     all_infered.append(infered)
        return model_preds


    def close_fn():
        K.clear_session()
        # gc.collect()
        # del model

    if rawpred:
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
