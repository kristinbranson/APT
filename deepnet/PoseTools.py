from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[ ]:

from past.builtins import cmp
from builtins import range
from past.utils import old_div
import numpy as np
import scipy, re
import math, h5py
# import caffe
from scipy import misc
from scipy import ndimage
import tensorflow as tf
import multiResData
import tempfile
#import cv2
#import PoseTrain
import myutils
import os
import cv2
from cvc import cvc
import math
import sys
import copy
from scipy import io
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import json
from skimage import transform
import datetime
from scipy.ndimage.interpolation import zoom

# from matplotlib.backends.backend_agg import FigureCanvasAgg


# In[ ]:

# not used anymore
# def scalepatches(patch,scale,num,rescale,cropsz):
#     sz = patch.shape
#     assert sz[0]%( (scale**(num-1))*rescale) is 0,"patch size isn't divisible by scale"

#     patches = []
#     patches.append(scipy.misc.imresize(patch[:,:,0],1.0/(scale**(num-1))/rescale))
#     curpatch = patch
#     for ndx in range(num-1):
#         sz = curpatch.shape
#         crop = int((1-1.0/scale)/2*sz[0])
# #         print(ndx,crop)

#         spatch = curpatch[crop:-crop,crop:-crop,:]
#         curpatch = spatch
#         sfactor = 1.0/(scale**(num-ndx-2))/rescale
#         tpatch = scipy.misc.imresize(spatch,sfactor,)
#         patches.append(tpatch[:,:,0])
#     return patches


# In[ ]:

def get_cmap(n_classes):
    cmap = cm.get_cmap('jet')
    return cmap(np.linspace(0, 1, n_classes))


def scale_images(img, locs, scale, conf):
    sz = img.shape
    simg = np.zeros((sz[0], int(float(sz[1])/ scale), int(float(sz[2])/ scale), sz[3]))
    for ndx in range(sz[0]):
        if sz[3] == 1:
            simg[ndx, :, :, 0] = transform.resize(img[ndx, :, :, 0], simg.shape[1:3], preserve_range=True)
        else:
            simg[ndx, :, :, :] = transform.resize(img[ndx, :, :, :], simg.shape[1:3], preserve_range= True)
    new_locs = locs.copy()
    new_locs = new_locs/scale
    return simg, new_locs


def normalize_mean(in_img, conf):
    zz = in_img.astype('float')
    if conf.normalize_img_mean:
        # subtract mean for each img.
        mm = zz.mean(axis=(1,2))
        xx = zz - mm[:, np.newaxis, np.newaxis,:]
        if conf.imgDim == 3:
            if conf.perturb_color:
                for dim in range(3):
                    to_add = old_div(((np.random.rand(conf.batch_size) - 0.5) * conf.imax), 8)
                    xx[:, :, :, dim] += to_add[:, np.newaxis, np.newaxis]
    # elif not hasattr(conf, 'normalize_batch_mean') or conf.normalize_batch_mean:
    elif conf.normalize_batch_mean:
        # subtract the batch mean if the variable is not defined.
        # don't know why I have it. :/
        xx = zz - zz.mean()
    else:
        xx = zz
    return xx


def multi_scale_images(in_img, rescale, scale, conf):
    # only crop the highest res image
    #     if l1_cropsz > 0:
    #         inImg_crop = inImg[:,l1_cropsz:-l1_cropsz,l1_cropsz:-l1_cropsz,:]
    #     else:
    #         inImg_crop = inImg

    in_img = adjust_contrast(in_img, conf)
    x0_in = scale_images(in_img, rescale, conf)
    x1_in = scale_images(in_img, rescale * scale, conf)
    x2_in = scale_images(x1_in, scale, conf)
    return x0_in, x1_in, x2_in


def multi_scale_label_images(in_img, rescale, scale):
    # only crop the highest res image
    #     if l1_cropsz > 0:
    #         inImg_crop = inImg[:,l1_cropsz:-l1_cropsz,l1_cropsz:-l1_cropsz,:]
    #     else:
    #         inImg_crop = inImg

    x0_in = scale_images(in_img, rescale)
    x1_in = scale_images(in_img, rescale * scale)
    x2_in = scale_images(x1_in, scale)
    return x0_in, x1_in, x2_in


def adjust_contrast(in_img, conf):
    if conf.adjustContrast:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(conf.clahegridsize, conf.clahegridsize))
        simg = np.zeros(in_img.shape)
        if in_img.shape[3] == 1:
            for ndx in range(in_img.shape[0]):
                simg[ndx, :, :, 0] = clahe.apply(in_img[ndx,:,:,0 ].astype('uint8')).astype('float')
        else:
            for ndx in range(in_img.shape[0]):
                lab = cv2.cvtColor(in_img[ndx,...], cv2.COLOR_RGB2LAB)
                lab_planes = cv2.split(lab)
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                simg[ndx,...] = rgb
        return simg
    else:
        return in_img


def process_image(frame_in, conf):
    #     cropx = (framein.shape[0] - conf.imsz[0])/2
    #     cropy = (framein.shape[1] - conf.imsz[1])/2
    #     if cropx > 0:
    #         framein = framein[cropx:-cropx,:,:]
    #     if cropy > 0:
    #         framein = framein[:,cropy:-cropy,:]
    frame_in = crop_images(frame_in, conf)
    frame_in = frame_in[np.newaxis, :, :, 0:1]
    x0, x1, x2 = multi_scale_images(frame_in, conf.rescale, conf.scale, conf.l1_cropsz)
    return x0, x1, x2


def crop_images(frame_in, conf):
    cshape = tuple(frame_in.shape[0:2])
    start = conf.cropLoc[cshape]  # cropLoc[0] should be for y.
    end = [conf.imsz[ndx] + start[ndx] for ndx in range(2)]
    return frame_in[start[0]:end[0], start[1]:end[1], :]


def randomly_flip_lr(img, locs, group_sz = 1):
    if locs.ndim == 3:
        reduce_dim = True
        locs = locs[:,np.newaxis,...]
    else:
        reduce_dim = False

    num = img.shape[0]
    n_groups = num/group_sz
    for ndx in range(n_groups):
        st = ndx*group_sz
        en = (ndx+1)*group_sz
        jj = np.random.randint(2)
        if jj > 0.5:
            img[st:en, ...] = img[st:en, :, ::-1, :]
            locs[st:en, :, :, 0] = img.shape[3] - locs[st:en, :, :, 0]

    locs = locs[:, 0, ...] if reduce_dim else locs
    return img, locs


def randomly_flip_ud(img, locs,group_sz = 1):
    if locs.ndim == 3:
        reduce_dim = True
        locs = locs[:,np.newaxis,...]
    else:
        reduce_dim = False

    num = img.shape[0]
    n_groups = num/group_sz
    for ndx in range(n_groups):
        st = ndx*group_sz
        en = (ndx+1)*group_sz
        jj = np.random.randint(2)
        if jj > 0.5:
            img[st:en, ...] = img[st:en, ::-1, : ,: ]
            locs[st:en, :, :, 1] = img.shape[2] - locs[st:en, :, : , 1]
    locs = locs[:, 0, ...] if reduce_dim else locs
    return img, locs


def randomly_translate(img, locs, conf, group_sz = 1):
    if conf.trange < 1:
        return img, locs

    if locs.ndim == 3: # hack for multi animal
        reduce_dim = True
        locs = locs[:,np.newaxis,...]
    else:
        reduce_dim = False

    num = img.shape[0]
    rows, cols = img.shape[1:3]
    n_groups = num/group_sz
    for ndx in range(n_groups):
        st = ndx*group_sz
        en = (ndx+1)*group_sz
        orig_locs = copy.deepcopy(locs[ndx, ...])
        orig_im = copy.deepcopy(img[st:en, ...])
        sane = False
        do_move = True

        count = 0
        ll = orig_locs.copy()
        out_ii = orig_im.copy()
        while not sane:
            valid = np.invert(np.isnan(orig_locs[:, :, 0]))
            dx = np.random.randint(-conf.trange, conf.trange)
            dy = np.random.randint(-conf.trange, conf.trange)
            count += 1
            if count > 5:
                dx = 0
                dy = 0
                sane = True
                do_move = False
            ll = copy.deepcopy(orig_locs)
            ll[:, :, 0] += dx
            ll[:, :, 1] += dy
            if np.all(ll[valid,0] >= 0) and \
                    np.all(ll[valid, 1] >= 0) and \
                    np.all(ll[valid, 0] < cols) and \
                    np.all(ll[valid, 1] < rows):
                sane = True
            elif do_move:
                continue

            # else:
            #                 print 'not sane {}'.format(count)
            mat = np.float32([[1, 0, dx], [0, 1, dy]])
            for g in range(group_sz):
                ii = copy.deepcopy(orig_im[g,...])
                ii = cv2.warpAffine(ii, mat, (cols, rows))
                if ii.ndim == 2:
                    ii = ii[..., np.newaxis]
                out_ii[g,...] = ii
        locs[ndx, ...] = ll
        img[st:en, ...] = out_ii

    locs = locs[:, 0, ...] if reduce_dim else locs
    return img, locs


def randomly_rotate(img, locs, conf, group_sz = 1):
    if conf.rrange < 1:
        return img, locs

    if locs.ndim == 3: # hack for multi animal
        reduce_dim = True
        locs = locs[:,np.newaxis,...]
    else:
        reduce_dim = False

    num = img.shape[0]
    rows, cols = img.shape[1:3]
    n_groups = num/group_sz
    for ndx in range(n_groups):
        st = ndx*group_sz
        en = (ndx+1)*group_sz
        orig_locs = copy.deepcopy(locs[ndx, ...])
        orig_im = copy.deepcopy(img[st:en, ...])
        sane = False
        do_rotate = True

        count = 0
        lr = orig_locs.copy()
        out_ii = orig_im.copy()
        while not sane:
            valid = np.invert(np.isnan(orig_locs[:, :, 0]))
            rangle = (np.random.rand() * 2 - 1) * conf.rrange
            count += 1
            if count > 5:
                rangle = 0
                sane = True
                do_rotate = False
            ll = copy.deepcopy(orig_locs)
            ll = ll - [old_div(cols, 2), old_div(rows, 2)]
            ang = np.deg2rad(rangle)
            rot = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
            lr = np.zeros(ll.shape)
            for i_ndx in range(ll.shape[0]):
                lr[i_ndx,...] = np.dot(ll[i_ndx], rot) + [old_div(cols, 2), old_div(rows, 2)]
            if np.all(lr[valid, 0] > 0) \
                    and np.all(lr[valid, 1] >0) \
                    and np.all(lr[valid, 0] <= cols) \
                    and np.all(lr[valid, 1] <= rows):
                sane = True
            elif do_rotate:
                continue

            # else:
            #                 print 'not sane {}'.format(count)
            mat = cv2.getRotationMatrix2D((old_div(cols, 2), old_div(rows, 2)), rangle, 1)
            for g in range(group_sz):
                ii = copy.deepcopy(orig_im[g,...])
                ii = cv2.warpAffine(ii, mat, (cols, rows))
                if ii.ndim == 2:
                    ii = ii[..., np.newaxis]
                out_ii[g,...] = ii
        locs[ndx, ...] = lr
        img[st:en, ...] = out_ii

    locs = locs[:, 0, ...] if reduce_dim else locs
    return img, locs


def randomly_adjust(img, conf, group_sz = 1):
    # For images between 0 to 255
    # and single channel
    num = img.shape[0]
    brange = conf.brange
    bdiff = brange[1] - brange[0]
    crange = conf.crange
    cdiff = crange[1] - crange[0]
    imax = conf.imax
    n_groups = num/group_sz
    for ndx in range(n_groups):
        st = ndx*group_sz
        en = (ndx+1)*group_sz
        bfactor = np.random.rand() * bdiff + brange[0]
        cfactor = np.random.rand() * cdiff + crange[0]
        mm = img[st:en, ...].mean()
        for g in range(group_sz):
            jj = img[st+g, ...] + bfactor * imax
            jj = np.minimum(imax, (jj - mm) * cfactor + mm)
            jj = jj.clip(0, imax)
            img[st+g, ...] = jj
    return img


def randomly_scale(img,locs,conf,group_sz=1):
    # For images between 0 to 255
    # and single channel
    im_sz = img.shape[1:]
    num = img.shape[0]
    srange = conf.scale_range
    n_groups = num/group_sz
    for ndx in range(n_groups):
        st = ndx*group_sz
        en = (ndx+1)*group_sz
        sfactor = (np.random.rand()-0.5)*srange + 1

        for g in range(group_sz):
            jj = img[st+g, ...].copy()
            cur_img = zoom(jj, sfactor) if srange != 0 else jj
            cur_img, dx, dy = crop_to_size(cur_img, im_sz)
            img[st+g, ...] =cur_img
            locs[st+g,...,0] = locs[st+g,...,0]*sfactor + dx/2
            locs[st + g, ..., 1] = locs[st + g, ..., 1]*sfactor + dy / 2
    return img, locs


def blur_label(im_sz, loc, scale, blur_rad):
    sz0 = int(math.ceil(old_div(float(im_sz[0]), scale)))
    sz1 = int(math.ceil(old_div(float(im_sz[1]), scale)))

    label = np.zeros([sz0, sz1])
    if not np.isnan(loc[0]):
        label[int(old_div(loc[0], scale)), int(old_div(loc[1], scale))] = 1
        #         blurL = ndimage.gaussian_filter(label,blur_rad)
        ksize = 2 * 3 * blur_rad + 1
        b_label = cv2.GaussianBlur(label, (ksize, ksize), blur_rad)
        b_label = old_div(b_label, b_label.max())
    else:
        b_label = label
    return b_label


def create_label_images(locs, im_sz, scale, blur_rad):
    n_classes = len(locs[0])
    sz0 = int(float(im_sz[0])/ scale)
    sz1 = int(float(im_sz[1])/ scale)

    label_ims = np.zeros((len(locs), sz0, sz1, n_classes))
    # labelims1 = np.zeros((len(locs),sz0,sz1,n_classes))
    k_size = 3 * blur_rad
    blur_l = np.zeros([2 * k_size + 1, 2 * k_size + 1])
    blur_l[k_size, k_size] = 1
    blur_l = cv2.GaussianBlur(blur_l, (2 * k_size + 1, 2 * k_size + 1), blur_rad)
    blur_l = old_div(blur_l, blur_l.max())
    for cls in range(n_classes):
        for ndx in range(len(locs)):
            if np.isnan(locs[ndx][cls][0]) or np.isinf(locs[ndx][cls][0]):
                continue
            if np.isnan(locs[ndx][cls][1]) or np.isinf(locs[ndx][cls][1]):
                continue
                #             modlocs = [locs[ndx][cls][1],locs[ndx][cls][0]]
            #             labelims1[ndx,:,:,cls] = blurLabel(imsz,modlocs,scale,blur_rad)
            modlocs0 = int(np.round(old_div(locs[ndx][cls][1], scale)))
            modlocs1 = int(np.round(old_div(locs[ndx][cls][0], scale)))
            l0 = min(sz0, max(0, modlocs0 - k_size))
            r0 = max(0, min(sz0, modlocs0 + k_size + 1))
            l1 = min(sz1, max(0, modlocs1 - k_size))
            r1 = max(0, min(sz1, modlocs1 + k_size + 1))
            label_ims[ndx, l0:r0, l1:r1, cls] = blur_l[(l0 - modlocs0 + k_size):(r0 - modlocs0 + k_size),
                                                (l1 - modlocs1 + k_size):(r1 - modlocs1 + k_size)]

    label_ims = 2.0 * (label_ims - 0.5)
    return label_ims


def create_reg_label_images(locs, im_sz, scale, blur_rad):
    n_classes = len(locs[0])
    sz0 = int(math.ceil(old_div(float(im_sz[0]), scale)))
    sz1 = int(math.ceil(old_div(float(im_sz[1]), scale)))

    labelims = np.zeros((len(locs), sz0, sz1, n_classes))
    regimsx = np.zeros((len(locs), sz0, sz1, n_classes))
    regimsy = np.zeros((len(locs), sz0, sz1, n_classes))
    for cls in range(n_classes):
        for ndx in range(len(locs)):
            # x,y = np.meshgrid(np.arange(sz0),np.arange(sz1))
            modlocs = [locs[ndx][cls][1], locs[ndx][cls][0]]
            labelims[ndx, :, :, cls] = blur_label(im_sz, modlocs, scale, blur_rad)

            #             np.sqrt((x-(round(locs[ndx][cls][0]/scale)))**2 +
            #                                (y-(round(locs[ndx][cls][1]/scale)))**2) < (rad-1)
            #             xmin = int(max(round((locs[ndx][cls][0])/scale - rad),0))
            #             xmax = int(min(round((locs[ndx][cls][0])/scale + rad),sz0))
            #             ymin = int(max(round((locs[ndx][cls][1])/scale - rad),0))
            #             ymax = int(min(round((locs[ndx][cls][1])/scale + rad),sz0))
            #             labelims[ndx,ymin:ymax,xmin:xmax,cls] = 1.
            tx, ty = np.meshgrid(np.arange(sz0) * scale, np.arange(sz1) * scale)
            tregx = tx.astype('float64')
            tregy = ty.astype('float64')
            tregx = locs[ndx][cls][0] - 1 - tregx
            tregy = locs[ndx][cls][1] - 1 - tregy
            regimsx[ndx, :, :, cls] = tregx
            regimsy[ndx, :, :, cls] = tregy

    labelims = 2.0 * (labelims - 0.5)
    return labelims, regimsx, regimsy


def create_fine_label_tensor(conf):
    tsz = int(conf.fine_sz + 2 * 6 * math.ceil(conf.fine_label_blur_rad))
    timg = np.zeros((tsz, tsz))
    timg[old_div(tsz, 2), old_div(tsz, 2)] = 1
    blur_l = ndimage.gaussian_filter(timg, conf.fine_label_blur_rad)
    blur_l = old_div(blur_l, blur_l.max())
    blur_l = 2.0 * (blur_l - 0.5)
    return tf.constant(blur_l)


def extract_fine_label_tensor(label_t, sz, dd, fsz):
    return tf.slice(label_t, dd + old_div(sz, 2) - old_div(fsz, 2), [fsz, fsz])


def create_fine_label_images(locs, max_locs, conf, label_t):
    max_locs = tf.to_int32(max_locs / conf.rescale)
    tsz = int(conf.fine_sz + 2 * 6 * math.ceil(conf.fine_label_blur_rad))
    hsz = old_div(conf.fine_sz, 2)
    limgs = []
    for inum in range(conf.batch_size):
        curlimgs = []
        for ndx in range(conf.n_classes):
            dx = max_locs[inum, ndx, 0] - tf.to_int32(old_div(locs[inum, ndx, 0], conf.rescale))
            dy = max_locs[inum, ndx, 1] - tf.to_int32(old_div(locs[inum, ndx, 1], conf.rescale))
            dd = tf.stack([dy, dx])
            dd = tf.maximum(tf.to_int32(hsz - old_div(tsz, 2)), tf.minimum(tf.to_int32(old_div(tsz, 2) - hsz - 1), dd))
            curlimgs.append(extract_fine_label_tensor(label_t, tsz, dd, conf.fine_sz))
        limgs.append(tf.stack(curlimgs))
    return tf.transpose(tf.stack(limgs), [0, 2, 3, 1])


def arg_max_2d(x_in):
    orig_shape = tf.shape(x_in)
    reshape_t = tf.concat([orig_shape[0:1], [-1], orig_shape[3:4]], 0)
    zz = tf.reshape(x_in, reshape_t)
    pp = tf.to_int32(tf.argmax(zz, 1))
    sz1 = tf.slice(orig_shape, [2], [1])
    cc1 = tf.div(pp, tf.to_int32(sz1))
    cc2 = tf.mod(pp, tf.to_int32(sz1))

    return tf.stack([cc1, cc2])


def get_base_pred_locs(pred, conf):
    pred_locs = np.zeros([pred.shape[0], conf.n_classes, 2])
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            max_ndx = np.argmax(pred[ndx, :, :, cls])
            cur_loc = np.array(np.unravel_index(max_ndx, pred.shape[1:3]))
            cur_loc = cur_loc * conf.pool_scale * conf.rescale
            pred_locs[ndx, cls, 0] = cur_loc[1]
            pred_locs[ndx, cls, 1] = cur_loc[0]
    return pred_locs


def get_pred_locs(pred, edge_ignore=1):
    if edge_ignore < 1:
        edge_ignore = 1
    n_classes = pred.shape[3]
    pred_locs = np.zeros([pred.shape[0], n_classes, 2])
    for ndx in range(pred.shape[0]):
        for cls in range(n_classes):
            cur_pred = pred[ndx, :, :, cls].copy()
            cur_pred[:edge_ignore,:] = cur_pred.min()
            cur_pred[:,:edge_ignore] = cur_pred.min()
            cur_pred[-edge_ignore:,:] = cur_pred.min()
            cur_pred[:,-edge_ignore:] = cur_pred.min()
            maxndx = np.argmax(cur_pred)
            curloc = np.array(np.unravel_index(maxndx, pred.shape[1:3]))
            pred_locs[ndx, cls, 0] = curloc[1]
            pred_locs[ndx, cls, 1] = curloc[0]
    return pred_locs


def get_pred_locs_multi(pred, n_max, sz):
    pred = pred.copy()
    n_classes = pred.shape[3]
    pred_locs = np.zeros([pred.shape[0], n_max, n_classes, 2])
    for ndx in range(pred.shape[0]):
        for cls in range(n_classes):
            for count in range(n_max):
                maxndx = np.argmax(pred[ndx, :, :, cls])
                curloc = np.array(np.unravel_index(maxndx, pred.shape[1:3]))
                pred_locs[ndx, count, cls, 0] = curloc[1]
                pred_locs[ndx, count, cls, 1] = curloc[0]
                miny = max(curloc[0]-sz,0)
                maxy = min(curloc[0]+sz,pred.shape[1])
                minx = max(curloc[1]-sz,0)
                maxx = min(curloc[1]+sz,pred.shape[2])
                pred[ndx, miny:maxy, minx:maxx, cls] = pred.min()
    return pred_locs


def get_fine_pred_locs(pred, fine_pred, conf):
    pred_locs = np.zeros([pred.shape[0], conf.n_classes, 2])
    fine_pred_locs = np.zeros([pred.shape[0], conf.n_classes, 2])
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            max_ndx = np.argmax(pred[ndx, :, :, cls])
            cur_loc = np.array(np.unravel_index(max_ndx, pred.shape[1:3]))
            cur_loc = cur_loc * conf.pool_scale * conf.rescale
            pred_locs[ndx, cls, 0] = cur_loc[1]
            pred_locs[ndx, cls, 1] = cur_loc[0]
            max_ndx = np.argmax(fine_pred[ndx, :, :, cls])
            curfineloc = (np.array(np.unravel_index(max_ndx, fine_pred.shape[1:3])) -
                          old_div(conf.fine_sz, 2)) * conf.rescale
            fine_pred_locs[ndx, cls, 0] = cur_loc[1] + curfineloc[1]
            fine_pred_locs[ndx, cls, 1] = cur_loc[0] + curfineloc[0]
    return pred_locs, fine_pred_locs


def get_base_error(locs, pred, conf):
    loc_err = np.zeros(locs.shape)
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            max_ndx = np.argmax(pred[ndx, :, :, cls])
            pred_loc = np.array(np.unravel_index(max_ndx, pred.shape[1:3]))
            pred_loc = pred_loc * conf.pool_scale * conf.rescale
            loc_err[ndx][cls][0] = float(pred_loc[1]) - locs[ndx][cls][0]
            loc_err[ndx][cls][1] = float(pred_loc[0]) - locs[ndx][cls][1]
    return loc_err


def get_fine_error(locs, pred, fine_pred, conf):
    fine_loc_err = np.zeros([len(locs), conf.n_classes, 2])
    base_loc_err = np.zeros([len(locs), conf.n_classes, 2])
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            maxndx = np.argmax(pred[ndx, :, :, cls])
            predloc = np.array(np.unravel_index(maxndx, pred.shape[1:3]))
            predloc = predloc * conf.pool_scale * conf.rescale
            maxndx = np.argmax(fine_pred[ndx, :, :, cls])
            finepredloc = (np.array(np.unravel_index(maxndx, fine_pred.shape[1:3])) - old_div(conf.fine_sz,
                                                                                              2)) * conf.rescale
            base_loc_err[ndx, cls, 0] = float(predloc[1]) - locs[ndx][cls][0]
            base_loc_err[ndx, cls, 1] = float(predloc[0]) - locs[ndx][cls][1]
            fine_loc_err[ndx, cls, 0] = float(predloc[1] + finepredloc[1]) - locs[ndx][cls][0]
            fine_loc_err[ndx, cls, 1] = float(predloc[0] + finepredloc[0]) - locs[ndx][cls][1]
    return base_loc_err, fine_loc_err


def init_mrf_weights(conf):
    lbl = h5py.File(conf.labelfile, 'r')

    if 'pts' in lbl:
        pts = np.array(lbl['pts'])
        v = conf.view
    else:
        pp = np.array(lbl['labeledpos'])
        nmovie = pp.shape[1]
        pts = np.zeros([0, conf.n_classes, 1, 2])
        v = 0
        for ndx in range(nmovie):
            curpts = np.array(lbl[pp[0, ndx]])
            frames = np.where(np.invert(np.any(np.isnan(curpts), axis=(1, 2))))[0]
            npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
            pts_st = int(conf.view * npts_per_view)
            selpts = pts_st + conf.selpts
            curlocs = curpts[:, :, selpts]
            curlocs = curlocs[frames, :, :]
            curlocs = curlocs.transpose([0, 2, 1])
            pts = np.append(pts, curlocs[:, :, np.newaxis, :], axis=0)

    if hasattr(conf, 'mrf_psz'):
        psz = conf.mrf_psz
        print('!!!Overriding MRF Size using conf.mrf_psz!!!')
        print('!!!Overriding MRF Size using conf.mrf_psz!!!')
        print('!!!Overriding MRF Size using conf.mrf_psz!!!')
    else:
        dx = np.zeros([pts.shape[0]])
        dy = np.zeros([pts.shape[0]])
        for ndx in range(pts.shape[0]):
            dx[ndx] = pts[ndx, :, v, 0].max() - pts[ndx, :, v, 0].min()
            dy[ndx] = pts[ndx, :, v, 1].max() - pts[ndx, :, v, 1].min()
        maxd = max((np.percentile(dx, 99), np.percentile(dy, 99)))
        psz = int(math.ceil(old_div((maxd * 2 / conf.rescale), conf.pool_scale)))
    bfilt = np.zeros([psz, psz, conf.n_classes, conf.n_classes])

    for ndx in range(pts.shape[0]):
        for c1 in range(conf.n_classes):
            for c2 in range(conf.n_classes):
                d12x = pts[ndx, c1, v, 0] - pts[ndx, c2, v, 0]
                d12y = pts[ndx, c1, v, 1] - pts[ndx, c2, v, 1]
                if np.isinf(d12y) or np.isinf(d12y):
                    continue
                if np.isnan(d12y) or np.isnan(d12y):
                    continue
                d12x = max(old_div(-psz, 2) + 1,
                           min(old_div(psz, 2) - 1, int(old_div((old_div(d12x, conf.rescale)), conf.pool_scale))))
                d12y = max(old_div(-psz, 2) + 1,
                           min(old_div(psz, 2) - 1, int(old_div((old_div(d12y, conf.rescale)), conf.pool_scale))))
                bfilt[old_div(psz, 2) + d12y, old_div(psz, 2) + d12x, c1, c2] += 1
    bfilt = (old_div(bfilt, pts.shape[0]))
    return bfilt


def init_mrf_weights_identity(conf):
    lbl = h5py.File(conf.labelfile)
    pts = np.array(lbl['pts'])
    v = conf.view
    dx = np.zeros([pts.shape[0]])
    dy = np.zeros([pts.shape[0]])
    for ndx in range(pts.shape[0]):
        dx[ndx] = pts[ndx, :, v, 0].max() - pts[ndx, :, v, 0].min()
        dy[ndx] = pts[ndx, :, v, 1].max() - pts[ndx, :, v, 1].min()
    maxd = max(dx.max(), dy.max())
    hsz = int(old_div(math.ceil(old_div((maxd * 2 / conf.rescale), conf.pool_scale)), 2))
    psz = hsz * 2 + 1
    #     psz = conf.mrf_psz
    bfilt = np.zeros([psz, psz, conf.n_classes, conf.n_classes])

    for c1 in range(conf.n_classes):
        bfilt[hsz, hsz, c1, c1] = 5.
    return bfilt


def get_vars(vstr):
    var_list = tf.global_variables()
    b_list = []
    for var in var_list:
        if re.match(vstr, var.name):
            b_list.append(var)
    return b_list


def compare_conf(curconf, oldconf):
    ff = dir(curconf)
    for f in ff:
        if f[0:2] == '__' or f[0:3] == 'get':
            continue
        if hasattr(curconf, f) and hasattr(oldconf, f):
            if type(getattr(curconf, f)) is np.ndarray:
                print('%s' % f)
                print('New:', getattr(curconf, f))
                print('Old:', getattr(oldconf, f))

            elif type(getattr(curconf, f)) is list:
                if type(getattr(oldconf, f)) is list:
                    if not cmp(getattr(curconf, f), getattr(oldconf, f)):
                        print('%s doesnt match' % f)
                else:
                    print('%s doesnt match' % f)

            elif getattr(curconf, f) != getattr(oldconf, f):
                print('%s doesnt match' % f)

        else:
            print('%s doesnt match' % f)


# def create_network(conf, outtype):
#     self = PoseTrain.PoseTrain(conf)
#     self.createPH()
#     self.createFeedDict()
#     do_batch_norm = self.conf.doBatchNorm
#     self.feed_dict[self.ph['phase_train_base']] = False
#     self.feed_dict[self.ph['phase_train_fine']] = False
#     self.feed_dict[self.ph['keep_prob']] = 1.
#     self.feed_dict[self.ph['learning_rate']] = 0
#     tt = self.ph['y'].get_shape().as_list()
#     tt[0] = 1
#     self.feed_dict[self.ph['y']] = np.zeros(tt)
#     tt = self.ph['locs'].get_shape().as_list()
#     self.feed_dict[self.ph['locs']] = np.zeros(tt)
#
#     with tf.variable_scope('base'):
#         self.createBaseNetwork(do_batch_norm)
#     self.createBaseSaver()
#
#     if outtype > 1 and self.conf.useMRF:
#         with tf.variable_scope('mrf'):
#             self.createMRFNetwork(do_batch_norm)
#         self.createMRFSaver()
#
#     if outtype > 2:
#         with tf.variable_scope('fine'):
#             self.createFineNetwork(do_batch_norm)
#         self.createFineSaver()
#
#     return self
#

def init_network(self, sess, outtype):
    self.restoreBase(sess, True)
    if outtype > 1:
        self.restoreMRF(sess, True)
    if outtype > 2:
        self.restoreFine(sess, True)
    self.initializeRemainingVars(sess)


def open_movie(movie_name):
    cap = cv2.VideoCapture(movie_name)
    nframes = int(cap.get(cvc.FRAME_COUNT))
    return cap, nframes


def create_pred_image(pred_scores, n_classes):
    im = np.zeros(pred_scores.shape[0:2] + (3,))
    im[:, :, 0] = np.argmax(pred_scores, 2).astype('float32') / n_classes * 180
    im[:, :, 1] = (np.max(pred_scores, 2) + 1) / 2 * 255
    im[:, :, 2] = 255.
    im = np.clip(im, 0, 255)
    im = im.astype('uint8')
    return cv2.cvtColor(im, cv2.COLOR_HSV2RGB)


# In[ ]:

def classify_movie(conf, movie_name, out_type, self, sess, max_frames=-1, start_at=0):
    # maxframes if specificied reads that many frames
    # start at specifies where to start reading.

    cap = cv2.VideoCapture(movie_name)
    n_frames = int(cap.get(cvc.FRAME_COUNT))

    # figure out how many frames to read
    if max_frames > 0:
        if max_frames + start_at > n_frames:
            n_frames = n_frames - start_at
        else:
            n_frames = max_frames
    else:
        n_frames = n_frames - start_at

    # since out of order frame access in python sucks, do it one by one
    for _ in range(start_at):
        cap.read()

    # pre allocate results
    pred_locs = np.zeros([n_frames, conf.n_classes, 2, 2])
    predmaxscores = np.zeros([n_frames, conf.n_classes, 2])

    if out_type == 3:
        if self.conf.useMRF:
            pred_pair = [self.mrfPred, self.finePred]
        else:
            pred_pair = [self.basePred, self.finePred]
    elif out_type == 2:
        pred_pair = [self.mrfPred, self.basePred]
    else:
        pred_pair = [self.basePred]

    bsize = conf.batch_size
    nbatches = int(math.ceil(old_div(float(n_frames), bsize)))
    #     framein = myutils.readframe(cap,1)
    #     framein = cropImages(framein,conf)
    #     framein = framein[np.newaxis,:,:,0:1]
    #     x0t,x1t,x2t = multiScaleImages(framein, conf.rescale,  conf.scale, conf.l1_cropsz,conf)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    #     print "WARNING!!!ATTENTION!!! DOING CONTRAST NORMALIZATION!!!!"
    #     print "WARNING!!!ATTENTION!!! DOING CONTRAST NORMALIZATION!!!!"

    allf = np.zeros((bsize,) + conf.imsz + (1,))
    for curl in range(nbatches):

        ndxst = curl * bsize
        ndxe = min(n_frames, (curl + 1) * bsize)
        ppe = min(ndxe - ndxst, bsize)
        for ii in range(ppe):
            success, framein = cap.read()
            assert success, "Could not read frame"

            framein = crop_images(framein, conf)
            #             framein = clahe.apply(framein[:,:,:1])
            allf[ii, ...] = framein[..., 0:1]

        x0, x1, x2 = multi_scale_images(allf, conf.rescale, conf.scale, conf.l1_cropsz, conf)

        self.feed_dict[self.ph['x0']] = x0
        self.feed_dict[self.ph['x1']] = x1
        self.feed_dict[self.ph['x2']] = x2
        pred = sess.run(pred_pair, self.feed_dict)
        if curl == 0:
            predscores = np.zeros((n_frames,) + pred[0].shape[1:] + (2,))
        if out_type == 3:
            base_locs, fine_locs = get_fine_pred_locs(pred[0], pred[1], conf)
            pred_locs[ndxst:ndxe, :, 0, :] = fine_locs[:ppe, :, :]
            pred_locs[ndxst:ndxe, :, 1, :] = base_locs[:ppe, :, :]
            for ndx in range(conf.n_classes):
                predmaxscores[ndxst:ndxe, :, 0] = pred[0][:ppe, :, :, ndx].max()
                predmaxscores[ndxst:ndxe, :, 1] = pred[1][:ppe, :, :, ndx].max()
            predscores[curl, :, :, :, 0] = pred[0][:ppe, :, :, :]
        elif out_type == 2:
            base_locs = get_base_pred_locs(pred[0], conf)
            pred_locs[ndxst:ndxe, :, 0, :] = base_locs[:ppe, :, :]
            base_locs = get_base_pred_locs(pred[1], conf)
            pred_locs[ndxst:ndxe, :, 1, :] = base_locs[:ppe, :, :]
            for ndx in range(conf.n_classes):
                predmaxscores[ndxst:ndxe, :, 0] = pred[0][:ppe, :, :, ndx].max()
                predmaxscores[ndxst:ndxe, :, 1] = pred[1][:ppe, :, :, ndx].max()
            predscores[ndxst:ndxe, :, :, :, 0] = pred[0][:ppe, :, :, :]
            predscores[ndxst:ndxe, :, :, :, 1] = pred[1][:ppe, :, :, :]
        elif out_type == 1:
            base_locs = get_base_pred_locs(pred[0], conf)
            pred_locs[ndxst:ndxe, :, 0, :] = base_locs[:ppe, :, :]
            for ndx in range(conf.n_classes):
                predmaxscores[ndxst:ndxe, :, 0] = pred[0][:ppe, :, :, ndx].max()
            predscores[ndxst:ndxe, :, :, :, 0] = pred[0][:ppe, :, :, :]
        sys.stdout.write('.')
        if curl % 20 == 19:
            sys.stdout.write('\n')

    cap.release()
    return pred_locs, predscores, predmaxscores


def classify_movie_fine(conf, movie_name, locs, self, sess, max_frames=-1, start_at=0):
    # maxframes if specificied reads that many frames
    # start at specifies where to start reading.

    cap = cv2.VideoCapture(movie_name)
    nframes = int(cap.get(cvc.FRAME_COUNT))

    # figure out how many frames to read
    if max_frames > 0:
        if max_frames + start_at > nframes:
            nframes = nframes - start_at
        else:
            nframes = max_frames
    else:
        nframes = nframes - start_at

    # since out of order frame access in python sucks, do it in order to exhaust it
    for _ in range(start_at):
        cap.read()

    # pre allocate results
    pred_locs = np.zeros([nframes, conf.n_classes, 2])

    if self.conf.useMRF:
        pred_pair = [self.mrfPred, self.finePred]
    else:
        pred_pair = [self.basePred, self.finePred]

    bsize = conf.batch_size
    nbatches = int(math.ceil(old_div(float(nframes), bsize)))
    #     framein = myutils.readframe(cap,1)
    #     framein = cropImages(framein,conf)
    #     framein = framein[np.newaxis,:,:,0:1]
    #     x0t,x1t,x2t = multiScaleImages(framein, conf.rescale,  conf.scale, conf.l1_cropsz,conf)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    #     print "WARNING!!!ATTENTION!!! DOING CONTRAST NORMALIZATION!!!!"
    #     print "WARNING!!!ATTENTION!!! DOING CONTRAST NORMALIZATION!!!!"

    allf = np.zeros((bsize,) + conf.imsz + (1,))
    alll = np.zeros((bsize, conf.n_classes, 2))
    for curl in range(nbatches):

        ndxst = curl * bsize
        ndxe = min(nframes, (curl + 1) * bsize)
        ppe = min(ndxe - ndxst, bsize)
        for ii in range(ppe):
            success, framein = cap.read()
            assert success, "Could not read frame"

            framein = crop_images(framein, conf)
            #             framein = clahe.apply(framein[:,:,:1])
            allf[ii, ...] = framein[..., 0:1]

        x0, x1, x2 = multi_scale_images(allf, conf.rescale, conf.scale, conf.l1_cropsz, conf)

        self.feed_dict[self.ph['x0']] = x0
        self.feed_dict[self.ph['x1']] = x1
        self.feed_dict[self.ph['x2']] = x2
        pred_int = sess.run(pred_pair[0], feed_dict=self.feed_dict)
        self.feed_dict[self.ph['fine_pred_in']] = pred_int
        alll[:ppe, ...] = locs[ndxst:ndxe, ...]
        self.feed_dict[self.ph['fine_pred_locs_in']] = alll
        pred = sess.run(pred_pair, self.feed_dict)

        base_locs, fine_locs = get_fine_pred_locs(pred[0], pred[1], conf)
        pred_locs[ndxst:ndxe, :, :] = fine_locs[:ppe, :, :]
        sys.stdout.write('.')
        if curl % 20 == 19:
            sys.stdout.write('\n')

    cap.release()
    return pred_locs


def create_result_image(im, locs, perc, ax = None):
    if ax is None:
        f, ax = plt.subplots()
    ax.imshow(im) if im.ndim == 3 else ax.imshow(im, cmap='gray')
    # ax.scatter(locs[:,0],locs[:,1],s=20)
    cmap = cm.get_cmap('jet')
    rgba = cmap(np.linspace(0, 1, perc.shape[0]))
    for ndx in range(locs.shape[0]):
        for pndx in range(perc.shape[0]):
            ci = plt.Circle(locs[ndx, :], fill=False,
                            radius=perc[pndx, ndx], color=rgba[pndx, :])
            ax.add_artist(ci)


def create_pred_movie(conf, predList, moviename, outmovie, outtype, maxframes=-1):
    predLocs, predscores, predmaxscores = predList
    #     assert false, 'stop here'
    tdir = tempfile.mkdtemp()

    cap = cv2.VideoCapture(moviename)
    nframes = int(cap.get(cvc.FRAME_COUNT))
    if maxframes > 0:
        nframes = maxframes

    cmap = cm.get_cmap('jet')
    rgba = cmap(np.linspace(0, 1, conf.n_classes))

    fig = mpl.figure.Figure(figsize=(9, 4))
    canvas = FigureCanvasAgg(fig)

    if conf.adjustContrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=conf.clahegridsize)
    else:
        clahe = None

    for curl in range(nframes):
        framein = cv2.read()
        framein = crop_images(framein, conf)
        if framein.shape[2] > 1:
            framein = framein[..., 0]

        if conf.adjustContrast:
            framein = clahe.apply(framein)

        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(framein, cmap=cm.gray)
        ax1.scatter(predLocs[curl, :, 0, 0], predLocs[curl, :, 0, 1],  # hold=True,
                    c=cm.hsv(np.linspace(0, 1 - old_div(1., conf.n_classes), conf.n_classes)),
                    s=np.clip(predmaxscores[curl, :, 0] * 100, 20, 40),
                    linewidths=0, edgecolors='face')
        ax1.axis('off')
        ax2 = fig.add_subplot(1, 2, 2)
        if outtype == 1:
            curpreds = predscores[curl, :, :, :, 0]
        elif outtype == 2:
            curpreds = predscores[curl, :, :, :, 0] * 2 - 1

        rgbim = create_pred_image(curpreds, conf.n_classes)
        ax2.imshow(rgbim)
        ax2.axis('off')

        fname = "test_{:06d}.png".format(curl)

        # to printout without X.
        # From: http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html
        # The size * the dpi gives the final image size
        #   a4"x4" image * 80 dpi ==> 320x320 pixel image
        canvas.print_figure(os.path.join(tdir, fname), dpi=160)

        # below is the easy way.
    #         plt.savefig(os.path.join(tdir,fname))

    tfilestr = os.path.join(tdir, 'test_*.png')
    mencoder_cmd = "mencoder mf://" + tfilestr + " -frames " + "{:d}".format(
        nframes) + " -mf type=png:fps=15 -o " + outmovie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
    os.system(mencoder_cmd)
    cap.release()


def create_pred_movie_no_conf(conf, predList, moviename, outmovie, outtype):
    predLocs, predscores, predmaxscores = predList
    #     assert false, 'stop here'
    tdir = tempfile.mkdtemp()

    cap = cv2.VideoCapture(moviename)
    nframes = int(cap.get(cvc.FRAME_COUNT))

    cmap = cm.get_cmap('jet')
    rgba = cmap(np.linspace(0, 1, conf.n_classes))

    fig = mpl.figure.Figure(figsize=(9, 4))
    canvas = FigureCanvasAgg(fig)
    for curl in range(nframes):
        framein = myutils.readframe(cap, curl)
        framein = crop_images(framein, conf)

        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(framein[:, :, 0], cmap=cm.gray)
        ax1.scatter(predLocs[curl, :, 0, 0], predLocs[curl, :, 0, 1],  # hold=True,
                    c=cm.hsv(np.linspace(0, 1 - old_div(1., conf.n_classes), conf.n_classes)),
                    s=20, linewidths=0, edgecolors='face')
        ax1.axis('off')
        ax2 = fig.add_subplot(1, 2, 2)
        if outtype == 1:
            curpreds = predscores[curl, :, :, :, 0]
        elif outtype == 2:
            curpreds = predscores[curl, :, :, :, 0] * 2 - 1

        rgbim = create_pred_image(curpreds, conf.n_classes)
        ax2.imshow(rgbim)
        ax2.axis('off')

        fname = "test_{:06d}.png".format(curl)

        # to printout without X.
        # From: http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html
        # The size * the dpi gives the final image size
        #   a4"x4" image * 80 dpi ==> 320x320 pixel image
        canvas.print_figure(os.path.join(tdir, fname), dpi=80)

        # below is the easy way.
    #         plt.savefig(os.path.join(tdir,fname))

    tfilestr = os.path.join(tdir, 'test_*.png')
    mencoder_cmd = "mencoder mf://" + tfilestr + " -frames " + "{:d}".format(
        nframes) + " -mf type=png:fps=15 -o " + outmovie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
    os.system(mencoder_cmd)
    cap.release()

#
# def gen_distorted_images(conf, train_type=0):
#     self = PoseTrain.PoseTrain(conf)
#     self.createPH()
#     self.createFeedDict()
#     self.trainType = train_type
#     self.openDBs()
#
#     with tf.Session() as sess:
#         self.createCursors(sess)
#         for _ in range(np.random.randint(50)):
#             self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
#         self.closeCursors()
#
#     orig_img = self.xs
#     dist_img = self.feed_dict[self.ph['x0']]
#     return orig_img, dist_img, self.locs
#

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        #         tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

def db_info(self, dbType='val',train_type=0):
    self.init_train(train_type=train_type)
    self.pred = self.create_network()
    self.create_saver()
    val_info = []
    if train_type is 1:
        fname = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename + '.tfrecords')
    else:
        if dbType is 'val':
            fname = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        else:
            fname = os.path.join(self.conf.cachedir, self.conf.trainfilename + '.tfrecords')
    num_val = count_records(fname)

    with tf.Session() as sess:
        start_at = self.init_and_restore(sess, True, ['loss', 'dist'])

        for step in range(num_val / self.conf.batch_size):
            if dbType is 'val':
                self.setup_val(sess)
            else:
                self.setup_train(sess)
            val_info.append(self.info)

    tf.reset_default_graph()
    return np.array(val_info).reshape([-1,2])


def analyze_gradients(loss, exclude, sess):
    var = tf.global_variables()
    tvar = []
    for vv in var:
        ix = max(map(vv.name.find,exclude))
        if ix < 0:
            tvar.append(vv)
    var = tvar
    gg = tf.gradients(loss,var)
    return gg, var


def count_records(filename):
    num = 0
    for record in tf.python_io.tf_record_iterator(filename):
        num += 1
    return num

def show_stack(im_s,xx,yy,cmap='gray'):
    pad_amt = xx*yy - im_s.shape[0]
    if pad_amt > 0:
        im_s = np.concatenate([im_s,im_s[:pad_amt,...]],axis=0)
    isz1 = im_s.shape[1]
    isz2 = im_s.shape[2]
    im_s = im_s.reshape([xx,yy,isz1, isz2])
    im_s = im_s.transpose([0, 2, 1, 3])
    im_s = im_s.reshape([xx * isz1, yy * isz2])
    plt.figure(); plt.imshow(im_s,cmap=cmap)


def show_result(ims, ndx, locs, predlocs= None):
    count = float(len(ndx))
    yy = np.ceil(np.sqrt(count/12)*4).astype('int')
    xx = np.ceil(count/yy).astype('int')
    f,ax = plt.subplots(xx,yy,figsize=(16,12),sharex=True,sharey=True)
    ax = ax.flatten()
    cmap = cm.get_cmap('jet')
    rgba = cmap(np.linspace(0, 1, locs.shape[1]))
    for idx in range(count):
        if ims.shape[3] == 1:
            ax[idx].imshow(ims[ndx[idx],:,:,0],cmap='gray')
        else:
            ax[idx].imshow(ims[ndx[idx],...])

        ax[idx].scatter(locs[ndx[idx],:,0],locs[ndx[idx],:,1],c=rgba,marker='.')
        if predlocs is not None:
            ax[idx].scatter(predlocs[ndx[idx], :, 0], predlocs[ndx[idx], :, 1],
                            c=rgba, marker='+')


def output_graph(logdir, sess):
    # sess = tf.get_default_session()
    train_writer = tf.summary.FileWriter(
        logdir,sess.graph)
    train_writer.add_summary(tf.Summary())


def get_timestamps(conf, info):
    L = h5py.File(conf.labelfile)
    pts = L['labeledposTS']
    ts_array  = []
    for ndx in range(pts.shape[1]):
        idx = np.array(L[pts[0, ndx]]['idx'])[0, :].astype('int') - 1
        val = np.array(L[pts[0, ndx]]['val'])[0, :] - 1
        sz = np.array(L[pts[0, ndx]]['size'])[:, 0].astype('int')
        Y = np.zeros(sz).flatten()
        Y[idx] = val
        Y = Y.reshape(np.flipud(sz))
        ts_array.append(Y)

    ts = np.zeros(info.shape[0:1])
    for ndx in range(info.shape[0]):
        cur_exp = info[ndx, 0].astype('int')
        cur_t = info[ndx,1].astype('int')
        cur_ts = ts_array[cur_exp][:,cur_t,:].max()
        ts[ndx] = cur_ts

    return ts


def tfrecord_to_coco(db_file, conf, img_dir, out_file, categories=None, scale = 1):

    # alice example category
    skeleton = [ [1,2],[1,3],[2,5],[3,4],[1,6],[6,7],[6,8],[6,10],[8,9],[10,11],[5,12],[9,13],[6,14],[6,15],[11,16],[4,17]]
    names = ['head','lneck','rneck','rshld','lshld','thrx','tail','lelb','lmid','relb','rmid','lfront','lmid','lrear','rrear','rmid','rfront']
    categories = [{'id': 1, 'skeleton': skeleton, 'keypoints': names, 'super_category': 'fly', 'name': 'fly'}]

    queue = tf.train.string_input_producer([db_file])
    data = multiResData.read_and_decode(queue, conf)
    n_records = count_records(db_file)

    bbox = [0,0,0,conf.imsz[0],conf.imsz[1],conf.imsz[0],conf.imsz[1],0]*scale
    area = conf.imsz[0]*conf.imsz[1]*scale*scale
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ann = {'images':[], 'info':[], 'annotations':[],'categories':categories}
        for ndx in range(n_records):
            cur_im, cur_locs, cur_info = sess.run(data)
            if cur_im.shape[2] == 1:
                cur_im = cur_im[:,:,0]
            if scale is not 1:
                cur_im = transform.resize(cur_im, cur_im.shape[:2]*scale, preserve_range= True)
                cur_locs = scale*cur_locs
            im_name = '{:012d}.png'.format(ndx)
            misc.imsave(os.path.join(img_dir, im_name),cur_im)

            ann['images'].append({'id':ndx, 'width':conf.imsz[1]*scale, 'height':conf.imsz[0]*scale, 'file_name':im_name})
            ann['annotations'].append({'iscrowd':0,'segmentation':[bbox],'area':area,'image_id':ndx, 'id':ndx,'num_keypoints':conf.n_classes,'bbox':bbox,'keypoints':cur_locs.flatten().tolist(),'category_id':1})



        coord.request_stop()
        coord.join(threads)
    with open(out_file,'w') as f:
        json.dump(ann, f)
    # code to show skeleton.
    # plt.figure();
    # plt.imshow(cur_im, 'gray')
    # for b in skeleton:
    #     a = np.array(b) - 1
    #     plt.plot(cur_locs[a, 0], cur_locs[a, 1])


def create_imseq(ims, reverse=False,val_func=np.mean,sat_func=np.std):
    n_classes = ims.shape[0]
    ims = ims.astype('float')
    out_im = np.zeros(ims.shape[1:3] + (3,))
    if not reverse:
        out_im[:, :, 0] = np.argmax(ims, 0).astype('float32') / n_classes * 180
    else:
        out_im[:, :, 0] = np.argmin(ims, 0).astype('float32') / n_classes * 180

    zz = sat_func(ims,axis=0)
    out_im[:, :, 1] = zz/zz.max()
    out_im[:, :, 2] = val_func(ims,axis=0)
    out_im = np.clip(out_im, 0, 255)
    out_im = out_im.astype('uint8')
    return cv2.cvtColor(out_im, cv2.COLOR_HSV2RGB)

def crop_to_size(img, sz):
    # crops image to sz.
    new_sz = img.shape[:2]
    dx = sz[1] - new_sz[1]
    dy = sz[0] - new_sz[0]
    out_img = np.zeros(sz).astype(img.dtype)
    if dx < 0 or dy < 0:
        hdx = -int(dx/2)
        hdy = -int(dy/2)
        out_img[:,:,...] = img[hdy:(sz[0]+hdy),hdx:(sz[1]+hdx),...]
    else:
        hdx = int(dx/2)
        hdy = int(dy/2)
        out_img[hdy:(new_sz[0] + hdy), hdx:(new_sz[1] + hdx), ...] = img
    return out_img, dx, dy



def preprocess_ims(ims, in_locs, conf, distort, scale):
#    assert ims.dtype == 'uint8', 'Preprocessing only work on uint8 images'
    locs = in_locs.copy()
    cur_im = ims.copy()
    cur_im = cur_im.astype('uint8')
    xs = adjust_contrast(cur_im, conf)
    xs, locs = scale_images(xs, locs, scale, conf)
    if distort:
        if conf.horzFlip:
            xs, locs = randomly_flip_lr(xs, locs)
        if conf.vertFlip:
            xs, locs = randomly_flip_ud(xs, locs)
        xs, locs = randomly_scale(xs, locs, conf)
        xs, locs = randomly_rotate(xs, locs, conf)
        xs, locs = randomly_translate(xs, locs, conf)
        xs = randomly_adjust(xs, conf)
    # xs = adjust_contrast(xs, conf)
    xs = normalize_mean(xs, conf)
    return xs, locs


def get_datestr():
    return datetime.datetime.now().strftime('%Y%m%d')


def runningInDocker():
    # From https://gist.github.com/anantkamath/623ce7f5432680749e087cf8cfba9b69
    with open('/proc/self/cgroup', 'r') as procfile:
        for line in procfile:
            fields = line.strip().split('/')
            if 'docker' in fields:
                return True

    return False


def json_load(filename):
    with open(filename,'r') as f:
        K = json.load(f)
    return K


def get_last_epoch(conf, name):
    train_data_file = os.path.join(
        conf.cachedir, conf.expname + '_' + name + '_traindata')
    with open(train_data_file + '.json', 'r') as json_file:
        json_data = json.load(json_file)
    return int(json_data['step'][-1])


def get_latest_model_file_keras(conf, name):
    last_epoch = get_last_epoch(conf, name)
    save_epoch = last_epoch
    latest_model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{}'.format(save_epoch))
    if not os.path.exists(latest_model_file):
        save_epoch = int(np.floor(last_epoch/conf.save_step)*conf.save_step)
        latest_model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{}'.format(save_epoch))
    return  latest_model_file


def get_crop_loc(lbl,ndx,view, on_gt=False):
    from APT_interface_mdn import read_entry
    # this is unnecessarily ugly just because matlab.
    if lbl['cropProjHasCrops'][0, 0] == 1:
        nviews = int(read_entry(lbl['cfg']['NumViews']))
        if on_gt:
            fname = 'movieFilesAllGTCropInfo'
        else:
            fname = 'movieFilesAllCropInfo'

        if nviews == 1:
            crop_loc = lbl[lbl[fname][0, ndx]]['roi'].value[:, 0].astype('int')
        else:
            crop_loc = lbl[lbl[lbl[fname][0, ndx]]['roi'][view][0]].value[:, 0].astype('int')
    else:
        crop_loc = None

    return crop_loc

