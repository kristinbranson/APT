from __future__ import division

import sys
import os
import logging
import cv2
import copy
import numpy as np
import tensorflow
vv = [int(v) for v in tensorflow.__version__.split('.')]
if (vv[0]==1 and vv[1]>12) or vv[0]==2:
    tf = tensorflow.compat.v1
else:
    tf = tensorflow

from itertools import islice
import time

import PoseTools
import heatmap

ISPY3 = sys.version_info >= (3, 0)

ISDPK = False
if ISPY3 and vv[0]==1:
    try:
        import deepposekit as dpk
        ISDPK = True
    except:
        print('deepposekit not available')


logr = logging.getLogger('APT')


def distsquaredpts2limb(zz, startxy, sehat):
    '''
    compute squared distance from pt to line

    :param zz: (2,n) array of coords (float type)
    :param startxy: (2,) array of starting pt in limb
    :param sehat: (2) unit vec pointing from limb0 to limb1

    :return: zzdist1 (2,n) array of squared distance from zz to line thru limb
    '''
    zzrel = zz - startxy[:, np.newaxis]
    zzrelmag2 = zzrel[0, :] ** 2 + zzrel[1, :] ** 2
    # zzrel dot startendhat
    zzrelmag2along = np.square(zzrel[0, :]*sehat[0] + zzrel[1, :]*sehat[1])
    zzdist2 = zzrelmag2 - zzrelmag2along
    return zzdist2


def distsquaredpts2limb2(zz, xs, ys, xe, ye, dse2):
    '''
    Prob better (numerically) version of distsquaredpts2limb
    xs, ys, xe, ye: x/ystart, x/yend
    dse2: (xe-xs)**2 + (ye-ys)**2
    '''

    assert zz.shape[0] == 2

    num = (ye - ys)*zz[0, :] - (xe - xs)*zz[1, :] + xe*ys - ye*xs
    zzdist2 = np.square(num) / dse2
    return zzdist2


def create_affinity_labels(locs, imsz, graph,
                           tubewidth=1.0,
                           tubeblur=False,
                           tubeblursig=None,
                           tubeblurclip=0.05):
    """
    Create/return part affinity fields

    locs: (nbatch x npts x 2) (x,y) locs, 0-based. (0,0) is the center of the
        upper-left pixel.
    imsz: [2] (nr, nc) size of affinity maps to create/return

    graph: (nlimb) array of 2-element tuples; connectivity/skeleton
    tubewidth: width of "limb". 
               - if tubeBlurred=False, the tube has "hard" edges with width==tubewidth.
                 *Warning* maybe don't choose tubewidth exactly equal to 1.0 in this case.
               - if tubeBlurred=True, then the tube has a clipped gaussian perpendicular
                 xsection. The stddev of this gaussian is tubeblursig. The tails of the
                 gaussian are clipped at tubeblurclip. 
                 
    In all cases the paf amplitude is in [0,1] ie the tube maximum is at y=1.

    returns (nbatch x imsz[0] x imsz[1] x nlimb*2) paf hmaps.
        4th dim ordering: limb1x, limb1y, limb2x, limb2y, ...
    """

    if tubeblur:
        # tubewidth ignored, tubeblursig must be set
        assert tubeblursig is not None, "tubeblursig must be set"
        # tube radius (squared) corresponding to clip limit tubeblurblip
        tuberadsq = -2.0 * tubeblursig**2 * np.log(tubeblurclip)
        tuberad = np.sqrt(tuberadsq)
        tubewidth = 2.0 * tuberad
        # only pixels within tuberad of the limb segment will fall inside clipping range
    else:
        if tubeblursig is not None:
            pass
            #logr.warning('Tubeblur is False; ignoring tubeblursig value')
        tuberad = tubewidth / 2.0

    locs = locs.copy()
    if locs.ndim == 3:
        locs = locs[:,np.newaxis,...]

    nlimb = len(graph)
    nbatch = locs.shape[0]
    out = np.zeros([nbatch, imsz[0], imsz[1], nlimb * 2])
    n_steps = 2 * max(imsz)

    for cur in range(nbatch):
        for mndx in range(locs.shape[1]):
            for ndx, e in enumerate(graph):
                startxy = locs[cur, mndx, e[0], :]
                start_x, start_y = locs[cur, mndx, e[0], :]
                end_x, end_y = locs[cur, mndx, e[1], :]
                assert not (np.isnan(start_x) or np.isnan(start_y) or np.isnan(end_x) or np.isnan(end_y))
                assert not (np.isinf(start_x) or np.isinf(start_y) or np.isinf(end_x) or np.isinf(end_y))
                if (start_x < -1000) or (start_y < -1000):
                    # multi labeled animal that are not labeled
                    continue

                ll2 = (start_x - end_x) ** 2 + (start_y - end_y) ** 2
                ll = np.sqrt(ll2)

                if ll == 0:
                    # Can occur if start/end labels identical
                    # Don't update out/PAF
                    continue

                costh = (end_x - start_x) / ll
                sinth = (end_y - start_y) / ll
                zz = None
                TUBESTEP = 0.25 # seems like overkill (smaller than nec)
                ntubestep = int(np.ceil(tubewidth / TUBESTEP + 1))
                for delta in np.linspace(-tuberad, tuberad, ntubestep):
                    # delta indicates perpendicular displacement from line/limb segment (in px)

                    xx = np.round(np.linspace(start_x + delta * sinth, end_x + delta * sinth, n_steps))
                    yy = np.round(np.linspace(start_y - delta * costh, end_y - delta * costh, n_steps))
                    if zz is None:
                        zz = np.stack([xx, yy])
                    else:
                        zz = np.concatenate([zz, np.stack([xx, yy])], axis=1)
                # zz now has all the pixels that are along the line.
                # or "tube" of width tubewidth around limb
                zz = np.unique(zz, axis=1)
                # zz now has all the unique pixels that are along the line with thickness==tubewidth.
                # zz shape is (2, n)
                # zz is rounded, representing px centers; startxy is not rounded
                if tubeblur:
                    # since zz is rounded, some points in zz may violate tubeblurclip.
                    zzdist2 = distsquaredpts2limb2(zz, start_x, start_y, end_x, end_y, ll2)
                    w = np.exp(-zzdist2/2.0/tubeblursig**2)
                    # tfwsmall = w < tubeblurclip
                    # if np.any(tfwsmall):
                    #     print "Small w vals: {}/{}".format(np.count_nonzero(tfwsmall), tfwsmall.size)
                    #     print w[tfwsmall]
                    assert zz.shape[1] == w.size

                    for i in range(w.size):
                        x, y = zz[:, i]
                        xint = int(round(x))  # should already be rounded
                        yint = int(round(y))  # etc
                        if xint < 0 or xint >= out.shape[2] or yint < 0 or yint >= out.shape[1]:
                            continue
                        out[cur, yint, xint, ndx * 2] = w[i] * costh
                        out[cur, yint, xint, ndx * 2 + 1] = w[i] * sinth

                else:
                    for x, y in zz.T:
                        xint = int(round(x)) # already rounded?
                        yint = int(round(y)) # etc
                        if xint < 0 or xint >= out.shape[2] or yint < 0 or yint >= out.shape[1]:
                            continue
                        out[cur, yint, xint, ndx * 2] = costh
                        out[cur, yint, xint, ndx * 2 + 1] = sinth

    return out

def parse_record(record, npts):
    example = tf.train.Example()
    example.ParseFromString(record)
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    depth = int(example.features.feature['depth'].int64_list.value[0])
    expid = int(example.features.feature['expndx'].float_list.value[0])
    t = int(example.features.feature['ts'].float_list.value[0])
    img_string = example.features.feature['image_raw'].bytes_list.value[0]
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, depth))
    locs = np.array(example.features.feature['locs'].float_list.value)
    locs = locs.reshape([npts, 2])
    if 'trx_ndx' in example.features.feature.keys():
        trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
    else:
        trx_ndx = 0
    info = np.array([expid, t, trx_ndx])

    return reconstructed_img, locs, info

def parse_record_multi(record, npts, n_max,apply_mask=False):
    example = tf.train.Example()
    example.ParseFromString(record)
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    depth = int(example.features.feature['depth'].int64_list.value[0])
    ntgt = int(example.features.feature['ntgt'].int64_list.value[0])
    expid = int(example.features.feature['expndx'].float_list.value[0])
    t = int(example.features.feature['ts'].float_list.value[0])
    img_string = example.features.feature['image_raw'].bytes_list.value[0]
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, depth))
    mask_string = example.features.feature['mask'].bytes_list.value[0]
    mask_1d = np.fromstring(mask_string, dtype=np.uint8)
    mask = mask_1d.reshape((height, width))
    if apply_mask:
        reconstructed_img = reconstructed_img * mask[...,np.newaxis]
    locs = np.array(example.features.feature['locs'].float_list.value)
    locs_in = locs.reshape([ntgt, npts, 2])
    locs = np.ones([n_max,npts,2])*np.nan
    locs[:ntgt] = locs_in

    if 'trx_ndx' in example.features.feature.keys():
        trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
    else:
        trx_ndx = 0
    info = np.array([expid, t, trx_ndx])

    return reconstructed_img, locs, info, mask


def pad_ims_blur(ims, locs, pady, padx):
    # Similar to PoseTools.pad_ims

    pady_b = pady//2 # before
    padx_b = padx//2
    pady_a = pady-pady_b # after
    padx_a = padx-padx_b
    zz = np.pad(ims, [[0, 0], [pady_b, pady_a], [padx_b, padx_a], [0, 0]], mode='edge')
    wt_im = np.ones(ims[0, :, :, 0].shape)
    wt_im = np.pad(wt_im, [[pady_b, pady_a], [padx_b, padx_a]], mode='linear_ramp')
    out_ims = zz.copy()
    for ex in range(ims.shape[0]):
        for c in range(ims.shape[3]):
            aa = cv2.GaussianBlur(zz[ex, :, :, c], (15, 15), 5)
            aa = aa * (1 - wt_im) + zz[ex, :, :, c] * wt_im
            out_ims[ex, :, :, c] = aa

    out_locs = locs.copy()
    out_locs[..., 0] += padx_b
    out_locs[..., 1] += pady_b
    return out_ims, out_locs

def pad_ims_edge(ims, locs, pady, padx):
    # Similar to PoseTools.pad_ims

    pady_b = pady//2 # before
    padx_b = padx//2
    pady_a = pady-pady_b # after
    padx_a = padx-padx_b
    out_ims = np.pad(ims, [[0, 0], [pady_b, pady_a], [padx_b, padx_a], [0, 0]], mode='edge')
    out_locs = locs.copy()
    out_locs[..., 0] += padx_b
    out_locs[..., 1] += pady_b
    return out_ims, out_locs

def pad_ims_black(ims, locs, pady, padx):
    # Similar to PoseTools.pad_ims

    pady_b = pady//2  # before
    padx_b = padx//2
    pady_a = pady-pady_b # after
    padx_a = padx-padx_b
    out_ims = np.pad(ims, [[0, 0], [pady_b, pady_a], [padx_b, padx_a], [0, 0]], mode='constant')
    out_locs = locs.copy()
    out_locs[..., 0] += padx_b
    out_locs[..., 1] += pady_b
    return out_ims, out_locs

def ims_locs_preprocess_openpose(imsraw, locsraw, conf, distort, gen_target_hmaps=True,mask=None):
    '''
    Openpose; Preprocess ims/locs; generate targets
    :param ims:
    :param locs:
    :param conf:
    :param distort:
    :return:
    '''

    assert conf.op_label_scale == 8, \
        "Expected openpose scale of 8"  # Any value should be ok tho. this is the BB output scale

    assert conf.imsz == imsraw.shape[1:3], "Image size is {} and conf size is {}".format(imsraw.shape[1:],conf.imsz)

    imspad, locspad = pad_ims_black(imsraw, locsraw, conf.op_im_pady, conf.op_im_padx)
    assert imspad.shape[1:3] == conf.op_imsz_pad

    if conf.is_multi:
        ims, locs, mask = PoseTools.preprocess_ims(imspad, locspad, conf, distort, conf.rescale,mask=mask)
    else:
        ims, locs = PoseTools.preprocess_ims(imspad, locspad, conf, distort, conf.rescale)

    # locs has been rescaled per rescale (but not op_label_scale)

    imszuse = conf.op_imsz_net
    assert ims.shape[1:3] == imszuse
    assert ims.shape[3] == conf.img_dim
    if conf.img_dim == 1:
        ims = np.tile(ims, 3)

    if not gen_target_hmaps:
        if conf.is_multi:
            return ims,locs,mask
        else:
            return ims, locs

    # locs -> PAFs, MAP
    # Generates hires maps here but only used below if conf.op_hires
    dc_scale = 2**conf.op_hires_ndeconv
    scale_hires = conf.op_label_scale / dc_scale  # downsample scale of hires (postDC) relative to network input
    assert scale_hires.is_integer()
    locs_lores = PoseTools.rescale_points(locs, conf.op_label_scale, conf.op_label_scale)  ## bb output/working res
    locs_hires = PoseTools.rescale_points(locs, scale_hires, scale_hires)
    #imsz_lores = [int(x / conf.op_label_scale) for x in imszuse]
    #imsz_hires = [int(x / scale_hires) for x in imszuse]

    label_map_lores = heatmap.create_label_hmap(locs_lores, conf.op_imsz_lores, conf.op_map_lores_blur_rad)
    label_map_hires = heatmap.create_label_hmap(locs_hires, conf.op_imsz_hires, conf.op_map_hires_blur_rad)
    label_paf_lores = create_affinity_labels(locs_lores, conf.op_imsz_lores, conf.op_affinity_graph, tubewidth=conf.op_paf_lores_tubewidth, tubeblur=conf.op_paf_lores_tubeblur, tubeblursig=conf.op_paf_lores_tubeblursig, tubeblurclip=conf.op_paf_lores_tubeblurclip)

    npafstg = conf.op_paf_nstage
    nmapstg = conf.op_map_nstage

    targets = [label_paf_lores, ] * npafstg + [label_map_lores, ] * nmapstg
    if conf.op_hires:
        targets.append(label_map_hires)
    if mask is None:
        mask = np.ones_like(ims[...,0])

    return ims, locs, targets, mask

#__ims_locs_preprocess_sb_has_run__ = False

def ims_locs_preprocess_sb(imsraw, locsraw, conf, distort, gen_target_hmaps=True):
    '''
    Openpose; Preprocess ims/locs; generate targets
    :param ims:
    :param locs:
    :param conf:
    :param distort:
    :param gen_target_hmaps: If false, don't draw hmaps, just return ims, locs
    :return: ims, locs, targets
    '''

    #global __ims_locs_preprocess_sb_has_run__

    assert conf.imsz == imsraw.shape[1:3]

    imspad, locspad = pad_ims_black(imsraw, locsraw, conf.sb_im_pady, conf.sb_im_padx)
    assert imspad.shape[1:3] == conf.sb_imsz_pad

    ims, locs = PoseTools.preprocess_ims(imspad, locspad, conf, distort, conf.rescale)

    imszuse = conf.sb_imsz_net  # network input dims
    (imnr_use, imnc_use) = imszuse
    assert ims.shape[1:3] == imszuse
    assert ims.shape[3] == conf.img_dim
    if conf.img_dim == 1:
        ims = np.tile(ims, 3)

    if not gen_target_hmaps:
        return ims, locs

    assert (imnr_use/conf.sb_output_scale).is_integer() and \
           (imnc_use/conf.sb_output_scale).is_integer(), \
        "Network input size is not even multiple of sb_output_scale"
    imsz_out = [int(x / conf.sb_output_scale) for x in imszuse]
    locs_outres = PoseTools.rescale_points(locs, conf.sb_output_scale, conf.sb_output_scale)
    label_map_outres = heatmap.create_label_hmap(locs_outres,
                                                 imsz_out,
                                                 conf.sb_blur_rad_output_res)
    targets = [label_map_outres,]

#    if not __ims_locs_preprocess_sb_has_run__:
#        logr.info('sb preprocess. sb_out_scale={}, imszuse={}, imszout={}, blurradout={#}'.format(conf.sb_output_scale, imszuse, imsz_out, conf.sb_blur_rad_output_res))
 #       __ims_locs_preprocess_sb_has_run__ = True

    return ims, locs, targets

def imgaug_augment(augmenter, images, keypoints, clip=True):
    '''
    Apply an imgaug augmenter. C+P dpk/TrainingGenerator/augment; in Py3 can prob just call meth directly
    :param augmenter:
    :param images: NHWC
    :param keypoints: B x npts x 2
    :return:
    '''

    assert images.shape[0] == keypoints.shape[0] and keypoints.shape[2] == 2

    images_aug = []
    keypoints_aug = []
    for idx in range(images.shape[0]):
        images_idx = images[idx, None]
        keypoints_idx = keypoints[idx, None]
        augmented_idx = augmenter(images=images_idx, keypoints=keypoints_idx)
        images_aug_idx, keypoints_aug_idx = augmented_idx
        images_aug.append(images_aug_idx)
        keypoints_aug.append(keypoints_aug_idx)

    images_aug = np.concatenate(images_aug)
    keypoints_aug = np.concatenate(keypoints_aug)

    if clip:
        # assume uint8s for now
        images_aug = images_aug.clip(min=0., max=255.)

    return images_aug, keypoints_aug

#__ims_locs_preprocess_dpk_has_run__ = False

def ims_locs_preprocess_dpk_base(imsraw, locsraw, conf, distort,
                                 draw_conf_maps=True, mask=None):
    '''

    :param imsraw:
    :param locsraw:
    :param conf:
    :param distort: Even if distort==False, *some image preproc may be done!!*
        (contrast adjust, intense normalization etc)
    :param draw_conf_maps: draw confidence hmaps
    :return: ims, locs, tgts. If draw_conf_maps==False, tgts are just a copy of locs
    '''

    '''
    conf.imsz: raw im size in TFR
    conf.dpk_imsz_net: after padding, then rescale; size of input to network
    conf.rescale:
    conf.dpk_im_pad*   
    '''
    #global __ims_locs_preprocess_dpk_has_run__

    assert mask is None

    assert conf.imsz == imsraw.shape[1:3]

    imspad, locspad = pad_ims_black(imsraw, locsraw, conf.dpk_im_pady, conf.dpk_im_padx)
    assert imspad.shape[1] == conf.dpk_imsz_pad[0]
    assert imspad.shape[2] == conf.dpk_imsz_pad[1]

    if conf.dpk_use_augmenter:
        # dpk out of the box doesn't do non-distortion img preproc
        # (eg prediction is a raw Keras predict on a cv VideoReader

        assert conf.rescale == 1

        # prob unnec at least if distort==True
        imspad = imspad.copy()
        locspad = locspad.copy()
        if distort:
            augmenter = conf.dpk_augmenter
            assert augmenter is not None
            ims, locs = imgaug_augment(augmenter, imspad, locspad)
        else:
            ims = imspad
            locs = locspad
    else:  # ours/PoseTools
        # Note here even if distort==False ims may be preprocessed

        ims, locs = PoseTools.preprocess_ims(imspad, locspad, conf, distort, conf.rescale)

    imsz_net = conf.dpk_imsz_net
    (imnr_net, imnc_net) = imsz_net
    assert ims.shape[1] == imnr_net
    assert ims.shape[2] == imnc_net
    assert ims.shape[3] == conf.img_dim

    #if conf.img_dim == 1:
    #    ims = np.tile(ims, 3)

    #locs_outres = rescale_points(locs, conf.sb_output_scale)
    #imsz_out = [int(x / conf.sb_output_scale) for x in imszuse]
    #label_map_outres = heatmap.create_label_hmap(locs_outres,
    #                                             imsz_out,
    #                                             conf.sb_blur_rad_output_res)

    if draw_conf_maps:
        y = dpk.utils.keypoints.draw_confidence_maps(
            ims,
            locs,
            graph=conf.dpk_graph,
            output_shape=conf.dpk_output_shape,
            use_graph=conf.dpk_use_graph,
            sigma=conf.dpk_output_sigma
        )
        nmaps = y.shape[-1]
        y *= 255
        if conf.dpk_use_graph:
            y[..., conf.n_classes:] *= conf.dpk_graph_scale  # scale grps, limbs, globals

        if conf.dpk_n_outputs > 1:
            y = [y for _ in range(conf.dpk_n_outputs)]

        targets = y
    else:
        assert conf.dpk_n_outputs == 1
        targets = locs.copy()

#    if not __ims_locs_preprocess_dpk_has_run__:
#        str = 'dpk preproc. distort={}, use_augmenter={}, use_graph={}, graph_scale={}, n_outputs={}'
#        logr.info(str.format(distort, conf.dpk_use_augmenter,
#                     conf.dpk_use_graph, conf.dpk_graph_scale, conf.dpk_n_outputs))
#        __ims_locs_preprocess_dpk_has_run__ = True

    if mask is None:
        mask = np.ones_like(ims[..., 0])

    return ims, locs, targets, mask

def ims_locs_preprocess_dpk(imsraw, locsraw, conf, distort, mask=None):
    return ims_locs_preprocess_dpk_base(imsraw, locsraw, conf, distort,
                                        draw_conf_maps=True, mask=mask)

def ims_locs_preprocess_dpk_noconf_nodistort(imsraw, locsraw, conf, distort):
    # Still can img preproc
    assert distort is False

    #assert conf.dpk_n_outputs == 1, "Unexpected dpk_n_outputs: {}".format(conf.dpk_n_outputs)
    # This assert is done in TGTFR when creating a generator; callsites that call this directly
    # can/should know what they are doing

    return ims_locs_preprocess_dpk_base(imsraw, locsraw, conf, distort,
                                        draw_conf_maps=False)


def ims_locs_preprocess_dummy(imsraw, locsraw, conf, distort):
    return imsraw, locsraw, None

def data_generator(tfrfilename, conf, distort, shuffle, ims_locs_proc_fn,
                   debug=False,
                   infinite=True,
                   instrumented=False,
                   instrumentedname=None):
    '''

    Generator that reads from tfrecords and returns preprocessed batches: ims, tgts etc.

    For infinite generators, batches will always have size conf.batch_size.

    For finite generators, the last batch can be "clipped",

    The targets that are computed/returned depend on the preproc fn.



    :param conf: A deep-copy should be passed in so that the generator is unaffected by chances to conf
    :param db_type:
    :param distort:
    :param shuffle:
    :param ims_locs_proc_fn: fn(ims,locs,conf,distort) and returns ims,locs,targets
    :param debug:
    :return:
    '''

    filename = tfrfilename

    isstr = isinstance(ims_locs_proc_fn, str) if ISPY3 else \
            isinstance(ims_locs_proc_fn, basestring)
    if isstr:
        ims_locs_proc_fn = globals()[ims_locs_proc_fn]

    batch_size = conf.batch_size
    N = PoseTools.count_records(filename)

    if instrumented and (instrumentedname is None):
        instrumentedname = "Unnamed-{}".format(os.path.basename(filename))

    #logr.warning("tfdatagen data gen. file={}, distort/shuf/inf={}/{}/{}, ppfun={}, N={}".format(
    #    filename, distort, shuffle, infinite, ims_locs_proc_fn.__name__, N))

    # Py 2.x workaround nested functions outer variable rebind
    # https://www.python.org/dev/peps/pep-3104/#new-syntax-in-the-binding-outer-scope
    class Namespace:
        pass

    ns = Namespace()
    ns.iterator = None

    def iterator_reset():
        if ns.iterator:
            ns.iterator.close()
        ns.iterator = tf.python_io.tf_record_iterator(filename)

    def iterator_read_next():
        if not ns.iterator:
            ns.iterator = tf.python_io.tf_record_iterator(filename)
        try:
            if ISPY3:
                record = next(ns.iterator)
            else:
                record = ns.iterator.next()
        except StopIteration:
            if infinite:
                iterator_reset()
                if ISPY3:
                    record = next(ns.iterator)
                else:
                    record = ns.iterator.next()
            else:
                raise
        return record

    while True:
        all_ims = []
        all_locs = []
        all_info = []
        all_mask = []
        for b_ndx in range(batch_size):
            # TODO: strange shuffle
            n_skip = np.random.randint(30) if shuffle else 0
            try:
                for _ in range(n_skip + 1):
                    record = iterator_read_next()
            except StopIteration:
                # did not make it to next record for this batch;
                # will only occur if infinite == False
                break

            is_multi = getattr(conf, 'is_multi', False)
            if is_multi:
                recon_img, locs, info, mask = parse_record_multi(record, conf.n_classes,conf.max_n_animals,apply_mask=conf.multi_use_mask)
                all_mask.append(mask)
            else:
                recon_img, locs, info = parse_record(record, conf.n_classes)
            all_ims.append(recon_img)
            all_locs.append(locs)
            all_info.append(info)

        if not all_ims:
            # we couldn't read a single new row anymore; exit generator
            if instrumented:
                logr.warning("tfdatagen:{} returning".format(instrumentedname))
            return

        imsraw = np.stack(all_ims)  # [nread x height x width x depth]
        locsraw = np.stack(all_locs)  # [nread x ncls x 2]
        info = np.stack(all_info)  # [nread x 3]
        if is_multi:
            maskraw = np.stack(all_mask)
        else:
            maskraw = None

        nread = imsraw.shape[0]
        # we read at least one new row, ie nread>0.
        # nread==batch_size typically. nread<batch_size only
        # for the last batch when infinite=false.
        tfclippedbatch = nread < batch_size

        if tfclippedbatch:
            logr.warning("Last batch, size={}. padding for now.".format(nread))

            '''
            AL: PoseTools.preprocess_ims (and downstream) should be insensitive to the bsize, 
            esp eg if imsraw.shape[0] disagrees with conf.batch_size. 
            '''
            nshort = batch_size-nread
            imsraw = np.pad(imsraw, ((0, nshort), (0, 0), (0, 0), (0, 0)),
                            mode='constant')
            locsraw = np.pad(locsraw, ((0, nshort), (0, 0), (0, 0)),
                             mode='constant')
            # info = ... dont pad

        ims, locs, targets, mask = ims_locs_proc_fn(imsraw, locsraw, conf, distort,mask=maskraw)
        if mask is None:
            mask = np.ones_like(ims[...,0])
        # targets should be a list here

        if tfclippedbatch:
            assert ims.shape[0] == batch_size
            assert locs.shape[0] == batch_size
            ims = ims[:nread, ...]
            locs = locs[:nread, ...]
            if isinstance(targets, list):
                for ndx, tgt in enumerate(targets):
                    assert tgt.shape[0] == batch_size
                    targets[ndx] = tgt[:nread, ... ]
            else:
                assert targets.shape[0] == batch_size
                targets = targets[:nread, ...]

        if instrumented:
            logr.warning("tfdatagen:{} yielding {}, ifo[0]={}".format(
                instrumentedname, ims.shape[0], info[0, 0]))

        if debug:
            yield [ims, mask], targets, locs, info
        else:
            yield [ims, mask], targets
            # (inputs, targets)


def make_data_generator(tfrfilename, conf0, distort, shuffle, ims_locs_proc_fn, silent=False,
                        batch_size=None, **kwargs):
    conf = copy.deepcopy(conf0)

    if batch_size is not None:
        conf.batch_size = batch_size
    if not silent:
        logr.warning("tfdatagen makedatagen: {}, distort/shuf={}/{}, ppfun={}, {}".format(
            tfrfilename, distort, shuffle, ims_locs_proc_fn, kwargs))
    return data_generator(tfrfilename, conf, distort, shuffle, ims_locs_proc_fn, **kwargs)

thecounter=0
'''
The TFData/keras/tf sucks log!
These are in tf1.14

Using a tfds as val data in K.fit(), val_steps should be settable to None/default as "run entire ds".
This does not work but setting verbose=0 is a suggested workaround ?!?!?!
https://github.com/tensorflow/tensorflow/issues/28995
maybe fixed in tf1.15

Setting val_iter to "one more than ceil(len(tfds))" causes the entire train to get canceled at epoch
end; but passing ceil(len(tfds)) or less, the valds gets auto-reset/reinitialized whenever necessary.

There is no way to instrument a tfds to know when records are pulled, py_funcs get called etc since
I guess it runs off in C++ or in some other universe.

In general, even as of tf2.2/Apr2020, K is not robust to cases when the len(valds) is not evenly 
divisible by val bsize 
https://github.com/tensorflow/tensorflow/issues/38596, 38165, downstream/refd issues

When using a val set even with K seqs, the last/clipped valbatch (when len(valds) is not evenly
divisible by valbsize) is just skipped rather than run and of course there is no notice/doc of this.

In K cbks, the log dict when using fit_generator gives the correct bsize for both trn and val/test 
steps, in the 'size' field of the logs dict. When using a tfds for val, the 'size' is always 1 (??!!).
'''

def create_tf_datasets(conf0,
                       bsize,  # mandatory arg overrides conf0.batch_size
                       n_outputs,
                       is_val=False,  # True for val, False for trn
                       is_raw=False,  # True for raw, False for preprocessed
                       distort=True,  # applies only if is_raw=True
                       shuffle=True,
                       infinite=True,
                       dobatch=True,
                       drawconf=True,
                       ):
    '''
    Create train/val TFRecordDatasets. This is basically PoseBaseGeneral/create_datasets
    (or PoseCommon_dataset) but we are doing some experimenting/massaging.

    cf C+P from PoseCommon_dataset
    '''

    conf = copy.deepcopy(conf0)
    conf.batch_size = bsize
    conf.dpk_n_outputs = 1  # outputs are handled here rather than in pp methods

    def _parse_function(serialized_example):
        '''
        C+P from PoseBaseGeneral
        '''
        features = tf.parse_single_example(
            serialized_example,
            features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                      'width': tf.FixedLenFeature([], dtype=tf.int64),
                      'depth': tf.FixedLenFeature([], dtype=tf.int64),
                      'trx_ndx': tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
                      'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                      'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                      'ts': tf.FixedLenFeature([], dtype=tf.float32),
                      'image_raw': tf.FixedLenFeature([], dtype=tf.string), #   'occ': tf.FixedLenFeature(shape=[conf.n_classes], default_value=None, dtype=tf.float32),
                      })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        trx_ndx = tf.cast(features['trx_ndx'], tf.int64)
        image = tf.reshape(image, conf.imsz + (conf.img_dim,))

        locs = tf.cast(features['locs'], tf.float32)
        exp_ndx = tf.cast(features['expndx'], tf.float32)
        ts = tf.cast(features['ts'], tf.float32)  # tf.constant([0]); #
        info = tf.stack([exp_ndx, ts, tf.cast(trx_ndx, tf.float32)])
        #occ = tf.cast(features['occ'], tf.bool)
        return image, locs, info # , occ

    def _parse_function_multi(serialized_example):
        max_n = conf.max_n_animals
        features = tf.parse_single_example(serialized_example,
            features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                    'width': tf.FixedLenFeature([], dtype=tf.int64),
                    'depth': tf.FixedLenFeature([], dtype=tf.int64),
                    'trx_ndx': tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
                    'locs': tf.FixedLenFeature(shape=[max_n, conf.n_classes, 2], dtype=tf.float32),
                    'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                    'ts': tf.FixedLenFeature([], dtype=tf.float32),
                    'image_raw': tf.FixedLenFeature([], dtype=tf.string),
                    'mask': tf.FixedLenFeature([], dtype=tf.string),
                    #'occ': tf.FixedLenFeature(shape=[max_n,conf.n_classes], default_value=np.zeros([max_n,conf.n_classes]), dtype=tf.float32),
                    })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, conf.imsz + (conf.img_dim,))
        mask = tf.decode_raw(features['mask'], tf.uint8)
        mask = tf.reshape(mask, conf.imsz)
        image = image*tf.expand_dims(mask,2)

        trx_ndx = tf.cast(features['trx_ndx'], tf.int64)
        locs = tf.cast(features['locs'], tf.float32)
        exp_ndx = tf.cast(features['expndx'], tf.float32)
        ts = tf.cast(features['ts'], tf.float32)  # tf.constant([0]); #
        info = tf.stack([exp_ndx, ts, tf.cast(trx_ndx, tf.float32)])
#        occ = tf.cast(features['occ'],tf.bool)

        return image , locs, info #, occ


    def pp_dpk_conf(imsraw, locsraw):
        ims, locs, tgts, mask = ims_locs_preprocess_dpk_base(imsraw,
                                                       locsraw,
                                                       conf,
                                                       distort,
                                                       draw_conf_maps=True)
        return ims.astype('float32'), tgts.astype('float32')

    def pp_dpk_noconf(imsraw, locsraw):
        ims, locs, tgts, mask = ims_locs_preprocess_dpk_base(imsraw,
                                                       locsraw,
                                                       conf,
                                                       distort,
                                                       draw_conf_maps=False)
        return ims.astype('float32'), tgts.astype('float32')

    train_db = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
    if is_val:
        val_db = os.path.join(conf.cachedir, conf.valfilename) + '.tfrecords'
        if os.path.exists(val_db) and os.path.getsize(val_db) > 0:
            logr.info("Val DB exists: Data for validation from:{}".format(val_db))
        else:
            logr.warning("Val DB does not exist: Data for validation from:{}".format(train_db))
            val_db = train_db
        dbfile = val_db
    else:
        dbfile = train_db

    ds = tf.data.TFRecordDataset(dbfile)
    is_multi = getattr(conf, 'is_multi', False)
    if is_multi:
        ds = ds.map(map_func=_parse_function_multi)
    else:
        numpara = getattr(conf, 'dpk_tfdata_num_para_calls_parse', 5)
        ds = ds.map(map_func=_parse_function, num_parallel_calls=numpara)
        logr.info("num para calls for parse: {}".format(numpara))

    #ds = ds.cache()
    if infinite:
        ds = ds.repeat()
    if shuffle:
        try:
            shufflebsize = conf.dpk_tfdata_shuffle_bsize
        except AttributeError:
            shufflebsize = 5000
            logr.warning("dpk_tfdata_shuffle_bsize not present in conf. using default value of {}".format(
                shufflebsize))

        #ds = ds.shuffle(buffer_size=100)
        ds = ds.shuffle(buffer_size=shufflebsize)
    if dobatch:
        ds = ds.batch(conf.batch_size)
    if is_raw:
        pass
        # raw parse; return image, locs, info, occ
    else:
        # TF issues encountered
        # * set_shape after py_func. https://github.com/tensorflow/tensorflow/issues/24520
        # * ds.map concats lists into Tensors. use a tuple
        #   https://github.com/tensorflow/tensorflow/issues/20698
        numpara = getattr(conf, 'dpk_tfdata_num_para_calls_dataaug', 8)
        logr.info("num para calls for dataaug: {}".format(numpara))
        if drawconf:
            def dataAugPyFunc(ims, locs, info):
                # not sure why we need call to tuple
                ims, tgts = tuple(tf.py_func(pp_dpk_conf, [ims, locs], [tf.float32, ] * 2))
                ims.set_shape([None,] * 4)  # ims
                tgts.set_shape([None,] * 4)  # confmaps/tgts
                tgtslist = tuple([tgts for _ in range(n_outputs)])

                # print("The shape of res[1] is {}".format(res[1].shape))
                return ims, tgtslist

            ds = ds.map(map_func=dataAugPyFunc, num_parallel_calls=numpara)
        else:
            def dataAugPyFunc(ims, locs, info):
                res = tuple(tf.py_func(pp_dpk_noconf, [ims, locs], [tf.float32, ] * 2))
                res[0].set_shape([None, ] * 4)  # ims
                res[1].set_shape([None, ] * 3)  # locs
                return res

            ds = ds.map(map_func=dataAugPyFunc, num_parallel_calls=numpara)
            assert n_outputs == 1

    ds = ds.prefetch(buffer_size=100)
    return ds

def read_ds_idxed(ds, indices):
    it = ds.make_one_shot_iterator()
    nextel = it.get_next()
    c = 0
    res = []
    with tf.Session() as sess:
        while True:
            restmp = sess.run(nextel)
            if c in indices:
                res.append(restmp)
                #print("Got {}".format(c))
            c += 1
            if all([c>x for x in indices]):
                break
    return res

def xylist2xyarr(xylist, xisscalarlist=False, yisscalarlist=False):
    x, y = zip(*xylist)
    if xisscalarlist:
        assert all([len(z) == 1 for z in x])
        x = [z[0] for z in x]
    if yisscalarlist:
        assert all([len(z) == 1 for z in y])
        y = [z[0] for z in y]
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y


def montage(ims0, ims0type='batchlast',
            locs=None, locs2=None, fignum=1, figsize=(25, 25), axes_pad=0.0,
            share_all=True, label_mode='1', cmap='viridis',
            cbclr='g',
            locsmrkr='.', locs2mrkr='x',
            locsmrkrsz=16, locscmap='jet'):
    '''

    :param ims0: [nr x nc x N] (assumed b/w); or [N x nr x nc] see next
    :param ims0type: 'batchlast' or 'batchfirst' (dflt)
    :param locs: [N x npt x 2]
    :param fignum:
    :param figsize:
    :param axes_pad:
    :param share_all:
    :param label_mode:
    :param cmap:
    :param locsmrkr:
    :param locsmrkrsz:
    :return:
    '''
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import mpl_toolkits.axes_grid1 as axg1

    if ims0type == 'batchfirst':
        ims = np.moveaxis(ims0, 0, -1)
        ims = ims[:, :, 0, :]
    else:
        ims = ims0

    nim = ims.shape[2]
    nplotr = int(np.floor(np.sqrt(nim)))
    nplotc = int(np.ceil(nim / nplotr))

    fig = plt.figure(fignum, figsize=figsize)
    grid = axg1.ImageGrid(fig, 111,  # similar to subplot(111)
                          nrows_ncols=(nplotr, nplotc),
                          axes_pad=axes_pad,  # pad between axes in inch.
                          share_all=share_all,
                          label_mode=label_mode,
                          cbar_mode='each',
                          )

    for iim in range(nim):
        him = grid[iim].imshow(ims[..., iim], cmap=cmap)
        him.set_clim(0., 255.)
        cb = grid.cbar_axes[iim].colorbar(him)
        cb.ax.tick_params(color='r')
        plt.setp(plt.getp(cb.ax, 'yticklabels'), color=cbclr)
        if iim == 0:
            cb0 = cb
        if locs is not None:
            assert locs.shape[0] == nim
            jetmap = cm.get_cmap(locscmap)
            rgba = jetmap(np.linspace(0, 1, locs.shape[1]))
            grid[iim].scatter(locs[iim, :, 0], locs[iim, :, 1], c=rgba,
                              marker=locsmrkr, s=locsmrkrsz)
        if locs2 is not None:
            assert locs2.shape[0] == nim
            jetmap = cm.get_cmap(locscmap)
            rgba = jetmap(np.linspace(0, 1, locs2.shape[1]))
            grid[iim].scatter(locs2[iim, :, 0], locs2[iim, :, 1], c=rgba,
                              marker=locs2mrkr, s=locsmrkrsz)

    for iim in range(nim, nplotr * nplotc):
        grid[iim].imshow(np.zeros(ims.shape[0:2]))

    plt.show()
    return fig, grid, cb0

if __name__ == "__main__":

    # class Timer(object):
    #     def __init__(self, name=None):
    #         self.name = name
    #
    #     def __enter__(self):
    #         self.tstart = time.time()
    #
    #     def __exit__(self, type, value, traceback):
    #         if self.name:
    #             print('[%s]' % self.name,)
    #         print('Elapsed: %s' % (time.time() - self.tstart))
    #
    #
    # tf.enable_eager_execution()

    import nbHG
    print("OPD MAIN!")

    #locs = np.array([[5., 10.], [15., 10.], [15., 20.], [10., 15.]], np.float32)
    #locs = locs[np.newaxis, :, :]
    affg = np.array([[0, 1], [1, 2], [1, 3]])
    imsz = (25, 25,)
    paf0 = create_affinity_labels(locs, imsz, affg, tubewidth=0.95)
    paf1 = create_affinity_labels(locs, imsz, affg, tubeblur=True, tubeblursig=0.95)

    conf = nbHG.createconf(nbHG.lblbub, nbHG.cdir, 'cvi_outer3_easy__split0', 'bub', 'openpose', 0)
    #conf.op_affinity_graph = conf.op_affinity_graph[::2]
    conf.imszuse = (192, 192)
    conf.sb_im_padx = 192-181
    conf.sb_im_pady = 192 - 181
    conf.sb_output_scale = 2
    conf.sb_blur_rad_output_res = 1.5
    # dst, dstmd, dsv, dsvmd = create_tf_datasets(conf)
    ditrn = data_generator(conf, 'train', True, True, ims_locs_preprocess_sb, debug=True)
    dival = data_generator(conf, 'val', False, False, ims_locs_preprocess_sb, debug=True)

    xtrn = [x for x in islice(ditrn,5)]
    xval = [x for x in islice(dival,5)]

    #imstrn, pafsmapstrn, locstrn, mdtrn = zip(*xtrn)
    #mdlintrn = zip(imstrn, pafsmapstrn)

    #imsval, pafsmapsval, locsval, mdval = zip(*xval)
    #mdlinval = zip(imsval, pafsmapsval)

    #ds1, ds2, ds3 = create_tf_datasets(conf)
    #
    #
    # if True:
    #     x1 = [x for x in ds1.take(10)]
    #     x2 = [x for x in ds2.take(10)]
    #     x3 = [x for x in ds3.take(10)]
    #     #locs10 = [x for x in dslocsinfo.take(10)]
    # else:
    #     dst10 = [x for x in dst.take(1)]
    #     dst10md = [x for x in dstmd.take(1)]
    #     dsv10 = [x for x in dsv.take(1)]
    #     dsv10md = [x for x in dsvmd.take(1)]

    # N = 100
    # with Timer('tf.data'):
    #     xds = [x for x in dst.take(N)]
    # with Timer('it'):
    #     xit = []
    #     for i in range(N):
    #         xit.append(ditrn.next())



    # ds2,ds3,ds4 = test_dataset_with_rand()


# locs = np.array([[0,0],[0,1.5],[0,4],[0,6.9],[0,7.],[0,7.49],[0,7.51],[1,2],[10,12],[16,16]])
# locs = locs[np.newaxis,:,:]
# imsz = (48, 40)
# locsrs = rescale_points(locs, 8)
# imszrs = (6, 5)
#
# import matplotlib.pyplot as plt
# hm1 = create_label_images_with_rescale(locs,imsz,8,3)
# hm2 = heatmap.create_label_hmap(locsrs, imszrs, 3)
