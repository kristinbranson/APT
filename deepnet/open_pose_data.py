import sys
import os
import numpy as np
import tensorflow as tf
import logging

import matplotlib.pyplot as plt
from itertools import islice
import time

import PoseTools
import heatmap

ISPY3 = sys.version_info >= (3, 0)


def create_affinity_labels(locs, imsz, graph, tubewidth=1.0):
    """
    Create/return part affinity fields

    locs: (nbatch x npts x 2) (x,y) locs, 0-based. (0,0) is the center of the
        upper-left pixel.
    imsz: [2] (nr, nc) size of affinity maps to create/return

    graph: (nlimb) array of 2-element tuples; connectivity/skeleton
    tubewidth: width of "limb". *Warning* maybe don't choose tubewidth exactly equal to 1.0

    returns (nbatch x imsz[0] x imsz[1] x nlimb*2) paf hmaps.
        4th dim ordering: limb1x, limb1y, limb2x, limb2y, ...
    """

    nlimb = len(graph)
    nbatch = locs.shape[0]
    out = np.zeros([nbatch, imsz[0], imsz[1], nlimb * 2])
    n_steps = 2 * max(imsz)

    for cur in range(nbatch):
        for ndx, e in enumerate(graph):
            start_x, start_y = locs[cur, e[0], :]
            end_x, end_y = locs[cur, e[1], :]
            assert not (np.isnan(start_x) or np.isnan(start_y) or np.isnan(end_x) or np.isnan(end_y))
            assert not (np.isinf(start_x) or np.isinf(start_y) or np.isinf(end_x) or np.isinf(end_y))

            ll = np.sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)

            if ll == 0:
                # Can occur if start/end labels identical
                # Don't update out/PAF
                continue

            dx = (end_x - start_x) / ll / 2
            dy = (end_y - start_y) / ll / 2
            zz = None
            TUBESTEP = 0.25
            ntubestep = int(2.0 * float(tubewidth) / TUBESTEP + 1)
            # for delta in np.arange(-tubewidth, tubewidth, 0.25):
            for delta in np.linspace(-tubewidth, tubewidth, ntubestep):
                # delta indicates perpendicular displacement from line/limb segment (in px)

                # xx = np.round(np.linspace(start_x,end_x,6000))
                # yy = np.round(np.linspace(start_y,end_y,6000))
                # zz = np.stack([xx,yy])
                xx = np.round(np.linspace(start_x + delta * dy, end_x + delta * dy, n_steps))
                yy = np.round(np.linspace(start_y - delta * dx, end_y - delta * dx, n_steps))
                if zz is None:
                    zz = np.stack([xx, yy])
                else:
                    zz = np.concatenate([zz, np.stack([xx, yy])], axis=1)
                # xx = np.round(np.linspace(start_x-dy,end_x-dy,6000))
                # yy = np.round(np.linspace(start_y+dx,end_y+dx,6000))
                # zz = np.concatenate([zz,np.stack([xx,yy])],axis=1)
            # zz now has all the pixels that are along the line.
            # or "tube" of width tubewidth around limb
            zz = np.unique(zz, axis=1)
            # zz now has all the unique pixels that are along the line with thickness==tubewidth.
            dx = (end_x - start_x) / ll
            dy = (end_y - start_y) / ll
            for x, y in zz.T:
                xint = int(round(x))
                yint = int(round(y))
                if xint < 0 or xint >= out.shape[2] or yint < 0 or yint >= out.shape[1]:
                    continue
                out[cur, yint, xint, ndx * 2] = dx
                out[cur, yint, xint, ndx * 2 + 1] = dy

    return out



# def create_label_images_with_rescale(locs, im_sz, scale, blur_rad):
#     '''
#     Like PoseTools.create_label_images
#     :param locs:
#     :param im_sz:
#     :param scale:
#     :param blur_rad: The blur is in the rescaled units
#     :return:
#     '''
#
#     n_classes = len(locs[0])
#     sz0 = int(im_sz[0] // scale)
#     sz1 = int(im_sz[1] // scale)
#
#     label_ims = np.zeros((len(locs), sz0, sz1, n_classes))
#     k_size = max(int(round(3 * blur_rad)), 1)
#     blur_l = np.zeros([2 * k_size + 1, 2 * k_size + 1])
#     blur_l[k_size, k_size] = 1
#     blur_l = cv2.GaussianBlur(blur_l, (2 * k_size + 1, 2 * k_size + 1), blur_rad)
#     blur_l = old_div(blur_l, blur_l.max())
#     for cls in range(n_classes):
#         for ndx in range(len(locs)):
#             if np.isnan(locs[ndx][cls][0]) or np.isinf(locs[ndx][cls][0]):
#                 continue
#             if np.isnan(locs[ndx][cls][1]) or np.isinf(locs[ndx][cls][1]):
#                 continue
#             yy = float(locs[ndx][cls][1] - float(scale - 1) / 2) / scale
#             xx = float(locs[ndx][cls][0] - float(scale - 1) / 2) / scale
#             modlocs0 = int(np.round(yy))
#             modlocs1 = int(np.round(xx))
#             l0 = min(sz0, max(0, modlocs0 - k_size))  # min unnec ?
#             r0 = max(0, min(sz0, modlocs0 + k_size + 1))  # max unnec ?
#             l1 = min(sz1, max(0, modlocs1 - k_size))  # etc
#             r1 = max(0, min(sz1, modlocs1 + k_size + 1))
#             label_ims[ndx, l0:r0, l1:r1, cls] = \
#                 blur_l[(l0 - modlocs0 + k_size):(r0 - modlocs0 + k_size),
#                 (l1 - modlocs1 + k_size):(r1 - modlocs1 + k_size)]
#
#     # label_ims = 2.0 * (label_ims - 0.5)
#     label_ims -= 0.5
#     label_ims *= 2.0
#     return label_ims


def rescale_points(locs, scale):
    '''
    Rescale (x/y) points to a lower res

    :param locs: (nbatch x npts x 2) (x,y) locs, 0-based. (0,0) is the center of the upper-left pixel.
    :param scale: downsample factor. eg if 2, the image size is cut in half
    :return: locsrs (nbatch x npts x 2) (x,y) locs, 0-based, rescaled
    '''

    bsize, npts, d = locs.shape
    assert d == 2
    assert issubclass(locs.dtype.type, np.floating)

    locsrs = (locs - float(scale - 1) / 2) / scale

    return locsrs

# tf.data

def create_tf_datasets(conf):
    '''
    Create train/val TFRecordDatasets.

    cf C+P from PoseCommon_dataset

    :return: train_ds1, train_ds2, val_ds
    '''

    def _parse_function(serialized_example):
        '''
        Parse raw images/locs/info from tfrecord example
        :param serialized_example:
        :return:
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
                      'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                      })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, conf.imsz + (conf.img_dim,))
        # assert size matches (height, width, depth)?
        trx_ndx = tf.cast(features['trx_ndx'], tf.int64)
        locs = tf.cast(features['locs'], tf.float32)
        exp_ndx = tf.cast(features['expndx'], tf.float32)
        ts = tf.cast(features['ts'], tf.float32)
        info = tf.stack([exp_ndx, ts, tf.cast(trx_ndx, tf.float32)])
        return image, locs, info

    def _preproc(ims, locs, info, distort):
        '''
        Apply PoseTools.preprocess_ims
        :param ims:
        :param locs:
        :param info:
        :return:
        '''
        assert conf.op_rescale == 1  # maybe multiple issues with op_rescale>1 (here+downstream)
        assert conf.op_label_scale == 8, "Expected openpose scale of 8"  # Any value should be ok tho

        ims, locs = PoseTools.preprocess_ims(ims, locs, conf, distort, conf.op_rescale)

        (imnr_use, imnc_use) = conf.imszuse
        ims = ims[:, 0:imnr_use, 0:imnc_use, :]

        return ims.astype('float32'), locs.astype('float32'), info.astype('float32')

    def _preproc_with_distort(ims, locs, info):
        return _preproc(ims, locs, info, True)

    def _preproc_no_distort(ims, locs, info):
        return _preproc(ims, locs, info, False)

    def _preproc_create_maps_pafs(locs):
        '''
        :param ims:
        :param locs:
        :param info:
        :return:
        '''

        assert conf.op_label_scale == 8, "Expected openpose scale of 8"  # Any value should be ok tho

        # if self.conf.img_dim == 1:
        #     assert ims.shape[-1] == 1, "Expected image depth of 1"
        #     ims = np.tile(ims, 3)

        locs_lores = rescale_points(locs, conf.op_label_scale)
        assert conf.op_rescale == 1  # etc
        imsz_lores = [int(x / conf.op_label_scale / conf.op_rescale) for x in conf.imszuse]
        label_map_lores = heatmap.create_label_hmap(locs_lores, imsz_lores, conf.op_map_lores_blur_rad)

        # label_ims = create_label_images(locs / self.conf.op_label_scale, mask_sz, 1)  # self.conf.label_blur_rad)
        # label_ims = PoseTools.create_label_images(locs/self.conf.op_label_scale, mask_sz,1,2)
        # label_ims = np.clip(label_ims, 0, 1)  # AL: possibly unnec?

        # label_ims_origres = create_label_images(locs, mask_sz_origres,
        #                                         self.conf.label_blur_rad)
        # label_ims_origres = np.clip(label_ims_origres, 0, 1)  # AL: possibly unnec?

        label_paf_lores = create_affinity_labels(locs_lores, imsz_lores, conf.op_affinity_graph,
                                                 tubewidth=conf.op_paf_lores_tubewidth)

        # affinity_ims = create_affinity_labels(locs / self.conf.op_label_scale,
        #                                       mask_sz, self.conf.op_affinity_graph, 1)  # self.conf.label_blur_rad)
        #
        # affinity_ims_origres = create_affinity_labels(locs,
        #                                               mask_sz_origres,
        #                                               self.conf.op_affinity_graph,
        #                                               self.conf.label_blur_rad)

        # return [ims, mask_im1, mask_im2, mask_im1_origres, mask_im2_origres], \
        #        [affinity_ims, label_ims,
        #         affinity_ims, label_ims,
        #         affinity_ims, label_ims,
        #         affinity_ims, label_ims,
        #         affinity_ims, label_ims,
        #         affinity_ims_origres, label_ims_origres]
        # (inputs, targets)

        return label_paf_lores.astype('float32'), label_map_lores.astype('float32')

    def create_tfrecord_ds(dbfilename, dataaugfunc, numparcalls1, numparcalls2, doshuffle):
        ds = tf.data.TFRecordDataset(dbfilename)
        ds = ds.map(map_func=_parse_function, num_parallel_calls=numparcalls1)
        ds = ds.repeat()
        if doshuffle:
            ds = ds.shuffle(buffer_size=100)
        ds = ds.batch(conf.batch_size)

        def dataAugPyFunc(ims, locs, info):
            return tuple(tf.py_func(dataaugfunc, [ims, locs, info], [tf.float32, ] * 3))

        def createTargetsPyFunc(ims, locs, info):
            return tuple(tf.py_func(_preproc_create_maps_pafs, [locs], [tf.float32, ]*2))

        # augmented data, images cropped to mod 8: (ims, locs, info)
        dsaug = ds.map(map_func=dataAugPyFunc, num_parallel_calls=numparcalls2)

        # This is exceedingly dumb, but py_func can't handle fns with nested outputs?
        # Ultimately for tf.keras.Model.fit() need a dataset that produces (inp, (out1, out2,...))
        dsaugim = dsaug.map(map_func=lambda ims, locs, info: ims, num_parallel_calls=numparcalls2)
        dsauglocsinfo = dsaug.map(map_func=lambda ims, locs, info: (locs, info), num_parallel_calls=numparcalls2)
        dspafsmaps = dsaug.map(map_func=createTargetsPyFunc, num_parallel_calls=numparcalls2)

        dsModelFit = tf.data.Dataset.zip((dsaugim, dspafsmaps))
        dsModelFit = dsModelFit.prefetch(buffer_size=100)
        dsModelFitMD = dsauglocsinfo.prefetch(buffer_size=100)
        return dsModelFit, dsModelFitMD

    def create_tfrecord_dstoy(dbfilename):
        ds = tf.data.TFRecordDataset(dbfilename)
        ds = ds.map(map_func=_parse_function)
        ds = ds.repeat()
        ds = ds.batch(conf.batch_size)

        def dataAugPyFunc(ims, locs, info):
            return tuple(tf.py_func(_preproc_with_distort, [ims, locs, info], [tf.float32, ] * 3))

        def createTargetsPyFunc(ims, locs, info):
            return tuple(tf.py_func(_preproc_create_maps_pafs, [locs], [tf.float32, ] * 2))

        # augmented data, images cropped to mod 8: (ims, locs, info)
        dsaug = ds.map(map_func=dataAugPyFunc)

        ds1 = dsaug.map(map_func=lambda ims, locs, info: ims)
        ds2 = dsaug.map(map_func=lambda ims, locs, info: locs)
        ds3 = ds1.map(map_func=lambda x:x)
        #dsauglocsinfo = dsaug.map(map_func=lambda ims, locs, info: (locs, info))
        #dspafsmaps = dsaug.map(map_func=createTargetsPyFunc)

        return ds1, ds2, ds3 #, dsauglocsinfo, dspafsmaps

    # out = self.convert_locs_to_targets(locs)
    # # Return the results as float32.
    # out_32 = [o.astype('float32') for o in out]
    # return [ims.astype('float32'), locs.astype('float32'), info.astype('float32')] + out_32

    train_db = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
    val_db = os.path.join(conf.cachedir, conf.valfilename) + '.tfrecords'
    if os.path.exists(val_db) and os.path.getsize(val_db) > 0:
        logging.info("Val DB exists: Data for validation from:{}".format(val_db))
    else:
        logging.warning("Val DB does not exist: Data for validation from:{}".format(train_db))
        val_db = train_db

    if False:
        return create_tfrecord_dstoy(train_db)
    else:
        print "XXX NO SHUFFLE, NO DISTORT, was 5 and 8 parcalls"
        dstrn, dstrnMD = create_tfrecord_ds(train_db, _preproc_with_distort, 5, 8, True)
        #train_ds2 = create_tfrecord_ds(train_db, _preproc_with_distort, 5, 8, True)

        print "XXX was 2 and 4 parcalls"
        dsval, dsvalMD  = create_tfrecord_ds(val_db, _preproc_no_distort, 2, 4, False)

        return dstrn, dstrnMD, dsval, dsvalMD

def test_dataset_with_rand():
    def fn1(x):
        print "foo"
        z = tf.random_uniform((),seed=42,name='randfoo')
        return z

    # tf.set_random_seed(42)
    # np.random.seed(42)

    d1 = tf.data.Dataset.from_tensor_slices(tf.zeros(10))
    d2 = d1.map(fn1)

    d3 = d2.map(lambda x: tf.identity(x,name='idfoo'))
    d4 = tf.data.Dataset.zip((d2, d2))
    return d2, d3, d4

def viz_dataset_contents(datavals, i, ibatch, op_af_graph, locvals=None, mdvals=None,):
    '''

    :param datavals: lists of ims, (pafs, maps)
    :param i: index into x
    :param ibatch: batch index
    :param mdvals: lists of info/metadata
    :param locvals: lists of locs
    :return:
    '''

    ims, pafsmaps = datavals[i]
    assert len(ims) == 1, "Expected single-el list"
    ims = ims[0]
    pafs = pafsmaps[-2]
    maps = pafsmaps[-1]

    haslocs = locvals is not None
    hasmd = mdvals is not None
    if haslocs:
        locs = locvals[i]
    if hasmd:
        info = mdvals[i]

    bsize = ims.shape[0]
    print "Batch size is {}".format(bsize)

    fig1, axs1 = plt.subplots(nrows=3, ncols=6, num=1)
    fig2, axs2 = plt.subplots(nrows=3, ncols=6, num=2)
    axs1.shape = (18,)
    axs2.shape = (18,)

    plt.sca(axs1[0])
    plt.cla()
    plt.imshow(ims[ibatch, :, :, 0])
    if haslocs:
        plt.scatter(locs[ibatch, :, 0], locs[ibatch, :, 1], c='r', marker='.')
        print locs[ibatch, :, :]
    if hasmd:
        ttlstr = "{:.2f}/{:.2f}/{:.2f}".format(info[ibatch, 0], info[ibatch, 1], info[ibatch, 2])
        plt.title(ttlstr)
        print info[ibatch, :]

    for ipt in range(17):
        plt.sca(axs1[1+ipt])
        plt.cla()
        plt.imshow(maps[ibatch, :, :, ipt])
        ttlstr = "pt{:.2f}".format(ipt)
        plt.title(ttlstr)
        if haslocs:
            locs_lores = rescale_points(locs, 8)
            plt.scatter(locs_lores[ibatch, ipt, 0], locs_lores[ibatch, ipt, 1], c='r', marker='.')
    plt.show()

    nlimb = pafs.shape[3]/2
    print "paf nlimbs={}".format(nlimb)
    assert nlimb == len(op_af_graph)

    for ilimb in range(nlimb):
        ilimbx = 2*ilimb
        ilimby = 2*ilimb + 1
        iaxx = ilimbx+1
        iaxy = ilimby+1

        limbspec = op_af_graph[ilimb]
        limbpt0, limbpt1 = limbspec

        plt.sca(axs2[iaxx])
        plt.cla()
        plt.imshow(pafs[ibatch, :, :, ilimbx])
        if haslocs:
            plt.scatter(locs_lores[ibatch, limbpt0, 0], locs_lores[ibatch, limbpt0, 1], c='r', marker='.')
            plt.scatter(locs_lores[ibatch, limbpt1, 0], locs_lores[ibatch, limbpt1, 1], c='r', marker='x')
        pafxun = set(np.unique(pafs[ibatch, :, :, ilimbx]))
        pafxun.remove(0.0)
        ttlstr = "{} {}: {}".format(limbspec, 'x', tuple(pafxun))
        plt.title(ttlstr)

        plt.sca(axs2[iaxy])
        plt.cla()
        plt.imshow(pafs[ibatch, :, :, ilimby])
        if haslocs:
            plt.scatter(locs_lores[ibatch, limbpt0, 0], locs_lores[ibatch, limbpt0, 1], c='r', marker='.')
            plt.scatter(locs_lores[ibatch, limbpt1, 0], locs_lores[ibatch, limbpt1, 1], c='r', marker='x')
        pafyun = set(np.unique(pafs[ibatch, :, :, ilimby]))
        pafyun.remove(0.0)
        ttlstr = "{} {}: {}".format(limbspec, 'y', tuple(pafyun))
        plt.title(ttlstr)

    plt.show()


class DataIteratorTF(object):

    def __init__(self, conf, db_type, distort, shuffle, debug=False):
        self.conf = conf
        if db_type == 'val':
            filename = os.path.join(self.conf.cachedir, self.conf.valfilename) + '.tfrecords'
        elif db_type == 'train':
            filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + '.tfrecords'
        else:
            raise IOError('Unspecified DB Type')  # KB 20190424 - py3
        self.file = filename
        self.iterator = None
        self.distort = distort
        self.shuffle = shuffle
        self.batch_size = self.conf.batch_size
        self.vec_num = len(conf.op_affinity_graph)
        self.heat_num = self.conf.n_classes
        self.N = PoseTools.count_records(filename)
        self.debug = debug

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
        all_info = []
        for b_ndx in range(self.batch_size):
            # AL: this 'shuffle' seems weird
            n_skip = np.random.randint(30) if self.shuffle else 0
            for _ in range(n_skip + 1):
                record = self.read_next()

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
            locs = locs.reshape([self.conf.n_classes, 2])
            if 'trx_ndx' in example.features.feature.keys():
                trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
            else:
                trx_ndx = 0
            info = np.array([expid, t, trx_ndx])

            all_ims.append(reconstructed_img)
            all_locs.append(locs)
            all_info.append(info)

        ims = np.stack(all_ims)  # [bsize x height x width x depth]
        locs = np.stack(all_locs)  # [bsize x ncls x 2]
        info = np.stack(all_info)  # [bsize x 3]

        assert self.conf.op_rescale == 1, \
            "Need further mods/corrections below for op_rescale~=1"
        assert self.conf.op_label_scale == 8, \
            "Expected openpose scale of 8"  # Any value should be ok tho

        ims, locs = PoseTools.preprocess_ims(ims, locs, self.conf,
                                             self.distort, self.conf.op_rescale)
        # locs has been rescaled per op_rescale (but not op_label_scale)

        imszuse = self.conf.imszuse
        (imnr_use, imnc_use) = imszuse
        ims = ims[:, 0:imnr_use, 0:imnc_use, :]

        #if self.conf.img_dim == 1:
        #    assert ims.shape[-1] == 1, "Expected image depth of 1"
        #    ims = np.tile(ims, 3)

        # locs -> PAFs, MAP
        locs_lores = rescale_points(locs, conf.op_label_scale)
        imsz_lores = [int(x / self.conf.op_label_scale / self.conf.op_rescale) for x in imszuse]
        label_map_lores = heatmap.create_label_hmap(locs_lores, imsz_lores, conf.op_map_lores_blur_rad)

        # label_ims = create_label_images(locs / self.conf.op_label_scale, mask_sz, 1)  # self.conf.label_blur_rad)
        # label_ims = PoseTools.create_label_images(locs/self.conf.op_label_scale, mask_sz,1,2)
        # label_ims = np.clip(label_ims, 0, 1)  # AL: possibly unnec?

        # label_ims_origres = create_label_images(locs, mask_sz_origres,
        #                                         self.conf.label_blur_rad)
        # label_ims_origres = np.clip(label_ims_origres, 0, 1)  # AL: possibly unnec?

        label_paf_lores = create_affinity_labels(locs_lores, imsz_lores, conf.op_affinity_graph,
                                                 tubewidth=conf.op_paf_lores_tubewidth)

        npafstg = self.conf.op_paf_nstage
        nmapstg = self.conf.op_map_nstage
        targets = [label_paf_lores,] * npafstg + [label_map_lores,] * nmapstg
        
        if self.debug:
            return [ims], targets, locs, info
        else:
            return [ims], targets
            # (inputs, targets)


        # Don't compute just locs/op_rescale etc
        #mask_sz = [int(x / self.conf.op_label_scale / self.conf.op_rescale) for x in imszuse]
        # mask_sz1 = [self.batch_size,] + mask_sz + [2*self.vec_num]
        # mask_sz2 = [self.batch_size,] + mask_sz + [self.heat_num]
        # mask_im1 = np.ones(mask_sz1)
        # mask_im2 = np.ones(mask_sz2)
        # mask_sz_origres = [int(x/self.conf.op_rescale) for x in imszuse]
        # mask_sz1_origres = [self.batch_size,] + mask_sz_origres + [2*self.vec_num]
        # mask_sz2_origres = [self.batch_size,] + mask_sz_origres + [self.heat_num]
        # mask_im1_origres = np.ones(mask_sz1_origres)
        # mask_im2_origres = np.ones(mask_sz2_origres)


        '''
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


        '''

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


if __name__ == "__main__":
    import nbHG


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

    conf = nbHG.createconf(nbHG.lblbub, nbHG.cdir, 'cvi_outer3_easy__split0', 'bub', 'openpose', 0)
    conf.op_affinity_graph = conf.op_affinity_graph[::2]
    conf.imszuse = (180, 180)
    # dst, dstmd, dsv, dsvmd = create_tf_datasets(conf)
    ditrn = DataIteratorTF(conf, 'train', True, True, debug=True)
    dival = DataIteratorTF(conf, 'val', False, False, debug=True)

    xtrn = [x for x in islice(ditrn,5)]
    xval = [x for x in islice(dival,5)]

    imstrn, pafsmapstrn, locstrn, mdtrn = zip(*xtrn)
    mdlintrn = zip(imstrn, pafsmapstrn)

    imsval, pafsmapsval, locsval, mdval = zip(*xval)
    mdlinval = zip(imsval, pafsmapsval)

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
