from __future__ import division
from __future__ import print_function
##
from builtins import range
from past.utils import old_div
import scipy.io as sio
from mpiiConfig import conf
import os
import numpy as np
from scipy import misc
import tensorflow as tf
import pickle

def _int64_feature(value):
    if not isinstance(value,(list,np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    if not isinstance(value,(list,np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def crash():
    vv = old_div(3,0)
    return vv

def createTFRecordMPII():
##
    A = sio.loadmat(conf.mpiifile, struct_as_record=False, squeeze_me=True)
    A = A['RELEASE']

##
    # load the list of validation images for pose hour glass
    with open(conf.validfile) as f:
        content = f.readlines()
    val_images = [x.strip() for x in content]

##

    trainfilename = os.path.join(conf.cachedir, conf.trainfilename)
    valfilename = os.path.join(conf.cachedir, conf.valfilename)
    testfilename = os.path.join(conf.cachedir, conf.testfilename)

    env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
    valenv = tf.python_io.TFRecordWriter(valfilename + '.tfrecords')
    testenv = tf.python_io.TFRecordWriter(testfilename + '.tfrecords')

##

    num_ex =len(A.annolist)
    valcount = 0
    traincount = 0
    testcount = 0

    for ndx in range(num_ex):

        imfile = os.path.join(conf.imdir, A.annolist[ndx].image.name)
        if not os.path.exists(imfile):
            print(A.annolist[ndx].image.name + ' Doesnt exist:{}'.format(ndx))
            continue
        im = misc.imread(imfile)


        if not A.img_train[ndx]:
            cur_env = testenv
            testcount += 1
        else:
            if val_images.count(A.annolist[ndx].image.name):
                cur_env = valenv
                valcount += 1
            else:
                cur_env = env
                traincount += 1

        if isinstance(A.annolist[ndx].annorect, np.ndarray):
            a_list = A.annolist[ndx].annorect
        else:
            a_list = [A.annolist[ndx].annorect, ]

        for ondx,curo in enumerate(a_list):
            if not curo._fieldnames.count('objpos'):
                continue
            if isinstance(curo.objpos,np.ndarray):
                continue
            if not curo.objpos._fieldnames.count('x'):
                continue

            cur_pos = curo.objpos
            scale = curo.scale
            c = [cur_pos.x, cur_pos.y]
            bsz = old_div(round((200 + 100) * scale), 2)
            top = int(c[1] - bsz)
            bot = int(c[1] + bsz)
            left = int(c[0] - bsz)
            right = int(c[0] + bsz)
            ht, wd = im.shape[0:2]
            padsz = int(max([-top, -left, bot - ht, right - wd]))

            if padsz < 0:
                padsz = 0
            padim = np.pad(im, ((padsz, padsz), (padsz, padsz), (0, 0)), 'constant')
            patch = padim[(top + padsz):(bot + padsz), (left + padsz):(right + padsz), :]
            psz_orig = patch.shape[0]
            patch = misc.imresize(patch, conf.imsz)

            zz = np.empty([16, 2])
            zz.fill(np.nan)

            if curo._fieldnames.count('annopoints'):
                if isinstance(curo.annopoints.point,sio.matlab.mio5_params.mat_struct):
                    points = [curo.annopoints.point,]
                else:
                    points = curo.annopoints.point
                for pt in points:
                    id = pt.id
                    zz[id, 0] = pt.x
                    zz[id, 1] = pt.y
                cur_locs = (zz.copy() - [left, top]) / psz_orig * conf.imsz[0]
            else:
                cur_locs = zz


            rows = patch.shape[0]
            cols = patch.shape[1]
            if np.ndim(patch) > 2:
                depth = patch.shape[2]
            else:
                depth = 1

            image_raw = patch.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'locs': _float_feature(cur_locs.flatten()),
                'expndx': _int64_feature(ndx),
                'ts': _int64_feature(ondx),
                'image_raw': _bytes_feature(image_raw)}))
            cur_env.write(example.SerializeToString())

    print('Train:{}, Val:{}, Test:{}'.format(traincount,valcount,testcount))
    testenv.close()
    valenv.close()
    env.close()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'height': tf.FixedLenFeature([], dtype=tf.int64),
                  'width': tf.FixedLenFeature([], dtype=tf.int64),
                  'depth': tf.FixedLenFeature([], dtype=tf.int64),
                  'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                  'ts': tf.FixedLenFeature([], dtype=tf.int64),
                  'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                  'image_raw': tf.FixedLenFeature([], dtype=tf.string)
                  })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    depth = tf.cast(features['depth'], tf.int64)
    image = tf.reshape(image, conf.imsz+(3,))

    locs = tf.cast(features['locs'], tf.float64)
    expndx = tf.cast(features['expndx'],tf.int64)
    ondx = tf.cast(features['ts'],tf.int64)

    return image, locs, [expndx, ondx]

##
#
# ndx = np.random.randint(numex)
# im = misc.imread(os.path.join(conf.imdir,A.annolist[ndx].image.name))
# if isinstance(A.annolist[ndx].annorect,np.ndarray):
#     curo = A.annolist[ndx].annorect[0]
# else:
#     curo = A.annolist[ndx].annorect
# cur_pos = curo.objpos
# scale = curo.scale
# c = [cur_pos.x, cur_pos.y]
#
# f = plt.figure(1)
# ax = f.add_subplot(2,1,1)
# ax.clear()
# ax.imshow(im)
# ax.scatter(c[0],c[1])
# bsz = round((200+200)*scale)/2
# top = int(c[1] - bsz)
# bot = int(c[1] + bsz)
# left = int(c[0] - bsz)
# right = int(c[0] + bsz)
# ax.plot([left,right,right,left,left],[top,top,bot,bot,top],c='r')
# zz = np.empty([16,2])
# zz.fill(np.nan)
# for pt in curo.annopoints.point:
#     id = pt.id
#     zz[id,0] = pt.x
#     zz[id,1] = pt.y
# ax.scatter(zz[:,0],zz[:,1])
# #
# ht,wd = im.shape[0:2]
# padsz = int(max([-top,-left,bot-ht,right-wd]))
#
# if padsz<0:
#     padsz = 0
# padim = np.pad(im,((padsz,padsz),(padsz,padsz),(0,0)),'constant')
# patch = padim[(top+padsz):(bot+padsz),(left+padsz):(right+padsz),:]
# psz_orig = patch.shape[0]
# patch = misc.imresize(patch,[256,256])
# ax = f.add_subplot(2,1,2)
# ax.clear()
# ax.imshow(patch)
# zz_new = (zz.copy() - [left,top])/psz_orig*256
# ax.scatter(zz_new[:,0],zz_new[:,1])
