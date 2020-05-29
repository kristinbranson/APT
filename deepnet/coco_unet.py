import PoseUNet_dataset as PoseUNet
from PoseCommon_dataset import PoseCommon
import PoseTools
import sys
sys.path.append('/groups/branson/home/kabram/bransonlab/coco/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import os
import tensorflow as tf
import imageio
import localSetup
from scipy.ndimage.interpolation import zoom
import numpy as np
import cv2
import traceback
import time

coco_dir='/groups/branson/home/kabram/bransonlab/coco'
val_name = 'val2017'
train_name = 'train2017'
train_ann='{}/annotations/person_keypoints_{}.json'.format(coco_dir,train_name)
val_ann='{}/annotations/person_keypoints_{}.json'.format(coco_dir,val_name)
train_dir = os.path.join(coco_dir,train_name)
val_dir = os.path.join(coco_dir,val_name)

from poseConfig import config


class coco_unet(PoseUNet.PoseUNetMulti):


    def __init__(self,name='coco_segmentation_unet'):

        conf = config()
        conf.n_classes = 17
        conf.max_n_animals = 20
        conf.set_exp_name(name)
        conf.cachedir = os.path.join(localSetup.bdir,'cache','coco','segmentation')
        conf.imsz = [240,320]
        conf.sel_sz = 256
        conf.unet_rescale = 1
        conf.adjust_contrast = False
        conf.normalize_img_mean = True
        conf.img_dim = 3
        conf.dl_steps = 100000
        conf.check_bounds_distort = False
        conf.label_blur_rad = 5

        PoseUNet.PoseUNet.__init__(self, conf, name=name)


    def create_datasets(self):

        conf = self.conf

        n_cls = conf.n_classes
        train_coco = COCO(train_ann)
        val_coco = COCO(val_ann)
        val_ids = val_coco.getImgIds()
        train_ids = train_coco.getImgIds()
        valid_train_ids = []
        for id1 in train_ids:
            img = train_coco.loadImgs(int(id1))[0]
            annIds = train_coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = train_coco.loadAnns(annIds)
            if len(anns)>0:
                valid_train_ids.append(id1)
        valid_val_ids = []
        for id1 in val_ids:
            img = val_coco.loadImgs(int(id1))[0]
            annIds = val_coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = val_coco.loadAnns(annIds)
            if len(anns)>0:
                valid_val_ids.append(id1)

        val_dataset = tf.data.Dataset.from_tensor_slices(valid_val_ids)
        train_dataset = tf.data.Dataset.from_tensor_slices(valid_train_ids)

        def coco_map(id, db_type):
            e1 = time.time()
            if db_type == 'train':
                coco = train_coco
                im_dir = train_dir
            else:
                coco = val_coco
                im_dir = val_dir

            img_info = coco.loadImgs(int(id))[0]
            img = imageio.imread(os.path.join(im_dir, img_info['file_name']))
            if img.ndim == 2:
                img = np.tile(img[:,:,np.newaxis],[1,1,3])
            elif img.shape[2] == 1:
                img = np.tile(img,[1,1,3])

            # Preprocessing
            annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            pts = []
            n_ppl = len(anns)
            for ndx, ann in enumerate(anns):
                kp = ann['keypoints']
                pts.extend(kp)

            img = img[np.newaxis,...]
            locs = np.ones([1, conf.max_n_animals, n_cls, 2]) * np.nan
            pts = np.array(pts).reshape([n_ppl,conf.n_classes,3])
            for p in range(n_ppl):
                locs[0, p,:,:] = pts[p,:,:2]
                not_valid = pts[p,:,2]<0.5
                locs[0, p, not_valid,:] = np.nan
            e2 = time.time()
#            print('Time to load {}'.format(e2-e1))

            if db_type == 'train':
                img, locs = PoseTools.preprocess_ims(img, locs, conf, distort=True, scale = 1)
            else:
                img, locs = PoseTools.preprocess_ims(img, locs, conf, distort=False, scale = 1)
            e3 = time.time()
#            print('Time to preprocess {}'.format(e3-e2))

            img = img[0,...]
            label_im = np.ones(img.shape[:2] + (conf.n_classes,)) * -1
            for ndx in range(n_ppl):
                cur_pts = locs[0, ndx, :, :]
                cur_l = PoseTools.create_label_images(cur_pts[np.newaxis, ...], img.shape[:2], 1, conf.label_blur_rad)
                label_im = np.maximum(cur_l[0, ...], label_im)

            img_shape = img.shape[:2]
            imsz = conf.imsz
            rr = max([float(imsz[0])/img_shape[0],float(imsz[1])/img_shape[1]])
            img = zoom(img,[rr,rr,1])
            label_im = zoom(label_im,[rr,rr,1])
            locs = locs*rr
            if img.shape[0] > conf.imsz[0]:
                dd = np.random.randint(img.shape[0] - conf.imsz[0])
                img = img[dd:(dd+conf.imsz[0]),:,:]
                label_im = label_im[dd:(dd+conf.imsz[0]),:,:]
                locs[...,1] -= dd

            if img.shape[1] > conf.imsz[1]:
                dd = np.random.randint(img.shape[1] - conf.imsz[1])
                img = img[:, dd:(dd+conf.imsz[1]),:]
                label_im = label_im[:, dd:(dd+conf.imsz[1]),:]
                locs[...,0] -= dd

            e4 = time.time()
#            print('Time to create label ims {}'.format(e4-e3))
#            print('Time total {}'.format(e4-e1))

            info = np.zeros([4])
            info[0] = id
            info[3] = n_ppl
            return img.astype('float32'), locs[0,...].astype('float32'), info.astype('float32'), label_im.astype('float32')

        def train_map(id):
            try:
                return coco_map(id,'train')
            except:
                traceback.print_exc()

        def val_map(id):
            try:
                return coco_map(id, 'val')
            except:
                traceback.print_exc()

        py_map_train = lambda im_id: tuple(tf.py_func(train_map, [im_id], [tf.float32, tf.float32, tf.float32, tf.float32]))
        py_map_val = lambda im_id: tuple(tf.py_func(val_map, [im_id], [tf.float32, tf.float32, tf.float32, tf.float32]))

        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(buffer_size=100)
#train_dataset = train_dataset.map(map_func=py_map_train, num_parallel_calls=20)
#       train_dataset = train_dataset.batch(self.conf.batch_size)
        train_dataset = train_dataset.apply(tensorflow.contrib.data.map_and_batch(map_func=py_map_train, batch_size=self.conf.batch_size,num_parallel_batches=15))
        train_dataset = train_dataset.prefetch(buffer_size=100)

        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.map(map_func=py_map_val, num_parallel_calls=4)
        val_dataset = val_dataset.batch(self.conf.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=100)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        train_iter = train_dataset.make_one_shot_iterator()
        train_next = train_iter.get_next()

        val_iter = val_dataset.make_one_shot_iterator()
        val_next = val_iter.get_next()

        self.inputs = []
        for ndx in range(len(train_next)):
            self.inputs.append(
                tf.cond(self.ph['is_train'], lambda: tf.identity(train_next[ndx]), lambda: tf.identity(val_next[ndx])))

