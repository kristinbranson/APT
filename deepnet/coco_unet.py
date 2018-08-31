from PoseUNet_dataset import PoseUNet
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

coco_dir='/groups/branson/home/kabram/bransonlab/coco'
val_name = 'val2017'
train_name = 'train2017'
train_ann='{}/annotations/person_keypoints_{}.json'.format(coco_dir,train_name)
val_ann='{}/annotations/person_keypoints_{}.json'.format(coco_dir,train_name)
train_dir = os.path.join(coco_dir,train_name)
val_dir = os.path.join(coco_dir,val_name)

from poseConfig import config


def rle_to_segment(mask):
    # opencv 3.2
    mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # before opencv 3.2
    # contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
    #                                                    cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
    return segmentation


class coco_unet(PoseUNet.PoseUNetMulti):


    def __init__(self):

        conf = config()
        conf.n_classes = 17
        conf.max_n_animals = 20
        proj_name = 'coco_segmentation_unet'
        conf.set_exp_name(proj_name)
        conf.cachedir = os.path.join(localSetup.bdir,'cache','coco','segmentation')
        conf.imsz = [256,256]
        conf.sel_sz = 256
        conf.unet_rescale = 1
        conf.adjustContrast = False
        conf.normalize_img_mean = True
        conf.imgDim = 3
        conf.dl_steps = 100000
        conf.check_bounds_distort = False

        PoseUNet.__init__(self, conf, name=proj_name)


    def create_datasets(self):

        conf = self.conf

        n_cls = conf.n_classes
        train_coco = COCO(train_ann)
        val_coco = COCO(val_ann)
        val_ids = val_coco.getImgIds()
        train_ids = train_coco.getImgIds()
        val_dataset = tf.data.Dataset.from_tensor_slices(val_ids)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_ids)

        def coco_map(id, db_type):
            if db_type == 'train':
                coco = train_coco
                im_dir = train_dir
            else:
                coco = val_coco
                im_dir = val_dir

            img_info = coco.loadImgs(int(id))[0]
            img = imageio.imread(os.path.join(im_dir, img_info['file_name']))
            print img.shape

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
            pts = np.array(pts).reshape([n_ppl,conf.n_clases,3])
            for p in n_ppl:
                locs[0, p,:,:] = pts[p,:,:2]
                not_valid = pts[p,:,3]<0.5
                locs[0, p, not_valid,:] = np.nan

            if db_type == 'train':
                img, locs = PoseTools.preprocess_ims(img, locs, conf, distort=True, scale = 1)
            else:
                img, locs = PoseTools.preprocess_ims(img, locs, conf, distort=False, scale = 1)

            print('preprocessed the images')

            img = img[0,...]
            label_im = np.ones(img.shape[:2] + (conf.n_classes,)) * -1
            for ndx in range(n_ppl):
                cur_pts = locs[0, (ndx) * n_cls:(ndx + 1) * n_cls, :]
                cur_l = PoseTools.create_label_images(cur_pts[np.newaxis, ...], img.shape[:2], 1, conf.label_blur_rad)
                label_im = np.maximum(cur_l[0, ...], label_im)

            img_shape = img.shape[:2]
            img = zoom(img,[256./img_shape[0],256./img_shape[1],1])
            label_im = zoom(label_im,[256./img_shape[0],256./img_shape[1],1])

            info = np.zeros([4])
            info[0] = id
            info[3] = n_ppl
            return img, locs[0,...], info, label_im

        def train_map(id):
            return coco_map(id, 'train')

        def val_map(id):
            return coco_map(id, 'val')

        py_map_train = lambda im_id: tuple(tf.py_func(train_map, [im_id], [tf.float32, tf.float32, tf.float32, tf.float32]))
        py_map_val = lambda im_id: tuple(tf.py_func(val_map, [im_id], [tf.float32, tf.float32, tf.float32, tf.float32]))

#        train_dataset = train_dataset.map(map_func=py_map_train, num_parallel_calls=5)
        train_dataset = train_dataset.map(map_func=py_map_train)
        train_dataset = train_dataset.repeat()
#        train_dataset = train_dataset.shuffle(buffer_size=100)
        train_dataset = train_dataset.batch(self.conf.batch_size)
#        train_dataset = train_dataset.prefetch(buffer_size=100)

#        val_dataset = val_dataset.map(map_func=py_map_val, num_parallel_calls=2)
        val_dataset = val_dataset.map(map_func=py_map_val)
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(self.conf.batch_size)
#        val_dataset = val_dataset.prefetch(buffer_size=100)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        train_iter = train_dataset.make_one_shot_iterator()
        train_next = train_iter.get_next()

        val_iter = val_dataset.make_one_shot_iterator()
        val_next = val_iter.get_next()

        self.inputs = []
        for ndx in range(len(train_next)):
            self.inputs.append(train_next[ndx])
#for ndx in range(len(train_next)):
#            self.inputs.append(
#                tf.cond(self.ph['is_train'], lambda: tf.identity(train_next[ndx]), lambda: tf.identity(val_next[ndx])))

