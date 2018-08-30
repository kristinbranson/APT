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

coco_dir='/groups/branson/home/kabram/bransonlab/coco'
val_name = 'val2017'
train_name = 'train2017'
train_ann='{}/annotations/instances_{}.json'.format(coco_dir,train_name)
val_ann='{}/annotations/instances_{}.json'.format(coco_dir,train_name)
train_dir = os.path.join(coco_dir,train_name)
val_dir = os.path.join(coco_dir,val_name)

from poseConfig import config as conf

class coco_unet(PoseUNet):


    def __init__(self):

        conf.n_classes = 80
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

        PoseUNet.__init__(self, conf, name=proj_name)


    def create_datasets(self):

        conf = self.conf

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

            img = coco.loadImgs(int(id))[0]
            img = imageio.imread(os.path.join(im_dir, img['file_name']))

            # Preprocessing
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            seg_sz = []
            seg_label = []
            for ann in anns:
                cur_seg_sz = []
                for seg in ann['segmentation']:
                    cur_seg_sz.append(len(seg))
                    seg_label.extend(seg)
                seg_sz.append(cur_seg_sz)

            seg_label = np.array(seg_label).reshape([1,int(len(seg_label/2)),2])
            img = img[np.newaxis,...]

            if db_type == 'train':
                img, seg_label = PoseTools.preprocess_ims(img, seg_label, conf, distort=True, scale = 1)
            else:
                img, seg_label = PoseTools.preprocess_ims(img, seg_label, conf, distort=False, scale = 1)

            img = img[0,...]
            seg_label = seg_label.flatten().tolist()
            seg_start = 0
            for ndx, ann in enumerate(anns):
                for seg_ndx, seg in enumerate(ann['segmentation']):
                    seg_end = seg_start + seg_sz[ndx][seg_ndx]
                    cur_seg = seg_label[seg_start:seg_end]
                    ann['segmentation'][seg_ndx] = cur_seg
                    seg_start = seg_end

            #Resizing to common size
            mask = np.zeros(img.shape[:2] + (conf.n_classes,))
            img = zoom(img,[256./img.shape[0],256./img.shape[1],1])

            for ann in anns:
                cur_cat = ann['category_id']
                mask[:,:,cur_cat] = coco.annToMask(ann)
            mask = zoom(mask,[256./img.shape[0],256./img.shape[1],1])

            return img, np.zeros([conf.batch_size, conf.n_classes,2]), np.ones([conf.batch_size,3])*id, mask

        def train_map(id):
            return coco_map(id, 'train')

        def val_map(id):
            return coco_map(id, 'val')

        py_map_train = lambda im_id: tuple(tf.py_func(train_map, [im_id], [tf.float32, tf.float32, tf.float32, tf.float32]))
        py_map_val = lambda im_id: tuple(tf.py_func(val_map, [im_id], [tf.float32, tf.float32, tf.float32, tf.float32]))

        train_dataset = train_dataset.map(map_func=py_map_train, num_parallel_calls=5)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(buffer_size=100)
        train_dataset = train_dataset.batch(self.conf.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=100)

        val_dataset = val_dataset.map(map_func=py_map_val, num_parallel_calls=2)
        val_dataset = val_dataset.repeat()
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

