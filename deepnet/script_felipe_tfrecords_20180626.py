import PoseTools
import socket
import numpy as np
import os
import imageio
import tensorflow as tf
from multiResData import int64_feature, float_feature, bytes_feature
import open_pose

def create_iterators():
    from stephenHeadConfig import conf
    if socket.gethostname() == 'mayankWS':
        conf.cachedir = '/home/mayank/Dropbox (HHMI)/temp/felipe/'
        db_types = ['val']
        val_distort = True
    else:
        conf.cachedir = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/'
        db_types = ['train','val']
        val_distort = False

    conf.n_classes = 5
    conf.op_affinity_graph = [[0,1],[1,2]] # EDIT THIS. Dummy affinity maps
    conf.normalize_img_mean = False #EDIT THIS to True if image mean should be subtracted.
    conf.scale_range = 0.2 # image will be scaled by (1-scale_range) to (1+scale_range)
    conf.rrange = 30 # image will rotated by +-30
    conf.trange = 30 # image will be translated by +-30 in both x and y direction
    conf.brange = [1-0.2, 1+0.2] # brightness is adjusted by this much. 0 is no adjustment and 1 is maximum adjustment
    conf.crange = [0.7, 1.3] # contrast range. [1, 1] does nothing, [0,2] adjusts the contrast by maximum
    conf.imgDim = 3
    conf.adjustContrast = False # this is different contrast adjustment than earlier. Should be false for colored images.
    conf.perturb_color = False
    conf.op_label_scale = 8
    conf.op_rescale = 1

    data_iterators = []
    for dtype in db_types:
        if dtype == 'train':
            cur_di = open_pose.DataIteratorTF(conf,dtype,True,True)
        else:
            cur_di = open_pose.DataIteratorTF(conf,dtype,val_distort,False)
        data_iterators.append(cur_di)

    return data_iterators


def create_db():
    hsz = 250 # EDIT THIS

    for dtype in ['train','val']:

        cachedir = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/'
        if dtype == 'train':
            ann_file = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/annotations/bee_keypoints_train.json'
            im_dir = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/train2017bee'
            out_filename = os.path.join(cachedir, 'fulltrain_TF.tfrecords')
        else:
            ann_file = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/annotations/bee_keypoints_val.json'
            im_dir = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/val2017bee'
            out_filename = os.path.join(cachedir, 'val_TF.tfrecords')

        env =  tf.python_io.TFRecordWriter(out_filename)

        L = PoseTools.json_load(ann_file)
        ann_im_id = np.array([k['image_id'] for k in L['annotations']])
        bbox = np.array([k['bbox'] for k in L['annotations']] )
        locs = np.array([k['keypoints'] for k in L['annotations']] )

        n_labels = len(L['annotations'])
        locs = locs.reshape([n_labels,-1,2])
        bbox = bbox.reshape([n_labels,4,2])

        mid_pt = bbox.mean(axis=1)

        im_id = [k['id'] for k in L['images']]
        im_name = [k['file_name'] for k in L['images']]

        for ndx in range(n_labels):
            im_ndx = np.where(im_id==ann_im_id[ndx])[0]
            cur_im = os.path.join(im_dir,im_name[im_ndx])
            im = imageio.imread(cur_im)

            cur_mid_x,cur_mid_y = mid_pt[ndx,:]
            cur_patch = im[cur_mid_y-hsz:cur_mid_y+hsz,cur_mid_x-hsz:cur_mid_x+hsz,:]
            cur_locs = locs[ndx,:,:]-mid_pt[ndx:ndx+1,:] + hsz

            image_raw = cur_patch.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': int64_feature(cur_patch.shape[0]),
                'width': int64_feature(cur_patch.shape[1]),
                'depth': int64_feature(cur_patch.shape[2]),
                'trx_ndx': int64_feature(0),
                'locs': float_feature(cur_locs.flatten()),
                'expndx': float_feature(ndx),
                'ts': float_feature(ndx),
                'image_raw': bytes_feature(image_raw)}))
            env.write(example.SerializeToString())

        env.close()
