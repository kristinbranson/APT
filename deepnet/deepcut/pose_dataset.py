import logging
import random as rand
from enum import Enum

import numpy as np
from numpy import array as arr
from numpy import concatenate as cat

import scipy.io as sio
from scipy.misc import imread, imresize
import PoseTools
import pickle

class Batch(Enum):
    inputs = 0
    part_score_targets = 1
    part_score_weights = 2
    locref_targets = 3
    locref_mask = 4
    data_item = 5
    locs = 6


def mirror_joints_map(all_joints, num_joints):
    res = np.arange(num_joints)
    symmetric_joints = [p for p in all_joints if len(p) == 2]
    for pair in symmetric_joints:
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res


def extend_crop(crop, crop_pad, image_size):
    crop[0] = max(crop[0] - crop_pad, 0)
    crop[1] = max(crop[1] - crop_pad, 0)
    crop[2] = min(crop[2] + crop_pad, image_size[2] - 1)
    crop[3] = min(crop[3] + crop_pad, image_size[1] - 1)
    return crop


def data_to_input(data):
    return np.expand_dims(data, axis=0).astype(float)


class DataItem:
    pass


class PoseDataset:
    def __init__(self, cfg, db_file_name):
        self.cfg = cfg
        self.data = self.load_dataset(db_file_name)
        self.num_images = len(self.data)
        if self.cfg.mirror:
            self.symmetric_joints = mirror_joints_map(cfg.all_joints, cfg.num_joints)
        self.curr_img = 0
        self.set_shuffle(cfg.shuffle)

    def load_dataset(self, db_file_name):
        cfg = self.cfg

        with open(db_file_name,'r') as f:
            mlab = pickle.load(f)
        num_images = len(mlab)

        # file_name = cfg.dataset
        # # Load Matlab file dataset annotation
        # mlab = sio.loadmat(file_name)
        # self.raw_data = mlab
        # mlab = mlab['dataset']
        # num_images = mlab.shape[1]


        data = []
        has_gt = True

        for i in range(num_images):
            sample = mlab[i]

            item = DataItem()
            item.image_id = i
            item.im_path = sample[0]
            item.im_size = sample[1]
            if len(sample) >= 3:
                joints = sample[2]
                joint_id = joints[:, 0]
                # make sure joint ids are 0-indexed
                if joint_id.size != 0:
                    assert((joint_id < cfg.num_joints).any())
                joints[:, 0] = joint_id
                item.joints = [joints]
            else:
                has_gt = False
            # if cfg.crop:
            #     crop = sample[3][0] - 1
            #     item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)

        self.has_gt = has_gt
        return data

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
        if not shuffle:
            assert not self.cfg.mirror
            self.image_indices = np.arange(self.num_images)

    def mirror_joint_coords(self, joints, image_width):
        # horizontally flip the x-coordinate, keep y unchanged
        joints[:, 1] = image_width - joints[:, 1] - 1
        return joints

    def mirror_joints(self, joints, symmetric_joints, image_width):
        # joint ids are 0 indexed
        res = np.copy(joints)
        res = self.mirror_joint_coords(res, image_width)
        # swap the joint_id for a symmetric one
        joint_id = joints[:, 0].astype(int)
        res[:, 0] = symmetric_joints[joint_id]
        return res

    def shuffle_images(self):
        num_images = self.num_images
        if self.cfg.mirror:
            image_indices = np.random.permutation(num_images * 2)
            self.mirrored = image_indices >= num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - num_images
            self.image_indices = image_indices
        else:
            self.image_indices = np.random.permutation(num_images)

    def num_training_samples(self):
        num = self.num_images
        if self.cfg.mirror:
            num *= 2
        return num

    def next_training_sample(self):
        if self.curr_img == 0 and self.shuffle:
            self.shuffle_images()

        curr_img = self.curr_img
        self.curr_img = (self.curr_img + 1) % self.num_training_samples()

        imidx = self.image_indices[curr_img]
        mirror = self.cfg.mirror and self.mirrored[curr_img]

        return imidx, mirror

    def get_training_sample(self, imidx):
        return self.data[imidx]

    def get_scale(self):
        cfg = self.cfg
        scale = cfg.global_scale
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(cfg.scale_jitter_lo, cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale

    def next_batch(self):

        data_items = []
        for ndx in range(self.cfg.batch_size):
            imidx, mirror = self.next_training_sample()
            data_item = self.get_training_sample(imidx)
            scale = self.get_scale()
            data_items.append([data_item,scale, mirror])

        return self.make_batch(data_items)

    def is_valid_size(self, image_size, scale):
        im_width = image_size[2]
        im_height = image_size[1]

        max_input_size = 100
        if im_height < max_input_size or im_width < max_input_size:
            return False

        if hasattr(self.cfg, 'max_input_size'):
            max_input_size = self.cfg.max_input_size
            input_width = im_width * scale
            input_height = im_height * scale
            if input_height * input_width > max_input_size * max_input_size:
                return False

        return True

    def crop_to_size(self, img, sz):
        new_sz = img.shape[:2]
        dx = sz[1] - new_sz[1]
        dy = sz[0] - new_sz[0]
        out_img = np.zeros(sz).astype(img.dtype)
        if dx < 0 or dy< 0:
            hdx = -int(dx/2)
            hdy = -int(dy/2)
            out_img[:,:,...] = img[hdy:(sz[0]+hdy),hdx:(sz[1]+hdx),...]
        else:
            hdx = int(dx/2)
            hdy = int(dy/2)
            out_img[hdy:(new_sz[0] + hdy), hdx:(new_sz[1] + hdx), ...] = img
        return out_img, dx, dy

    def make_batch(self, data_items):
        all_ims = []
        all_part_scores = []
        all_part_weights = []
        all_locref_targets = []
        all_locref_mask = []
        all_scaled_joints = []

        for data_item, scale, mirror in data_items:
            im_file = data_item.im_path
            logging.debug('image %s', im_file)
            logging.debug('mirror %r', mirror)
            image = imread(im_file, mode='RGB')

            if self.has_gt:
                joints = np.copy(data_item.joints)

            if self.cfg.crop:
                crop = data_item.crop
                image = image[crop[1]:crop[3] + 1, crop[0]:crop[2] + 1, :]
                if self.has_gt:
                    joints[:, 1:3] -= crop[0:2].astype(joints.dtype)

            im_sz = image.shape
            img = imresize(image, scale) if scale != 1 else image
            img, dx, dy = self.crop_to_size(img, im_sz)
            scaled_img_size = arr(img.shape[0:2])

            if mirror:
                img = np.fliplr(img)


            if self.has_gt:
                stride = self.cfg.stride

                if mirror:
                    joints = [self.mirror_joints(person_joints, self.symmetric_joints, image.shape[1]) for person_joints in
                              joints]

                sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2

                scaled_joints = [ ((person_joints[:, 1:3] * scale) + int(dx/2))  \
                                  for person_joints in joints]

                assert len(joints)==1, 'This doesnt work for multi animal'
                locs = scaled_joints[0]
                img, locs = PoseTools.preprocess_ims(
                    img[np.newaxis,...],locs[np.newaxis,...], self.cfg,
                    distort=True, scale =1)

                scaled_joints = [locs[0,...]]

                all_scaled_joints.append(scaled_joints[0])
                joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
                part_score_targets, part_score_weights, locref_targets, locref_mask = self.compute_target_part_scoremap(
                    joint_id, scaled_joints, data_item, sm_size, scale)

                all_part_scores.append(part_score_targets)
                all_part_weights.append(part_score_weights)
                all_locref_mask.append(locref_mask)
                all_locref_targets.append(locref_targets)

                all_ims.append(img[0,...])
            else:
                all_ims.append(img)

        all_ims = np.stack(all_ims)
        batch = {Batch.inputs: all_ims}

        if self.has_gt:
            all_part_scores = np.stack(all_part_scores)
            all_part_weights = np.stack(all_part_weights)
            all_locref_mask = np.stack(all_locref_mask)
            all_locref_targets= np.stack(all_locref_targets)
            all_scaled_joints = np.stack(all_scaled_joints)
            batch.update({
                Batch.part_score_targets: all_part_scores,
                Batch.part_score_weights: all_part_weights,
                Batch.locref_targets: all_locref_targets,
                Batch.locref_mask: all_locref_mask,
                Batch.locs: all_scaled_joints
            })

        batch[Batch.data_item] = data_items
        return batch

    def compute_target_part_scoremap(self, joint_id, coords, data_item, size, scale):
        stride = self.cfg.stride
        dist_thresh = self.cfg.pos_dist_thresh * scale
        num_joints = self.cfg.num_joints
        half_stride = stride / 2
        scmap = np.zeros(cat([size, arr([num_joints])]))
        locref_size = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        locref_scale = 1.0 / self.cfg.locref_stdev
        dist_thresh_sq = dist_thresh ** 2

        width = size[1]
        height = size[0]

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asscalar(joint_pt[0])
                j_y = np.asscalar(joint_pt[1])

                # don't loop over entire heatmap, but just relevant locations
                j_x_sm = round((j_x - half_stride) / stride)
                j_y_sm = round((j_y - half_stride) / stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

                for j in range(int(min_y), int(max_y) + 1):  # range(height):
                    pt_y = j * stride + half_stride
                    for i in range(int(min_x), int(max_x )+ 1):  # range(width):
                        # pt = arr([i*stride+half_stride, j*stride+half_stride])
                        # diff = joint_pt - pt
                        # The code above is too slow in python
                        pt_x = i * stride + half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx ** 2 + dy ** 2
                        # print(la.norm(diff))
                        if dist <= dist_thresh_sq:
                            scmap[j, i, j_id] = 1
                            locref_mask[j, i, j_id * 2 + 0] = 1
                            locref_mask[j, i, j_id * 2 + 1] = 1
                            locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
                            locref_map[j, i, j_id * 2 + 1] = dy * locref_scale

        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)

        return scmap, weights, locref_map, locref_mask

    def compute_scmap_weights(self, scmap_shape, joint_id, data_item):
        cfg = self.cfg
        if cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights
