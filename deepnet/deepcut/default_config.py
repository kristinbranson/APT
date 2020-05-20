"""
Modified by Mayank Kabra
Adapted from DeepLabCut2.0 Toolbox (deeplabcut.org)
Copyright A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

"""
from easydict import EasyDict as edict

cfg = edict()

cfg.stride = 8.0
cfg.weigh_part_predictions = False
cfg.weigh_negatives = False
cfg.fg_fraction = 0.25
cfg.weigh_only_present_joints = False
cfg.mean_pixel = [123.68, 116.779, 103.939]
#cfg.mean_pixel = [128.,128.,128.]
cfg.shuffle = True
#cfg.snapshot_prefix = "./snapshot"
cfg.log_dir = "log"
cfg.global_scale = 1.0
cfg.location_refinement = False
cfg.locref_stdev = 7.2801
cfg.locref_loss_weight = 1.0
cfg.locref_huber_loss = True
cfg.optimizer = "sgd"
cfg.intermediate_supervision = False
cfg.intermediate_supervision_layer = 12
cfg.regularize = False
cfg.weight_decay = 0.0001
cfg.mirror = False
cfg.crop = False
cfg.crop_pad = 0
cfg.scoremap_dir = "test"
cfg.dataset_type = "default"
cfg.use_gt_segm = False
cfg.batch_size = 1
cfg.video = False
cfg.video_batch = False
cfg.init_weights = 'deeplabcut/pretrained/resnet_v1_50.ckpt'
