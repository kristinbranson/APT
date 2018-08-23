import os
import pprint
import logging

import yaml
from easydict import EasyDict as edict

import default_config


cfg = default_config.cfg


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        #if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)

    logging.info("Config:\n"+pprint.pformat(cfg))
    return cfg


def load_config(filename = "pose_cfg.yaml"):
    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.environ['POSE_PARAM_PATH'] + '/' + filename
    return cfg_from_file(filename)


def convert_to_deepcut(conf):
    conf = edict(conf.__dict__)
    conf.all_joints = []
    conf.all_joints_names = []
    for ndx in range(conf.n_classes):
        conf.all_joints.append([ndx])
        conf.all_joints_names.append('part_{}'.format(ndx))
        conf.dataset = os.path.join(conf.cachedir,conf.dlc_train_data_file)
        conf.global_scale = 1./conf.dlc_rescale
        conf.num_joints = conf.n_classes
        conf.scale_jitter_lo = 0.9
        conf.scale_jitter_up = 1.1
        conf.net_type = 'resnet_50'
        conf.pos_dist_thresh = 17
        conf.max_input_size = 1000
        conf.intermediate_supervision = False
        conf.intermediate_supervision_layer = 12
        conf.location_refinement = True
        conf.locref_huber_loss = True
        conf.locref_loss_weight = 0.05
        conf.locref_stdev = 7.2801
        conf.mirror = False

    _merge_a_into_b(conf, cfg)
    return cfg

if __name__ == "__main__":
    print(load_config())