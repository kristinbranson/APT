import pathlib
import os
# from urllib.request import HTTPPasswordMgrWithDefaultRealm
# import sys
# sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(),'mmpose'))
import mmcv
# from mmcv import Config
import mmpose
# from mmpose.models import build_posenet
import poseConfig
import copy
# from mmpose import __version__
# from mmcv.utils import get_git_hash
from packaging import version

assert version.parse(mmpose.__version__).major > 0
from mmpose.datasets import CombinedDataset


from mmpose.datasets import build_dataset
import torch

from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.config import Config
from mmpose.datasets.datasets import BaseCocoStyleDataset as TopDownCocoDataset
from mmpose.apis import init_model
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate


import logging
import time
import PoseCommon_pytorch
# from PoseCommon_pytorch import PoseCommon_pytorch
# from mmcv.runner import init_dist, set_random_seed
import pickle
import json
# from mmpose.core import wrap_fp16_model
# from mmpose.datasets.pipelines import Compose
# from mmcv.parallel import collate, scatter
import glob
import PoseTools
import numpy as np
import APT_interface
import cv2
import multiprocessing as mp

from xtcocotools.coco import COCO
from collections import OrderedDict


class TrainDataHookDummy(Hook):
    def __init__(self, out_file=None, conf=None, interval=50):
        # self.interval = interval
        # self.out_file = out_file
        # self.conf = conf
        # self.td_data = {'train_loss':[],'train_dist':[],'step':[],'val_loss':[],'val_dist':[]}
        pass

    def after_train_iter(self, runner):
        # if not self.every_n_iters(runner,self.interval):
        #    return
        # self.td_data['step'].append(runner.iter + 1)
        # runner.log_buffer.average(self.interval)
        # if 'loss' in runner.log_buffer.output.keys():
        #     self.td_data['train_loss'].append(runner.log_buffer.output['loss'].copy())
        # else:
        #     self.td_data['train_loss'].append(runner.log_buffer.output['all_loss'].copy())
        # self.td_data['train_dist'].append(np.nan)
        # self.td_data['val_dist'].append(np.nan)
        # self.td_data['val_loss'].append(np.nan)

        # train_data_file = self.out_file
        # with open(train_data_file, 'wb') as td_file:
        #     pickle.dump([self.td_data, self.conf], td_file, protocol=2)
        # json_data = {}
        # for x in self.td_data.keys():
        #     json_data[x] = np.array(self.td_data[x]).astype(np.float64).tolist()
        # with open(train_data_file + '.json', 'w') as json_file:
        #     json.dump(json_data, json_file)
        pass


#@DATASETS.register_module()
class TopDownAPTDataset(TopDownCocoDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False,
                 dataset_info=None):
        # Overriding topdowncoocodataset init code because it is bad and awful with hardcoded values.
        super(TopDownCocoDataset, self).__init__(ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode,
                                                 dataset_info=dataset_info)
        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        if 'image_thr' in data_cfg:
            logging.warning(
                'image_thr is deprecated, '
                'please use det_bbox_thr instead', DeprecationWarning)
            self.det_bbox_thr = data_cfg['image_thr']
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        # self.ann_info['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
        #                                [11, 12], [13, 14], [15, 16]]
        #
        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.ann_info['lower_body_ids'] = (11, 12, 13, 14, 15, 16)

        self.ann_info['use_different_joint_weights'] = False
        # self.ann_info['joint_weights'] = np.array(
        #     [
        #         1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
        #         1.2, 1.5, 1.5
        #     ],
        #     dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        # 'https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/'
        # 'pycocotools/cocoeval.py#L523'
        # self.sigmas = np.array([
        #     .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
        #     .87, .87, .89, .89
        # ]) / 10.0

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'coco'

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

        ## END MMPOSE INIT CODE

        # My init code.
        import poseConfig
        conf = poseConfig.conf

        self.use_gt_bbox = True
        flip_idx = list(range(conf.n_classes))
        pairs = []
        done = []
        for kk in conf.flipLandmarkMatches.keys():
            if int(kk) in done:
                continue
            pairs.append([int(kk), int(conf.flipLandmarkMatches[kk])])
            done.append(int(kk))
            done.append(int(conf.flipLandmarkMatches[kk]))
        self.ann_info['flip_pairs'] = pairs
        self.ann_info['joint_weights'] = np.ones([conf.n_classes])
        self.sigmas = np.ones([conf.n_classes]) * 0.6 / 10.0

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        scale = np.array([1.0, 1.0], dtype=np.float32)
        return center, scale


## APT pipeline
# from mmpose.datasets.registry import PIPELINES

#@PIPELINES.register_module()
class APTtransform:
    """Data augmentation using APT's posetools.

    """

    def __init__(self, distort):
        import poseConfig
        self.conf = poseConfig.conf
        self.conf.normalize_img_mean = False
        self.conf.normalize_batch_mean = False
        self.distort = distort

    def __call__(self, results):
        import PoseTools as pt
        conf = self.conf
        if conf.is_multi:
            image, joints, mask = results['img'], results['joints'], results['mask']
            # assert mask[0].min(), 'APT transform only supports dummy masks'
            assert not results['ann_info']['scale_aware_sigma'], 'APT doesnt support this'
            for jndx in range(len(joints) - 1):
                # assert len(joints) ==2, "APT Transform is tested only for at most two scales"

                assert np.allclose(joints[jndx], joints[jndx + 1],
                                   equal_nan=True), "APT transform is tested only for two identical scale inputs"
            jlen = len(joints)
            joints_in = joints[0][..., :2]
            occ_in = joints[0][..., 2]
            joints_in[occ_in < 0.5, :] = -100000
            image, joints_out, mask_out, occ = pt.preprocess_ims(image[np.newaxis, ...],
                                                                 joints_in[np.newaxis, ...],
                                                                 conf,
                                                                 self.distort,
                                                                 conf.rescale,
                                                                 mask=mask[0][None, ...],
                                                                 occ=occ_in)
            image = image.astype('float32')
            joints_out_occ1 = np.isnan(joints_out[0, ..., 0:1]) | (joints_out[0, ..., 0:1] < -1000)
            joints_out_occ = occ < 0.5
            assert np.array_equal(joints_out_occ1,
                                  joints_out_occ), 'Occlusion from processing and from joints should be equal'
            joints_out = np.concatenate([joints_out[0, ...], (~joints_out_occ) * 2], axis=-1)
            in_sz = results['ann_info']['image_size']
            out_sz = results['ann_info']['heatmap_size']
            assert all([round(in_sz / o) == in_sz / o for o in
                        out_sz]), 'Output sizes should be integer multiples of input sizes'
            outs = [int(round(in_sz / o)) for o in out_sz]
            results['joints'] = [joints_out * osz / in_sz for osz in out_sz]
            results['mask'] = [mask_out[0, ::o, ::o] > 0.5 for o in outs]

        else:
            image, joints, occ_in = results['img'], results['joints_3d'], results['joints_3d_visible']
            assert joints[:, 2].max() < 0.00001, 'APT does not work 3d'
            occ_in = occ_in[:, 0]
            joints_in = joints[:, :2]
            joints_in[occ_in < 0.5, :] = -100000

            image, joints_out, occ = pt.preprocess_ims(image[np.newaxis, ...], joints_in[np.newaxis, ...], conf,
                                                       self.distort, conf.rescale, occ=1 - occ_in)
            isz = results['ann_info']['image_size']
            if image.shape[1] != isz[1] or image.shape[2] != isz[0]:
                # print('Resizing image to ',isz)
                image = cv2.resize(image[0, ...], (isz[0], isz[1]))
                image = image[None, ...]
            image = image.astype('float32')
            joints_out_occ1 = np.isnan(joints_out[0, ..., 0:1]) | (joints_out[0, ..., 0:1] < -1000)
            joints_out_occ = occ[0, :, None] > 0.5
            # if conf.check_bounds_distort:
            assert np.array_equal(joints_out_occ1,
                                  joints_out_occ), 'Occlusion from processing and from joints should be equal'

            results['joints_3d'] = np.concatenate([joints_out[0, ...], np.zeros_like(joints_out[0, :, :1])], 1)
            results['joints_3d_visible'] = np.concatenate(
                [1 - joints_out_occ, 1 - joints_out_occ, np.zeros_like(joints_out_occ)], 1)

        results['img'] = np.clip(image[0, ...], 0, 255).astype('uint8')
        results['center'] = np.array([image.shape[2] / 2, image.shape[1] / 2], dtype='float32')
        results['scale'] = np.array([1.0, 1.0], dtype='float32')
        results['flip_pairs'] = []
        return results


def create_mmpose_cfg(conf, mmpose_config_file, run_name):
    curdir = pathlib.Path(__file__).parent.absolute()
    mmpose_init_file_path = mmpose.__file__
    mmpose_dir = os.path.dirname(mmpose_init_file_path)
    dot_mim_folder_path = os.path.join(mmpose_dir, '.mim')  # this feels not-robust
    mmpose_config_file_path = os.path.join(dot_mim_folder_path, mmpose_config_file)
    data_bdir = conf.cachedir

    cfg = Config.fromfile(mmpose_config_file_path)
    imsz = [int(c / conf.rescale / 32) * 32 for c in conf.imsz[::-1]]  # conf.imsz[0]
    if type(cfg.codec) == list:
        ex_codec = cfg.codec[0]
        for ndx in range(len(cfg.codec)):
            cfg.codec[ndx].input_size = imsz
    else:
        ex_codec = cfg.codec.copy()
        cfg.codec.input_size = imsz
    default_im_sz = ex_codec.input_size
    if 'heatmap_size' in ex_codec:
        default_hm_sz = ex_codec.heatmap_size
    else:
        default_hm_sz = None


    if conf.is_multi:
        if 'heatmap_size' in cfg.codec:
            cfg.codec.heatmap_size = [int(default_hm_sz[0] / default_im_sz[0] * i) for i in imsz]
        if hasattr(cfg.model.head, 'loss_keypoint'):
            cfg.model.keypoint_head.loss_keypoint.num_joints = conf.n_classes
        if 'head' in cfg.model:
            cfg.model.head.decoder = cfg.codec
            if 'num_keypoints' in cfg.model.head:
                cfg.model.head.num_keypoints = conf.n_classes
        if conf.mmpose_net == 'dekr':
            cfg.model.head.rescore_cfg = None
            if conf.multi_loss_mask:
                cfg.codec.type = 'SPR_mask'
        if conf.mmpose_net == 'cid' and conf.multi_loss_mask:
            raise RuntimeError('For the CiD model, if a frame has any labeled animals, all animals in the frame must be labeled.  Therefore in the training parameters "Unlabeled animals present" (a.k.a. multi_loss_mask) must be false.')


    else:
        if default_hm_sz is not None:
            # assert default_im_sz[0] / default_hm_sz[
            # 0] == 4, 'Single animal mmpose is tested only for hmaps downsampled by 4'
            if type(cfg.codec) == list:
                for ndx in range(len(cfg.codec)):
                    cfg.codec[ndx].heatmap_size = [csz // 4 for csz in cfg.codec[ndx].input_size]
            else:
                cfg.codec.heatmap_size = [csz // 4 for csz in cfg.codec.input_size]
        if 'keypoint_head' in cfg.model and 'out_shape' in cfg.model.keypoint_head:
            cfg.model.keypoint_head.out_shape = [csz // 4 for csz in cfg.codec.input_size[::-1]]
        if 'head' in cfg.model:
            if type(cfg.codec) == list:
                cfg.model.head.decoder = cfg.codec[-1]
            else:
                cfg.model.head.decoder = cfg.codec
            cfg.model.head.out_channels = conf.n_classes

    if 'train_cfg' in cfg.model:
        cfg.model.train_cfg.max_train_instances = conf.max_n_animals

    # dataloader and pipelines
    for ttype in ['train', 'val']:
        pname = ttype + '_pipeline'
        dname = ttype + '_dataloader'
        ename = ttype + '_evaluator'

        fname = conf.trainfilename if ttype == 'train' else conf.valfilename
        ann_file = os.path.join(data_bdir, f'{fname}.json')
        if not os.path.exists(ann_file):
            cfg[dname].dataset.ann_file = cfg['train_dataloader']['dataset']['ann_file']
        else:
            cfg[dname].dataset.ann_file = ann_file
        # ALTTODO: Fix this:
        # cfg.data[ttype].img_prefix = os.path.join(data_bdir, name)
        cfg[dname].dataset.data_prefix.img = ''
        if ttype != 'train':
            cfg[dname].dataset.bbox_file = None

        # meta info -- flip pairs and skeleton
        key_info = OrderedDict()
        for ndx in range(conf.n_classes):
            key_info[ndx] = {'name': f'{ndx}', 'id': ndx, 'color': [128, 128, 128], 'type': 'upper', 'swap': ''}
        for kk in conf.flipLandmarkMatches.keys():
            key_info[int(kk)]['swap'] = f'{conf.flipLandmarkMatches[kk]}'
        skel_info = {}
        for ndx, cure in enumerate(conf.op_affinity_graph):
            skel_info[ndx] = {'link': (f'{cure[0]}', f'{cure[1]}'), 'id': ndx, 'color': [128, 128, 128]}
        if conf.horz_flip:
            fdir = 'horizontal'
        else:
            fdir = 'vertical'
        metainfo = {'keypoint_info': key_info,
                    'skeleton_info': skel_info,
                    'flip_direction':fdir,
                    'joint_weights': [1., ] * conf.n_classes,
                    'sigmas': [0.06, ] * conf.n_classes,
                    'dataset_name':'apt'}
        cfg[dname].dataset.metainfo = metainfo


        # Remove some of the transforms.
        in_pipe = cfg[pname]
        cfg[pname] = [ii for ii in in_pipe if ii.type not in ['RandomHalfBody']]

        if conf.get('mmpose_use_apt_augmentation', False):
            # substitute their augmentation pipeline with ours
            if ttype == 'train':
                if conf.is_multi:
                    assert \
                        (cfg.data[ttype].pipeline[1].type == 'BottomUpRandomAffine') and \
                        (cfg.data[ttype].pipeline[2].type == 'BottomUpRandomFlip'), \
                        'Unusual mmpose augmentation pipeline cannot be substituted by APT augmentation'
                    cfg.data[ttype].pipeline[2:3] = []
                    cfg.data[ttype].pipeline[1] = ConfigDict({'type': 'APTtransform', 'distort': True})
                elif (cfg.data[ttype].pipeline[1].type == 'TopDownRandomFlip'):
                    # old style top down pipeline
                    assert \
                        (cfg.data[ttype].pipeline[1].type == 'TopDownRandomFlip') and \
                        (cfg.data[ttype].pipeline[2].type == 'TopDownGetRandomScaleRotation') and \
                        (cfg.data[ttype].pipeline[3].type == 'TopDownAffine'), \
                        'Unusual mmpose augmentation pipeline cannot be substituted by APT augmentation'
                    cfg.data[ttype].pipeline[2:4] = []
                    cfg.data[ttype].pipeline[1] = ConfigDict({'type': 'APTtransform', 'distort': True})
                else:
                    assert \
                        (cfg.data[ttype].pipeline[1].type == 'TopDownGetBboxCenterScale') and \
                        (cfg.data[ttype].pipeline[2].type == 'TopDownRandomShiftBboxCenter') and \
                        (cfg.data[ttype].pipeline[3].type == 'TopDownRandomFlip') and \
                        (cfg.data[ttype].pipeline[4].type == 'TopDownGetRandomScaleRotation') and \
                        (cfg.data[ttype].pipeline[5].type == 'TopDownAffine'), \
                        'Unusual mmpose augmentation pipeline cannot be substituted by APT augmentation'
                    cfg.data[ttype].pipeline[2:6] = []
                    cfg.data[ttype].pipeline[1] = ConfigDict({'type': 'APTtransform', 'distort': True})
                    # else:
        #     assert conf.rescale == 1, 'MMpose aug with rescale has not been implemented'

        for p in cfg[pname]:
            if p.type == 'BottomUpRandomAffine':
                p.rot_factor = conf.rrange / 2
                sfactor = 1 / conf.scale_factor_range if (conf.scale_factor_range < 1) else conf.scale_factor_range
                p.scale_factor = [1 / sfactor, sfactor]
                p.shift_factor = conf.trange / conf.imsz[0]
                p.input_size = imsz
            if p.type == 'BottomupRandomAffine':
                p.input_size = imsz
            elif p.type == 'TopDownGetRandomScaleRotation':
                p.rot_factor = conf.rrange / 2
                # p.rot_prob = 1.
                sfactor = 1 / conf.scale_factor_range if (conf.scale_factor_range < 1) else conf.scale_factor_range
                p.scale_factor = sfactor - 1
            elif p.type in ['RandomFlip', 'BottomUpRandomFlip']:
                p.prob = 0.5 if (conf.horz_flip or conf.vert_flip) else 0.
            elif p.type == 'RandomHalfBody':
                p.prob = 0.0
            elif p.type == 'GetBBoxCenterScale':
                p.padding = conf.get('mmpose_pad', 1.)
            elif p.type == 'RandomBBoxTransform':
                p.shift_factor = conf.trange / conf.imsz[0]
                p.rotate_factor = conf.rrange / 2
                sfactor = 1 / conf.scale_factor_range if (conf.scale_factor_range < 1) else conf.scale_factor_range
                p.scale_factor = [1/sfactor, sfactor]
            elif p.type == 'GenerateTarget':
                p.encoder = cfg.codec
            elif p.type in [ 'TopdownAffine','BottomupResize','BottomupRandomAffine']:
                p.input_size = imsz
                if p.type == 'BottomupResize' and not conf.get('imresize_expand',False):
                    p.resize_mode = 'fit'
            elif p.type == 'BottomupGetHeatmapMask':
                p.get_invalid = conf.get('mmpose_mask_valid',True)

        cfg[dname].dataset.pipeline = cfg[pname]
        cfg[dname].batch_size = conf.batch_size
        if ttype != 'train':
            cfg[ename].ann_file = cfg[dname].dataset.ann_file


    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.test_evaluator


    # if torch.cuda.is_available():
    #     cfg.gpu_ids = [0]
    # else:
    #     cfg.gpu_ids = []
    cfg.work_dir = conf.cachedir

    # Use a fixed seed if APT is in debug mode
    # if APT_interface.IS_APT_IN_DEBUG_MODE:
    #     cfg.seed = 0
    # else:
    #     cfg.seed = None

    # NEEDS REVIEW MK: Does this look right?  CiD cfg does not have
    # cfg.data.samples_per_gpu, it has these three:
    #
    #         cfg.data.train_dataloader['samples_per_gpu']
    #         cfg.data.test_dataloader['samples_per_gpu']
    #         cfg.data.val_dataloader['samples_per_gpu']
    #
    # I'm not sure I've handled this s.t. it will work properly for testing and
    # validation.
    #
    # Idea is that there's an LR and a samples_per_gpu set in the MMPose config
    # file.  But there's a batch_size set in the conf that we want to honor.
    # And according to Goyal et al (2017), the LR should be scaled
    # proportionally if the batch size is changed. So in the cfg, we set the
    # samples_per_gpu to the batch_size as set in the conf, and then we scale
    # the LR specified in the MMPose config file by conf.batch_size /
    # default_samples_per_gpu to be consistent with Goyal et al's
    # recommendation.
    # (We also multiply by a conf.learning_rate_multiplier which is specified
    # in the APT GUI.)
    default_samples_per_gpu = cfg['train_dataloader']['batch_size']

    cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr * conf.learning_rate_multiplier * conf.batch_size / default_samples_per_gpu


    t_steps = 0
    for sched in cfg.param_scheduler:
        if not sched.type == 'MultiStepLR':
            t_steps += sched.end
            continue
        total_epochs = sched.end
        sched.by_epoch = False
        sched.end = conf.dl_steps
        sched.begin = t_steps
        sched.milestones = [int(dd * conf.dl_steps / total_epochs) for dd in sched.milestones]


    cfg.default_hooks.checkpoint.interval = min(conf.save_step, conf.dl_steps)
    cfg.default_hooks.checkpoint.filename_tmpl = run_name + '-{}'
    cfg.default_hooks.checkpoint.by_epoch = False
    cfg.default_hooks.checkpoint.max_keep_ckpts = conf.maxckpt
    cfg.default_hooks.checkpoint.save_best = None

    # Disable flip testing.
    cfg.model.test_cfg.flip_test = conf.get('flip_test',False)

    if hasattr(cfg.model,'keypoint_head') and hasattr(cfg.model.keypoint_head, 'loss_keypoint') and ('with_ae_loss' in cfg.model.keypoint_head.loss_keypoint):
        # setup ae push factor.
        td = PoseTools.json_load(os.path.join(conf.cachedir, conf.trainfilename + '.json'))
        nims = len(td['images'])
        i_id = [s['image_id'] for s in td['annotations']]
        rr = [i_id.count(x) for x in range(nims)]
        push_fac_mul = nims / (10 + nims - rr.count(1))
        for sidx in range(len(cfg.model.keypoint_head.loss_keypoint.with_ae_loss)):
            if cfg.model.keypoint_head.loss_keypoint.with_ae_loss[sidx]:
                cfg.model.keypoint_head.loss_keypoint.push_loss_factor[sidx] = \
                    cfg.model.keypoint_head.loss_keypoint.push_loss_factor[sidx] * push_fac_mul


    # if 'keypoint_head' in cfg.model:
    #     cfg.model.keypoint_head.out_channels = conf.n_classes
    # if conf.n_classes < 8:
    #     try:
    #         if cfg.model.keypoint_head.loss_keypoint[3].type == 'JointsOHKMMSELoss':
    #             cfg.model.keypoint_head.loss_keypoint[3].topk = conf.n_classes
    #     except:
    #         pass

    # disable dynamic loss_scale for fp16 because the hooks in the current mmpose don't support it. MK 20230807. mmpose version is 0.29.0. REMOVE THIS WHEN UPDATING MMPOSE
    # cfg.fp16 = {}
#    cfg.fp16 = None

    J = PoseTools.json_load(cfg.train_dataloader.dataset.ann_file)
    n_train = len(J['images'])
    del cfg.train_cfg['max_epochs']
    cfg.train_cfg.max_iters = conf.dl_steps
    cfg.train_cfg.val_interval = conf.dl_steps+1
    cfg.train_cfg.by_epoch = False


    return cfg


class TraindataHook(Hook):
    def __init__(self, out_file, conf, interval=50):
        self.interval = interval
        self.out_file = out_file
        self.conf = conf
        self.td_data = {'train_loss': [], 'train_dist': [], 'step': [], 'val_loss': [], 'val_dist': []}

    def after_train_iter(self, runner,batch_idx=None,**kwargs):
        if not self.every_n_train_iters(runner, self.interval):
            return
        self.td_data['step'].append(runner.iter + 1)
        rout = runner.log_processor.get_log_after_iter(runner,batch_idx,'train')[0]
        if 'loss' in rout.keys():
            self.td_data['train_loss'].append(rout['loss'].copy())
        else:
            self.td_data['train_loss'].append(rout['all_loss'].copy())
        self.td_data['train_dist'].append(np.nan)
        self.td_data['val_dist'].append(np.nan)
        self.td_data['val_loss'].append(np.nan)

        train_data_file = self.out_file
        with open(train_data_file, 'wb') as td_file:
            pickle.dump([self.td_data, self.conf], td_file, protocol=2)
        json_data = {}
        for x in self.td_data.keys():
            json_data[x] = np.array(self.td_data[x]).astype(np.float64).tolist()
        with open(train_data_file + '.json', 'w') as json_file:
            json.dump(json_data, json_file)


class TrainingDebuggingHook(Hook):
    def __init__(self):
        self.step_from_i = []
        self.training_loss_from_i = []

    def after_train_iter(self, runner,**kwargs):
        step = runner.iter + 1
        self.step_from_i.append(step)
        runner.log_buffer.clear_output()  # Don't want other hooks to interfere with us
        runner.log_buffer.average(1)
        if 'loss' in runner.log_buffer.output.keys():
            training_loss = runner.log_buffer.output['loss'].copy()
        else:
            training_loss = runner.log_buffer.output['all_loss'].copy()
        runner.log_buffer.clear_output()  # Don't want to interfere with other hooks
        self.training_loss_from_i.append(training_loss)
        # TODOALT: Take this out
        print('step: %d, loss: %g' % (step, training_loss))
        pass


def get_handler_by_name(logger, name):
    '''
    Find the named handler in the given logger.
    Returns None if no such handler.
    '''
    did_find_it = False
    for handler in logger.handlers:
        if handler.name == name:
            return handler
    return None


def rectify_log_level_bang(logger, debug=False):
    '''
    Reset the log level for the "log" handler of the root logger to DEBUG or INFO, depending.
    mmpose.models.build_posenet() seems to bork this, so this function fixes it.
    '''
    handler = get_handler_by_name(logger, 'log')
    if handler:
        if debug:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)


class Pose_mmpose(PoseCommon_pytorch.PoseCommon_pytorch):

    def __init__(self, conf, name, **kwargs):
        super().__init__(conf, name)
        self.conf = conf
        self.name = name
        mmpose_net = conf.mmpose_net
        if mmpose_net == 'hrnet':
            # self.cfg_file = 'configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py'
            self.cfg_file = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
        elif mmpose_net == 'multi_hrnet': # aka associative embedding (ae)
            # self.cfg_file = 'configs/bottom_up/hrnet/coco/hrnet_w32_coco_512x512.py'
            self.cfg_file = 'configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py'
        elif mmpose_net == 'higherhrnet':
            raise RuntimeError("MMPose network %s does not seem to be supported in this version of MMPose (%s)" % (
            mmpose_net, mmpose.__version__))
            # self.cfg_file = 'configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_512x512.py'
            self.cfg_file = 'configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/higherhrnet_w32_coco_wholebody_512x512.py'
        elif mmpose_net == 'higherhrnet_2x':
            # self.cfg_file = 'configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_512x512_2xdeconv.py'
            raise RuntimeError("MMPose network %s does not seem to be supported in this version of MMPose (%s)" % (
            mmpose_net, mmpose.__version__))
        elif mmpose_net == 'mspn':
            # self.cfg_file = 'configs/top_down/mspn/coco/mspn50_coco_256x192.py'
            self.cfg_file = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mspn50_8xb32-210e_coco-256x192.py'
        elif mmpose_net == 'hrnet_ap10k':
            self.cfg_file = '/groups/branson/bransonlab/mayank/code/AP-10K/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/hrnet_w32_ap10k_256x256.py'
        elif mmpose_net == 'hrformer':
            self.cfg_file = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-base_8xb32-210e_coco-256x192.py'
            # create a dummy distributed group to keep sync batch norm happy. REMOVE THIS WHEN ENABLING MULTI GPU
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = f'{np.random.randint(10000, 65535)}'
            try:
                torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
            except RuntimeError as e:
                if str(e) == 'trying to initialize the default process group twice!':
                    pass
        elif mmpose_net == 'vitpose':
            self.cfg_file = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py'
            # create a dummy distributed group to keep sync batch norm happy. REMOVE THIS WHEN ENABLING MULTI GPU
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = f'{np.random.randint(10000, 65535)}'
            try:
                torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
            except RuntimeError as e:
                if str(e) == 'trying to initialize the default process group twice!':
                    pass

        elif mmpose_net == 'cid':
            self.cfg_file = 'configs/body_2d_keypoint/cid/coco/cid_hrnet-w32_8xb20-140e_coco-512x512.py'
        elif mmpose_net == 'dekr':
            self.cfg_file = 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py'

        elif mmpose_net =='vitpose':
            self.cfg_file = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
        else:
            assert False, 'Unknown mmpose net type'

        poseConfig.conf = conf
        self.cfg = create_mmpose_cfg(self.conf, self.cfg_file, name)

    def get_td_file(self):
        if self.name == 'deepnet':
            td_name = os.path.join(self.conf.cachedir, 'traindata')
        else:
            td_name = os.path.join(self.conf.cachedir, self.conf.expname + '_' + self.name + '_traindata')
        return td_name

    def train_wrapper(self, restore=False, model_file=None, debug=False, logger=None):
        if logger is None:
            logger = logging.getLogger()  # the root logger
        cfg = self.cfg

        logger.info(f'Config:\n{cfg.pretty_text}')

        # Hack to work around what is seemingly a bug in MMPose 0.29.0...
        try:
            if cfg.model.type == 'CID':
                del cfg.model.keypoint_head['out_channels']
        except:
            pass


        cfg.resume = False
        cfg.load_from = None
        if model_file is not None:
            cfg.load_from = model_file
        elif restore:
            cfg.resume = True
            cfg.load_from = self.get_latest_model_file()

        runner = Runner.from_cfg(cfg)

        td_hook = TraindataHook(self.get_td_file(), self.conf, self.conf.display_step)
        runner.register_hook(td_hook)

        # Add a hook object for debugging
        trainingDebuggingHook = TrainingDebuggingHook()
        # runner.register_hook(trainingDebuggingHook)

        # runner.max_iters = steps
        logging.debug("Running the runner...")

        # ALTTODO: Get rid of this debugging code
        # thangs = {'runner': runner, 'workflow': cfg.workflow, 'cfg': cfg} #'dataloader': dataloader,
        # import pickle
        # with open('/groups/branson/bransonlab/taylora/apt/compare-cid-in-apt-to-plain-cid/pre-run-variables-in-apt.pkl',
        #           'wb') as f:
        #     pickle.dump(thangs, f)

        if debug:
            with torch.autograd.detect_anomaly():
                runner.train()
        else:
            runner.train()
        logging.debug("Runner is finished running.")


    def get_latest_model_file(self):
        model_file_ptn = os.path.join(self.conf.cachedir, self.name + '-[0-9]*')
        files = glob.glob(model_file_ptn)
        files.sort(key=os.path.getmtime)
        if len(files) > 0:
            model_file = files[-1]
        else:
            model_file = None

        return model_file

    def get_pred_fn(self, model_file=None, max_n=None, imsz=None):

        cfg = self.cfg
        conf = self.conf

        assert not conf.is_multi, 'This prediction function is only for single animal (top-down)'

        model_file = self.get_latest_model_file() if model_file is None else model_file
        model = init_model(cfg,model_file)
        scope = model.cfg.get('default_scope', 'mmpose')
        if scope is not None:
            init_default_scope(scope)

        if conf.batch_size > 4:
            use_pool = True
            pool = mp.Pool(processes=32)
        else:
            use_pool = False
            pool = None
        test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

        rgb2bgr = False
        if 'data_preprocessor' in cfg.model:
            if 'bgr_to_rgb' in cfg.model.data_preprocessor:
                rgb2bgr = cfg.model.data_preprocessor.bgr_to_rgb

        def pred_fn(ims, retrawpred=False):
            pose_results = np.ones([ims.shape[0], conf.n_classes, 2]) * np.nan
            conf_res = np.zeros([ims.shape[0], conf.n_classes])
            all_data = []
            for b in range(ims.shape[0]):
                if ims.shape[3] == 1:
                    ii = np.tile(ims[b, ...], [1, 1, 3])
                else:
                    ii = ims[b, ...]
                    if rgb2bgr:
                        ii = ii[..., ::-1]

                # prepare data
                data = {'img': ii.astype('uint8'),
                        'bbox': np.array([0,0,conf.imsz[1],conf.imsz[0]])[None],
                        'bbox_score': np.ones(1, dtype='float32'),
                        }
                data.update(model.dataset_meta)
                all_data.append(data)

            if use_pool:
                all_data = pool.map(test_pipeline, all_data)
            else:
                all_data = [test_pipeline(d) for d in all_data]
            all_data = pseudo_collate(all_data)

            with torch.no_grad():
                model_out = model.test_step(all_data)

            for ndx,pred in enumerate(model_out):
                pose_results[ndx,:] = pred.pred_instances.keypoints.copy()
                conf_res[ndx,:] = pred.pred_instances.keypoint_scores.copy()

            ret_dict = {'locs': pose_results, 'conf': conf_res}
            return ret_dict


        def close_fn():
            torch.cuda.empty_cache()
            pool.close()

        return pred_fn, close_fn, model_file
