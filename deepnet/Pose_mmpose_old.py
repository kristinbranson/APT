import pathlib
import os
#from urllib.request import HTTPPasswordMgrWithDefaultRealm
#import sys
#sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(),'mmpose'))
import mmcv
#from mmcv import Config
import mmpose
#from mmpose.models import build_posenet
import poseConfig
import copy
#from mmpose import __version__
#from mmcv.utils import get_git_hash
from mmpose.datasets import build_dataloader, build_dataset
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, IterBasedRunner, Hook, load_checkpoint
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.core import (DistEvalHook, EvalHook, Fp16OptimizerHook, build_optimizers)
import logging
import time
import PoseCommon_pytorch
#from PoseCommon_pytorch import PoseCommon_pytorch
#from mmcv.runner import init_dist, set_random_seed
import pickle
import json
#from mmpose.core import wrap_fp16_model
#from mmpose.datasets.pipelines import Compose
#from mmcv.parallel import collate, scatter
import glob
from mmcv.utils.config import ConfigDict
import PoseTools
from mmpose.datasets.pipelines.shared_transform import ToTensor, NormalizeTensor
import numpy as np
import APT_interface
import cv2
import multiprocessing as mp

## Topdown dataset
from mmpose.datasets.builder import DATASETS, PIPELINES
from mmpose.datasets.datasets.top_down.topdown_coco_dataset import TopDownCocoDataset
from xtcocotools.coco import COCO


###### Bug fix for np.int in multi_hrnet 8 Feb 2024

from mmpose.datasets.pipelines.bottom_up_transform import HeatmapGenerator,BottomUpGenerateHeatmapTarget,JointsEncoder,BottomUpGenerateTarget
class myHMAP(HeatmapGenerator):
    def __init__(self, output_size, num_joints, sigma=-1, use_udp=False):
        if not isinstance(output_size, np.ndarray):
            output_size = np.array(output_size)
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size],
                                        dtype=np.int32)
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_size.prod()**0.5 / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        self.use_udp = use_udp
        if use_udp:
            self.x = np.arange(0, size, 1, np.float32)
            self.y = self.x[:, None]
        else:
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            x0, y0 = 3 * sigma + 1, 3 * sigma + 1
            self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

class myJoint(JointsEncoder):
    def __init__(self, max_num_people, num_joints, output_size, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        if not isinstance(output_size, np.ndarray):
            output_size = np.array(output_size)
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size],
                                        dtype=np.int32)
        self.tag_per_joint = tag_per_joint

@PIPELINES.register_module(name='BottomUpGenerateTarget',force=True)
class myBUH(BottomUpGenerateTarget):
    def _generate(self, num_joints, heatmap_size):
        """Get heatmap generator and joint encoder."""
        heatmap_generator = [
            myHMAP(output_size, num_joints, self.sigma, self.use_udp)
            for output_size in heatmap_size
        ]
        joints_encoder = [
            myJoint(self.max_num_people, num_joints, output_size, True)
            for output_size in heatmap_size
        ]
        return heatmap_generator, joints_encoder

## End bug fix for multi_hrnet

class TrainDataHookDummy(mmcv.runner.Hook):
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


@DATASETS.register_module()
class TopDownAPTDataset(TopDownCocoDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False,
                 dataset_info=None):
        # Overriding topdowncoocodataset init code because it is bad and awful with hardcoded values.
        super(TopDownCocoDataset, self).__init__(ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode, dataset_info=dataset_info)
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

        logging.info(f'=> num_images: {self.num_images}')
        logging.info(f'=> load {len(self.db)} samples')


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
            pairs.append([int(kk),int(conf.flipLandmarkMatches[kk])])
            done.append(int(kk))
            done.append(int(conf.flipLandmarkMatches[kk]))
        self.ann_info['flip_pairs'] = pairs
        self.ann_info['joint_weights'] = np.ones([conf.n_classes])
        self.sigmas = np.ones([conf.n_classes])*0.6/10.0

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
#from mmpose.datasets.registry import PIPELINES
from mmpose.datasets.builder import PIPELINES

@PIPELINES.register_module()
class APTtransform:
    """Data augmentation using APT's posetools.

    """

    def __init__(self,distort):
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
            assert  not results['ann_info']['scale_aware_sigma'], 'APT doesnt support this'
            for jndx in range(len(joints)-1):
                # assert len(joints) ==2, "APT Transform is tested only for at most two scales"

                assert np.allclose(joints[jndx],joints[jndx+1],equal_nan=True), "APT transform is tested only for two identical scale inputs"
            jlen = len(joints)
            joints_in = joints[0][...,:2]
            occ_in = joints[0][...,2]
            joints_in[occ_in<0.5,:] = -100000
            image,joints_out,mask_out,occ = pt.preprocess_ims(image[np.newaxis,...],
                                                              joints_in[np.newaxis,...],
                                                              conf,
                                                              self.distort,
                                                              conf.rescale,
                                                              mask=mask[0][None,...],
                                                              occ=occ_in)
            image = image.astype('float32')
            joints_out_occ1 = np.isnan(joints_out[0, ..., 0:1]) | (joints_out[0, ..., 0:1] < -1000)
            joints_out_occ = occ<0.5
            assert np.array_equal(joints_out_occ1,joints_out_occ), 'Occlusion from processing and from joints should be equal'
            joints_out = np.concatenate([joints_out[0,...],(~joints_out_occ)*2],axis=-1)
            in_sz = results['ann_info']['image_size']
            out_sz = results['ann_info']['heatmap_size']
            assert all([round(in_sz/o)==in_sz/o for o in out_sz]), 'Output sizes should be integer multiples of input sizes'
            outs = [int(round(in_sz/o)) for o in out_sz]
            results['joints'] = [joints_out * osz / in_sz for osz in out_sz]
            results['mask'] = [mask_out[0,::o,::o]>0.5 for o in outs]

        else:
            image, joints, occ_in = results['img'], results['joints_3d'], results['joints_3d_visible']
            assert joints[:,2].max() < 0.00001, 'APT does not work 3d'
            occ_in = occ_in[:,0]
            joints_in = joints[:,:2]
            joints_in[occ_in<0.5,:] = -100000

            image,joints_out, occ = pt.preprocess_ims(image[np.newaxis,...],joints_in[np.newaxis,...],conf,self.distort,conf.rescale,occ=1-occ_in)
            isz = results['ann_info']['image_size']
            if image.shape[1] != isz[1] or image.shape[2] != isz[0]:
                #print('Resizing image to ',isz)
                image = cv2.resize(image[0,...],(isz[0],isz[1]))
                image = image[None,...]
            image = image.astype('float32')
            joints_out_occ1 = np.isnan(joints_out[0,...,0:1]) | (joints_out[0,...,0:1]<-1000)
            joints_out_occ = occ[0,:,None]>0.5
            # if conf.check_bounds_distort:
            assert np.array_equal(joints_out_occ1,joints_out_occ), 'Occlusion from processing and from joints should be equal'

            results['joints_3d'] = np.concatenate([joints_out[0,...],np.zeros_like(joints_out[0,:,:1])],1)
            results['joints_3d_visible'] = np.concatenate([1-joints_out_occ,1-joints_out_occ,np.zeros_like(joints_out_occ)],1)

        results['img'] = np.clip(image[0,...],0,255).astype('uint8')
        results['center'] = np.array([image.shape[2]/2,image.shape[1]/2],dtype='float32')
        results['scale'] = np.array([1.0,1.0],dtype='float32')
        results['flip_pairs'] = []
        return results



def create_mmpose_cfg(conf, mmpose_config_file, run_name, zero_seeds=False, img_prefix_override=None, debug=False):
    #curdir = pathlib.Path(__file__).parent.absolute()
    mmpose_init_file_path = mmpose.__file__
    mmpose_dir = os.path.dirname(mmpose_init_file_path)
    dot_mim_folder_path = os.path.join(mmpose_dir, '.mim')  # this feels not-robust
    mmpose_config_file_path = os.path.join(dot_mim_folder_path, mmpose_config_file)
    data_bdir = conf.cachedir

    cfg = mmcv.Config.fromfile(mmpose_config_file_path)
    default_im_sz = cfg.data_cfg.image_size
    default_hm_sz = cfg.data_cfg.heatmap_size
    cfg.data_cfg.image_size = [int(c / conf.rescale/32)*32 for c in conf.imsz[::-1]]  # conf.imsz[0]
    if conf.is_multi:
        if conf.mmpose_net == 'cid' and conf.multi_loss_mask:
            raise RuntimeError('For the CiD model, if a frame has any labeled animals, all animals in the frame must be labeled.  Therefore in the training parameters "Unlabeled animals present" (a.k.a. multi_loss_mask) must be false.')
        # imsz = cfg.data_cfg.image_size[0]
        imsz = min(cfg.data_cfg.image_size)
        cfg.data_cfg.image_size = imsz
        cfg.data_cfg.heatmap_size = [int(h/default_im_sz*imsz) for h in default_hm_sz]
        cfg.model.train_cfg.img_size = cfg.data_cfg.image_size
        cfg.model.test_cfg.img_size = cfg.data_cfg.image_size
        cfg.model.keypoint_head.num_joints = conf.n_classes
        if hasattr(cfg.model.keypoint_head, 'loss_keypoint') :
            cfg.model.keypoint_head.loss_keypoint.num_joints = conf.n_classes
    else:
        assert default_im_sz[0]/default_hm_sz[0] == 4, 'Single animal mmpose is tested only for hmaps downsampled by 4'
        cfg.data_cfg.heatmap_size = [csz // 4 for csz in cfg.data_cfg.image_size]
        cfg.data_cfg.use_gt_bbox = True
        if 'keypoint_head' in cfg.model and 'out_shape' in cfg.model.keypoint_head:
            cfg.model.keypoint_head.out_shape = [csz // 4 for csz in cfg.data_cfg.image_size[::-1]]

    cfg.data_cfg.num_joints = conf.n_classes
    cfg.data_cfg.dataset_channel = [list(range(conf.n_classes))]
    cfg.data_cfg.inference_channel = list(range(conf.n_classes))
    cfg.data_cfg.num_output_channels = conf.n_classes

    for ttype in ['train', 'val', 'test']:
        #name = ttype if ttype is not 'test' else 'val'
        name = 'val' if ttype=='test' else ttype  # avoids warning about 'is not' with literal
        fname = conf.trainfilename if ttype == 'train' else conf.valfilename
        cfg.data[ttype].ann_file = os.path.join(data_bdir, f'{fname}.json')
        file = os.path.join(data_bdir, f'{fname}.json')
        if img_prefix_override:
            cfg.data[ttype].img_prefix = img_prefix_override
            #cfg.data[ttype].img_prefix = ''
        else:
            cfg.data[ttype].img_prefix = os.path.join(data_bdir, name)
        cfg.data[ttype].data_cfg = cfg.data_cfg

        key_info = {}
        for ndx in range(conf.n_classes):
            key_info[ndx] = {'name': f'{ndx}', 'id': ndx, 'color': [128, 128, 128], 'type': 'upper', 'swap': ''}
        for kk in conf.flipLandmarkMatches.keys():
            key_info[int(kk)]['swap'] = f'{conf.flipLandmarkMatches[kk]}'
        skel_info = {}
        for ndx, cure in enumerate(conf.op_affinity_graph):
            skel_info[ndx] = {'link': (f'{cure[0]}', f'{cure[1]}'), 'id': ndx, 'color': [128, 128, 128]}
        dataset_info = {'dataset_name': 'apt',
                        'keypoint_info': key_info,
                        'skeleton_info': skel_info,
                        'joint_weights': [1., ] * conf.n_classes,
                        'sigmas': [0.06, ] * conf.n_classes,
                        'paper_info': {}}
        cfg.data[ttype].dataset_info = dataset_info

        if conf.is_multi:
            cfg.data[ttype].type = 'BottomUpAPTDataset'
            cfg.model.train_cfg.num_joints = conf.n_classes
            cfg.model.test_cfg.num_joints = conf.n_classes
            cfg.model.test_cfg.max_num_people = conf.max_n_animals
            cfg.model.test_cfg.min_num_people = conf.min_n_animals
            cfg.model.test_cfg.dist_grouping = True
            cfg.model.test_cfg.detection_threshold = conf.multi_mmpose_detection_threshold

        else:
            pass
            # cfg.data[ttype].type = 'TopDownAPTDataset'

        # Remove some of the transforms.
        in_pipe = cfg.data[ttype].pipeline
        cfg.data[ttype].pipeline = [ii for ii in in_pipe if ii.type not in ['TopDownHalfBodyTransform']]

        if conf.get('mmpose_use_apt_augmentation',False):
            # substitute their augmentation pipeline with ours
            if ttype =='train':
                if conf.is_multi:
                    assert \
                        (cfg.data[ttype].pipeline[1].type == 'BottomUpRandomAffine') and \
                        (cfg.data[ttype].pipeline[2].type == 'BottomUpRandomFlip'), \
                        'Unusual mmpose augmentation pipeline cannot be substituted by APT augmentation'
                    cfg.data[ttype].pipeline[2:3] = []
                    cfg.data[ttype].pipeline[1] = ConfigDict({'type':'APTtransform','distort':True})
                elif (cfg.data[ttype].pipeline[1].type == 'TopDownRandomFlip'):
                    # old style top down pipeline
                    assert \
                        (cfg.data[ttype].pipeline[1].type == 'TopDownRandomFlip') and \
                        (cfg.data[ttype].pipeline[2].type =='TopDownGetRandomScaleRotation') and \
                        (cfg.data[ttype].pipeline[3].type =='TopDownAffine'), \
                        'Unusual mmpose augmentation pipeline cannot be substituted by APT augmentation'
                    cfg.data[ttype].pipeline[2:4] = []
                    cfg.data[ttype].pipeline[1] = ConfigDict({'type':'APTtransform','distort':True})
                else:
                    assert \
                        (cfg.data[ttype].pipeline[1].type == 'TopDownGetBboxCenterScale') and \
                        (cfg.data[ttype].pipeline[2].type == 'TopDownRandomShiftBboxCenter') and \
                        (cfg.data[ttype].pipeline[3].type == 'TopDownRandomFlip') and \
                        (cfg.data[ttype].pipeline[4].type == 'TopDownGetRandomScaleRotation') and \
                        (cfg.data[ttype].pipeline[5].type == 'TopDownAffine') , \
                            'Unusual mmpose augmentation pipeline cannot be substituted by APT augmentation'
                    cfg.data[ttype].pipeline[2:6] = []
                    cfg.data[ttype].pipeline[1] = ConfigDict({'type': 'APTtransform', 'distort': True})
                    # else:
        #     assert conf.rescale == 1, 'MMpose aug with rescale has not been implemented'

        for p in cfg.data[ttype].pipeline:
            if p.type == 'BottomUpRandomAffine':
                p.rot_factor = conf.rrange/2
                sfactor = 1/conf.scale_factor_range if (conf.scale_factor_range < 1) else conf.scale_factor_range
                p.scale_factor = [1/sfactor, sfactor]
                p.trans_factor = conf.trange/conf.imsz[0]*200
                # translation in mmpose is relative to 200px standard size.
            elif p.type == 'TopDownGetRandomScaleRotation':
                p.rot_factor = conf.rrange/2
                # p.rot_prob = 1.
                sfactor = 1/conf.scale_factor_range if (conf.scale_factor_range < 1) else conf.scale_factor_range
                p.scale_factor = sfactor-1
            elif p.type in ['TopDownRandomFlip','BottomUpRandomFlip']:
                p.flip_prob = 0.5 if (conf.horz_flip or conf.vert_flip) else 0.
            elif p.type == 'TopDownHalfBodyTransform':
                p.prob_half_body = 0.0
            elif p.type == 'TopDownGetBboxCenterScale':
                p.padding = conf.get('mmpose_pad',1.)
            elif p.type == 'TopDownRandomShiftBboxCenter':
                p.shift_factor = conf.trange/conf.imsz[0]


    if torch.cuda.is_available():
        cfg.gpu_ids = [0]
    else:
        cfg.gpu_ids = []
    cfg.work_dir = conf.cachedir

    # Use a fixed seed if APT is in zero_seeds mode    
    if zero_seeds:
        cfg.seed = 0
    else:
        cfg.seed = None

    # Don't use a separate thread for loading data if in debug mode
    if debug:
        cfg.data.workers_per_gpu = 0

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
    actual_batch_size = conf.batch_size
    default_lr = cfg.optimizer.lr 
    if hasattr(cfg.data, 'samples_per_gpu') :
        default_samples_per_gpu = cfg.data.samples_per_gpu
        cfg.data.samples_per_gpu = actual_batch_size
    else :
        # default_samples_per_gpu is only used to set the learning rate, so we use the value in cfg that pertains to training.
        default_samples_per_gpu = cfg.data.train_dataloader['samples_per_gpu']
        cfg.data.train_dataloader['samples_per_gpu'] = actual_batch_size
    new_lr = conf.learning_rate_multiplier * default_lr * actual_batch_size / default_samples_per_gpu
    cfg.optimizer.lr = new_lr

    assert cfg.lr_config.policy == 'step', 'Works only for steplr for now'
    if cfg.lr_config.policy == 'step':
        def_epochs = cfg.total_epochs
        def_steps = cfg.lr_config.step
        cfg.lr_config.step = [int(dd/def_epochs*conf.dl_steps) for dd in def_steps]

    # pretrained weights are now urls. So torch does the mapping
    # cfg.model.pretrained = os.path.join('mmpose',cfg.model.pretrained)

    cfg.checkpoint_config.interval = min(conf.save_step,conf.dl_steps)
    cfg.checkpoint_config.filename_tmpl = run_name + '-{}'
    cfg.checkpoint_config.by_epoch = False
    cfg.checkpoint_config.max_keep_ckpts = conf.maxckpt

    # Disable flip testing.
    cfg.model.test_cfg.flip_test = False

    if hasattr(cfg.model.keypoint_head, 'loss_keypoint') and ('with_ae_loss' in cfg.model.keypoint_head.loss_keypoint):
        # setup ae push factor.
        td = PoseTools.json_load(os.path.join(conf.cachedir, conf.trainfilename + '.json'))
        nims = len(td['images'])
        i_id = [s['image_id'] for s in td['annotations']]
        rr = [i_id.count(x) for x in range(nims)]
        push_fac_mul = nims/(10+nims-rr.count(1))
        for sidx in range(len(cfg.model.keypoint_head.loss_keypoint.with_ae_loss)):
            if cfg.model.keypoint_head.loss_keypoint.with_ae_loss[sidx]:
                cfg.model.keypoint_head.loss_keypoint.push_loss_factor[sidx] = \
                    cfg.model.keypoint_head.loss_keypoint.push_loss_factor[sidx] * push_fac_mul


    if 'keypoint_head' in cfg.model and 'out_channels' in cfg.model.keypoint_head:
        cfg.model.keypoint_head.out_channels = conf.n_classes
    if conf.n_classes < 8:
        try:
            if cfg.model.keypoint_head.loss_keypoint[3].type == 'JointsOHKMMSELoss':
                cfg.model.keypoint_head.loss_keypoint[3].topk = conf.n_classes
        except:
            pass

    # disable dynamic loss_scale for fp16 because the hooks in the current mmpose don't support it. MK 20230807. mmpose version is 0.29.0. REMOVE THIS WHEN UPDATING MMPOSE
    cfg.fp16 = None
    return cfg

class TraindataHook(Hook):
    def __init__(self,out_file,conf,interval=50):
        self.interval = interval
        self.out_file = out_file
        self.conf = conf
        self.td_data = {'train_loss':[],'train_dist':[],'step':[],'val_loss':[],'val_dist':[]}

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner,self.interval):
           return
        self.td_data['step'].append(runner.iter + 1)
        runner.log_buffer.average(self.interval)
        if 'loss' in runner.log_buffer.output.keys():
            cur_loss = runner.log_buffer.output['loss'].copy()
        else:
            cur_loss = runner.log_buffer.output['all_loss'].copy()
        self.td_data['train_loss'].append(cur_loss)
        if np.isnan(cur_loss):
            raise RuntimeError(f"Training loss is NaN at step {runner.iter + 1}. Please restart training")

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

    def after_train_iter(self, runner):        
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
        # print('step: %d, loss: %g' % (step, training_loss))
        # pass


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

    def __init__(self, conf, name, zero_seeds=False, img_prefix_override=None, debug=False, **kwargs):
        super().__init__(conf,name)
        self.conf = conf
        self.name = name
        mmpose_net = conf.mmpose_net
        if mmpose_net == 'hrnet':
            #self.cfg_file = 'configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py'
            self.cfg_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
        elif mmpose_net == 'multi_hrnet':
            #self.cfg_file = 'configs/bottom_up/hrnet/coco/hrnet_w32_coco_512x512.py'
            self.cfg_file = 'configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/hrnet_w32_coco_wholebody_512x512.py'
        elif mmpose_net == 'higherhrnet':
            #self.cfg_file = 'configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_512x512.py'
            self.cfg_file = 'configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/higherhrnet_w32_coco_wholebody_512x512.py'
        elif mmpose_net == 'higherhrnet_2x':
            #self.cfg_file = 'configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_512x512_2xdeconv.py'
            raise RuntimeError("MMPose network %s does not seem to be supported in this version of MMPose (%s)" % (mmpose_net, mmpose.__version__) )
        elif mmpose_net =='mspn':
            #self.cfg_file = 'configs/top_down/mspn/coco/mspn50_coco_256x192.py'
            self.cfg_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mspn50_coco_256x192.py'
        elif mmpose_net == 'hrnet_ap10k':
            self.cfg_file = '/groups/branson/bransonlab/mayank/code/AP-10K/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/hrnet_w32_ap10k_256x256.py'
        elif mmpose_net == 'hrformer':
            self.cfg_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_256x192.py'
            # create a dummy distributed group to keep sync batch norm happy. REMOVE THIS WHEN ENABLING MULTI GPU
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = f'{np.random.randint(10000, 65535)}'
            try:
                print('Initializing process group')
                torch.distributed.init_process_group(backend='gloo',rank=0,world_size=1)
            except RuntimeError as e:
                if str(e) == 'trying to initialize the default process group twice!':
                    pass
        elif mmpose_net == 'cid':
            self.cfg_file = 'configs/body/2d_kpt_sview_rgb_img/cid/coco/hrnet_w32_coco_512x512.py'
        elif mmpose_net == 'dekr':
            self.cfg_file = 'configs/body/2d_kpt_sview_rgb_img/dekr/crowdpose/hrnet_w32_crowdpose_512x512.py'
        else:
            assert False, 'Unknown mmpose net type'

        poseConfig.conf = conf
        cfg = create_mmpose_cfg(self.conf, self.cfg_file, name, zero_seeds=zero_seeds, img_prefix_override=img_prefix_override, debug=debug)
        self.cfg = cfg


    def get_td_file(self):
        if self.name =='deepnet':
            td_name = os.path.join(self.conf.cachedir,'traindata')
        else:
            td_name = os.path.join(self.conf.cachedir, self.conf.expname + '_' + self.name + '_traindata')
        return td_name


    def train_wrapper(self, restore=False, model_file=None, debug=False, logger=None):
        # From mmpose/tools/train.py
        if logger is None:
            logger = logging.getLogger()  # the root logger
        cfg = self.cfg

        logger.info(f'Config:\n{cfg.pretty_text}')

        # Hack to work around what is seemingly a bug in MMPose 0.29.0...
        try :
            if cfg.model.type == 'CID' :
                del cfg.model.keypoint_head['out_channels']
        except :
            pass
        model = mmpose.models.build_posenet(cfg.model)  # messes up logging!
        rectify_log_level_bang(logger, debug)
        dataset = mmpose.datasets.build_dataset(cfg.data.train)
        if len(dataset)==0:
            logging.warning('The dataset has zero elements.  Seems bad...')
        else:
            if debug:
                n_to_check = min(10, len(dataset))
                logging.info('Testing access to the first %s elements of the dataset...' % n_to_check)
                for i in range(n_to_check):
                    datum = dataset[i]  # Test if we can access the first few elements of the dataset
                logging.info('Successfully read the first %d elements of the dataset.' % n_to_check)
        datasets = [ dataset ]

        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(mmpose.datasets.build_dataset(val_dataset))

        if cfg.checkpoint_config is not None:
            # save mmpose version, config file content
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmpose_version=mmpose.__version__ + mmcv.utils.get_git_hash(digits=7),
                config=cfg.pretty_text,
            )


        # Rest is from mmpose/apis/train.py

        validate = False
        distributed = len(cfg.gpu_ids)>1
        if distributed:
            mmcv.runner.init_dist('pytorch', **cfg.dist_params)

        meta = None
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if model_file is not None:
            cfg.load_from = model_file
        elif restore:
            cfg.resume_from = self.get_latest_model_file()

        dataloader_setting = dict(
            samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids) if len(cfg.gpu_ids)>0 else 1,
            dist=distributed,
            seed=cfg.seed)
        dataloader_setting = dict(dataloader_setting, **cfg.data.get('train_dataloader', {}))

        dataloaders = [ build_dataloader(ds, **dataloader_setting) for ds in datasets ]

        # determine wether use adversarial training precess or not
        use_adverserial_train = cfg.get('use_adversarial_train', False)

        # put model on gpus
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', True)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel

            if use_adverserial_train:
                # Use DistributedDataParallelWrapper for adversarial training
                model = DistributedDataParallelWrapper(
                    model,
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                lr = os.environ['LOCAL_RANK']
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[lr], #torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
        else:
            if torch.cuda.is_available():
                model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

        # build runner
        optimizer = build_optimizers(model, cfg.optimizer)
        if self.conf.get('mmpose_use_epoch_runner', False):
            get_runner = EpochBasedRunner
            steps = self.conf.dl_steps // len(dataloaders[0])
        else:
            get_runner = IterBasedRunner
            steps = self.conf.dl_steps


        runner = get_runner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta, 
            max_iters=steps)
        # an ugly workaround to make .log and .log.json filenames the same
        runner.timestamp = timestamp

        if use_adverserial_train:
            # The optimizer step process is included in the train_step function
            # of the model, so the runner should NOT include optimizer hook.
            optimizer_config = None
        else:
            # fp16 setting
            fp16_cfg = cfg.get('fp16', None)
            if fp16_cfg is not None:
                optimizer_config = Fp16OptimizerHook(
                    **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
            elif distributed and 'type' not in cfg.optimizer_config:
                optimizer_config = OptimizerHook(**cfg.optimizer_config)
            else:
                optimizer_config = cfg.optimizer_config

        # register hooks
        runner.register_training_hooks(cfg.lr_config,
                                       optimizer_config,
                                       cfg.checkpoint_config,
                                       cfg.log_config,
                                       cfg.get('momentum_config', None))
        if distributed:
            runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
        if validate:
            eval_cfg = cfg.get('evaluation', {})
            val_dataset = mmpose.datasets.build_dataset(cfg.data.val, dict(test_mode=True))
            dataloader_setting = dict(
                samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
                workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
                # cfg.gpus will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                shuffle=False)
            dataloader_setting = dict(dataloader_setting,
                                      **cfg.data.get('val_dataloader', {}))
            val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

        #td_hook = TrainDataHookDummy()
        td_hook = TraindataHook(self.get_td_file(), self.conf, self.conf.display_step)
        runner.register_hook(td_hook)

        # Add a hook object for debugging
        trainingDebuggingHook = TrainingDebuggingHook()
        runner.register_hook(trainingDebuggingHook)

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        #runner.max_iters = steps    
        logging.debug("Running the runner...")

        # If debugging, write out the cfg just before running
        if debug:
            dataloader = dataloaders[0]
            thangs = { 'runner': runner, 'dataloader': dataloader, 'workflow': cfg.workflow, 'cfg': cfg }
            import pickle
            pickle_file_leaf_name = 'training-pre-running-variables.pkl'
            json_label_file_path = self.conf.labelfile
            pickle_folder_path = os.path.dirname(json_label_file_path)
            pickle_file_path = os.path.join(pickle_folder_path, pickle_file_leaf_name)
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(thangs, f)

        if debug:
            with torch.autograd.detect_anomaly():
                runner.run(dataloaders, cfg.workflow)
        else:
            runner.run(dataloaders, cfg.workflow)
        logging.debug("Runner is finished running.")


    def get_latest_model_file(self):
        model_file_ptn = os.path.join(self.conf.cachedir,self.name + '-[0-9]*')
        files = glob.glob(model_file_ptn)
        files.sort(key=os.path.getmtime)
        if len(files)>0:
            model_file = files[-1]
        else:
            model_file = None

        return  model_file

    def get_pred_fn(self, model_file=None,max_n=None,imsz=None):

        cfg = self.cfg
        conf = self.conf

        assert not conf.is_multi, 'This prediction function is only for single animal (top-down)'

        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        model = mmpose.models.build_posenet(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        model_file = self.get_latest_model_file() if model_file is None else model_file
        if fp16_cfg is not None:
            mmpose.core.wrap_fp16_model(model)

        _ = load_checkpoint(model, model_file, map_location='cpu')
        logging.info(f'Loaded model from {model_file}')
        if torch.cuda.is_available():
            model = MMDataParallel(model,device_ids=[0])
        model = model.eval()
        # build part of the pipeline to do the same preprocessing as training
        test_pipeline = cfg.test_pipeline[2:]
        test_pipeline = mmpose.datasets.pipelines.Compose(test_pipeline)
        device = next(model.parameters()).device

        pairs = []
        done = []
        for kk in conf.flipLandmarkMatches.keys():
            if int(kk) in done:
                continue
            pairs.append([int(kk),int(conf.flipLandmarkMatches[kk])])
            done.append(int(kk))
            done.append(int(conf.flipLandmarkMatches[kk]))

        to_tensor_trans = ToTensor()
        norm_trans = NormalizeTensor(cfg.test_pipeline[-2]['mean'],cfg.test_pipeline[-2]['std'])
        pool = mp.Pool(processes=32)

        def pred_fn(ims,retrawpred=False):
            pose_results = np.ones([ims.shape[0],conf.n_classes,2])*np.nan
            conf_res = np.zeros([ims.shape[0],conf.n_classes])
            all_hmaps = []
            all_data = []
            for b in range(ims.shape[0]):
                if ims.shape[3] == 1:
                    ii = np.tile(ims[b,...],[1,1,3])
                else:
                    ii = ims[b,...]
                # prepare data
                data = {'img': ii.astype('uint8'),
                    'dataset': 'coco',
                        'joints_3d': np.ones([conf.n_classes,2])*30,
                        'joints_3d_visible': np.ones([conf.n_classes,1]),
                        'center':np.array([ii.shape[1]/2,ii.shape[0]/2]),
                        'scale':(np.array([ii.shape[1],ii.shape[0]]))/200,
                        'rotation':0.,
                        'ann_info': {
                        'image_size':
                            cfg.data_cfg['image_size'],
                        'num_joints':
                            cfg.data_cfg['num_joints'],
                        'image_file':'',
                        'bbox_score':[0],
                        'bbox_id':[b],
                        'flip_pairs':pairs

                    }
                }
                all_data.append(data)

            all_data = pool.map(test_pipeline,all_data)
            imgs = torch.tensor(np.stack([d['img'] for d in all_data]))
            img_metas = [d['img_metas'].data for d in all_data]

            with torch.no_grad():
                model_out = model(return_loss=False, img=imgs, img_metas=img_metas)

            all_preds = model_out['preds']
            heatmap = model_out['output_heatmap']
            pose_results = all_preds[:,:,:2]
            conf_res = all_preds[:,:,2].copy()

            ret_dict = {'locs':pose_results,'conf':conf_res}
            if retrawpred:
                ret_dict['hmap']=heatmap
            return ret_dict


        def pred_fn_single(ims,retrawpred=False):

            pose_results = np.ones([ims.shape[0],conf.n_classes,2])*np.nan
            conf_res = np.zeros([ims.shape[0],conf.n_classes])

            # pad_y = int(conf.imsz[0]*0.25)
            # pad_sz_y1 = int( pad_y/2)
            # pad_sz_y2 = pad_y-pad_sz_y1
            #
            # pad_x = int(conf.imsz[1]*0.25)
            # pad_sz_x1 = int(pad_x/2)
            # pad_sz_x2 = pad_x-pad_sz_x1
            #
            # ims = np.pad(ims,[[0,0],[pad_sz_y1,pad_sz_y2],[pad_sz_x1,pad_sz_x2],[0,0]])
            # ims, _ = PoseTools.preprocess_ims(ims.copy(),np.zeros([ims.shape[0],conf.n_classes,2]),conf,False,conf.rescale,interp_method=cv2.INTER_CUBIC)
            all_hmaps = []
            for b in range(ims.shape[0]):
                if ims.shape[3] == 1:
                    ii = np.tile(ims[b,...],[1,1,3])
                else:
                    ii = ims[b,...]
                # prepare data
                data = {'img': ii.astype('uint8'),
                    'dataset': 'coco',
                        'joints_3d': np.ones([conf.n_classes,2])*30,
                        'joints_3d_visible': np.ones([conf.n_classes,1]),
                        'center':np.array([ii.shape[1]/2,ii.shape[0]/2]),
                        'scale':(np.array([ii.shape[1],ii.shape[0]]))/200,
                        'rotation':0.,
                        'ann_info': {
                        'image_size':
                            cfg.data_cfg['image_size'],
                        'num_joints':
                            cfg.data_cfg['num_joints'],
                        'image_file':'',
                        'bbox_score':[0],
                        'flip_pairs':pairs
                    }
                }

                # replace the test_pipeline with ours
                # data = test_pipeline(data)
                data = test_pipeline(data)
                data['img'] = data['img'][None]
                # data = to_tensor_trans(data)
                # data = norm_trans(data)
                # data['img'] = torch.unsqueeze(data['img'],0)
                # Don't use this for now.
                # data = collate([data], samples_per_gpu=1)
                # if next(model.parameters()).is_cuda:
                #     # scatter to specified GPU
                #     data = scatter(data, [device])[0]
                # else:
                #     # just get the actual data from DataContainer
                #     data['img_metas'] = data['img_metas'].data[0]

                # data['img_metas'] = [data['ann_info']]
                # forward the model
                with torch.no_grad():
                    model_out = model(return_loss=False, img=data['img'], img_metas=[data['img_metas'].data])

                all_preds = model_out['preds']
                heatmap = model_out['output_heatmap']
                # pose_results_x = (all_preds[0,:,0].copy()-ims.shape[2]/2+0.5)*1.25+ims.shape[2]/2-0.5
                # pose_results_y = (all_preds[0,:,1].copy()-ims.shape[1]/2+0.5)*1.25+ims.shape[1]/2-0.5
                # pose_results_cur = np.array([pose_results_x,pose_results_y])*conf.rescale
                # pose_results[b,:,:] = pose_results_cur.T
                pose_results[b,:,:] = all_preds[0,:,:2]
                conf_res[b,:] = all_preds[0,:,2].copy()
                if retrawpred:
                    all_hmaps.append(heatmap)

            ret_dict = {'locs':pose_results,'conf':conf_res}
            if retrawpred:
                ret_dict['hmap']=all_hmaps
            return ret_dict

        def close_fn():
            torch.cuda.empty_cache()
            pool.close()

        if conf.batch_size>4:
            return pred_fn, close_fn, model_file
        else:
            return pred_fn_single, close_fn, model_file
