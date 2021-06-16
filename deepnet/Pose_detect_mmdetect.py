import pathlib
import os
import sys
sys.path.append('mmdetection')
from mmcv import Config

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, IterBasedRunner, Hook, load_checkpoint,Fp16OptimizerHook,build_optimizer,build_runner
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.assigners import MaxIoUAssigner, HungarianAssigner, AssignResult
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy

from mmdet.datasets.pipelines import Compose
from mmdet.apis import inference
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool

from scipy.optimize import linear_sum_assignment


import logging
import time
import copy
import numpy as np
import json
import pickle
import glob

from PoseCommon_pytorch import PoseCommon_pytorch
import poseConfig
import PoseTools

@BBOX_ASSIGNERS.register_module(force=True)
class APTHungarianAssigner(HungarianAssigner):
    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        # assigned_gt_inds[:] = 0
        assigned_gt_inds[:] = -1

        overlap_tr = 0.2
        overlaps = bbox_overlaps(gt_bboxes, bboxes)
        overlaps, _ = overlaps.max(dim=0)
        overlaps[matched_row_inds] = 0.
        masked_negs = (overlaps>overlap_tr)
        if gt_bboxes_ignore.shape[0]>0:
            ignore_overlaps, _ = bbox_overlaps(bboxes, gt_bboxes_ignore, mode='iof').max(dim=1)
            ignore_overlaps[matched_row_inds] = 0.
            masked_negs = masked_negs | (ignore_overlaps>overlap_tr)
        assigned_gt_inds[masked_negs] = 0

        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module(force=True)
class APTMaxIoUAssigner(MaxIoUAssigner):
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """UPdated MaxIoUAssigner from mmdetection to work with APT masks.
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            # Add to negatives bboxes that have high overlap with gt_bboxes_ignore which are labeled as sort of neg ROIs in APT front-end.
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            neg_gt_ignore = ignore_max_overlaps > self.ignore_iof_thr
            neg = neg_gt_ignore & ( (assign_result.gt_inds<0) & (assign_result.max_overlaps<self.neg_iou_thr[0]))
            # The last condition is so that bboxes that have high overlap with gt_bboxes (>neg_iou_thr[1]) that should be ignored should not get added as negative again.
            assign_result.gt_inds[neg] = 0

        # Add bboxes that completely contain gt_bboxes as neg.
        overlaps_f = self.iou_calculator(gt_bboxes, bboxes, mode='iof')
        overlaps_f, _ = overlaps_f.max(dim=0)
        big_boxes = (overlaps_f>0.95) & (assign_result.max_overlaps<0.5)
        # Big boxes should have boxes that contain most of the gt_bboxes but are at least 1/0.5 = 2x in size.
        assign_result.gt_inds[big_boxes] = 0

        if False: # For debugging
            plt.plot(bboxes[assign_result.gt_inds == 0, :][:, [0, 2]].cpu().numpy().T, bboxes[assign_result.gt_inds == 0, :][:, [1, 3]].cpu().numpy().T, color='r')
            plt.plot(gt_bboxes[:, [0, 2]].cpu().numpy().T, gt_bboxes[:, [1, 3]].cpu().numpy().T, color='k', linewidth=3)
            if gt_bboxes_ignore is not None:
                plt.plot(gt_bboxes_ignore[:, [0, 2]].cpu().numpy().T, gt_bboxes_ignore[:, [1, 3]].cpu().numpy().T, color='c', linewidth=3)


            plt.plot(bboxes[assign_result.gt_inds > 0, :][:, [0, 2]].cpu().numpy().T,
         bboxes[assign_result.gt_inds > 0, :][:, [1, 3]].cpu().numpy().T, color='g')
#            plt.plot(bboxes[assign_result.gt_inds < 0, :][:, [0, 2]].cpu().numpy().T, bboxes[assign_result.gt_inds < 0, :][:, [1, 3]].cpu().numpy().T, color='b', linewidth=0.2)
            plt.show()
            # sctr(bboxes[assign_result.gt_inds < 0, :].cpu().numpy())
            # sctr(bboxes[assign_result.gt_inds == 0, :].cpu().numpy())
            # sctr(bboxes[assign_result.gt_inds > 0, :].cpu().numpy())

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result



def create_mmdetect_cfg(conf,mmdetection_config_file,run_name):
    curdir = pathlib.Path(__file__).parent.absolute()
    mmdir = os.path.join(curdir,'mmdetection')
    mmdetect_config = os.path.join(mmdir,mmdetection_config_file)
    data_bdir = conf.cachedir

    cfg = Config.fromfile(mmdetect_config)
    cfg.checkpoint_config.interval = conf.save_step
    cfg.checkpoint_config.filename_tmpl = run_name + '-{}'
    cfg.checkpoint_config.by_epoch = False
    cfg.checkpoint_config.max_keep_ckpts = conf.maxckpt


    assert cfg.lr_config.policy == 'step', 'Works only for steplr for now'
    if cfg.lr_config.policy == 'step':
        def_epochs = cfg.runner.max_epochs
        def_steps = cfg.lr_config.step
        cfg.lr_config.step = [int(dd/def_epochs*conf.dl_steps) for dd in def_steps]

    cfg.gpu_ids = range(1)
    cfg.seed = None
    cfg.work_dir = conf.cachedir

    cfg.dataset_type = 'COCODataset'
    cfg.classes = ('fly','neg_box')

    im_sz = tuple(int(c / conf.rescale) for c in conf.imsz[::-1])  # conf.imsz[0]

    if conf.mmdetect_net == 'frcnn':
        for ttype in ['train', 'val', 'test']:
            name = ttype if ttype is not 'test' else 'val'
            fname = conf.trainfilename if ttype == 'train' else conf.valfilename
            cfg.data[ttype].ann_file = os.path.join(data_bdir, f'{fname}.json')
            file = os.path.join(data_bdir, f'{fname}.json')
            cfg.data[ttype].img_prefix = os.path.join(data_bdir, name)
            cfg.data[ttype].classes = cfg.classes
            if ttype == 'train':
                cfg.data[ttype].pipeline[2].img_scale = im_sz
            else:
                cfg.data[ttype].pipeline[1].img_scale = im_sz

        cfg.model.roi_head.bbox_head.num_classes = 1
        if conf.get('mmdetection_oversample',True):
            cfg.model.test_cfg.rpn.nms.iou_threshold = 0.9
            cfg.model.test_cfg.rcnn.nms.iou_threshold = 0.7
            cfg.model.test_cfg.rcnn.score_thr = 0.01

        if conf.multi_loss_mask:
            cfg.model.train_cfg.rpn.assigner.neg_iou_thr = (0.15, 0.25)
            # Roughly boxes that have 0.5 to 0.25 overlap will be marked as negative.
            cfg.model.train_cfg.rpn.assigner.type = 'APTMaxIoUAssigner'
            cfg.model.train_cfg.rpn.assigner.ignore_iof_thr = 0.85

            cfg.model.train_cfg.rcnn.assigner.neg_iou_thr = (0.15, 0.33)
#            cfg.model.train_cfg.rcnn.assigner.pos_iou_thr = 0.75
            cfg.model.train_cfg.rcnn.assigner.type = 'APTMaxIoUAssigner'
            cfg.model.train_cfg.rcnn.assigner.ignore_iof_thr = 0.85
            cfg.model.train_cfg.rpn_proposal.max_per_img *= 10
        assert (cfg.train_pipeline[2].type == 'Resize'), 'Unsupported train pipeline'
        cfg.train_pipeline[2].img_scale = im_sz
        cfg.model.test_cfg.rcnn.max_per_img = conf.max_n_animals

    elif conf.mmdetect_net == 'detr':
        for ttype in ['train', 'val', 'test']:
            name = ttype if ttype is not 'test' else 'val'
            fname = conf.trainfilename if ttype == 'train' else conf.valfilename
            cfg.data[ttype].ann_file = os.path.join(data_bdir, f'{fname}.json')
            file = os.path.join(data_bdir, f'{fname}.json')
            cfg.data[ttype].img_prefix = os.path.join(data_bdir, name)
            cfg.data[ttype].classes = cfg.classes
            if ttype == 'train':
                cfg.data[ttype].pipeline[3] = dict(type='Resize', img_scale=im_sz, keep_ratio=True)
            else:
                cfg.data[ttype].pipeline[1].img_scale = im_sz


        cfg.model.bbox_head.num_classes = 1
        assert (cfg.train_pipeline[3].type == 'AutoAugment'), 'Unsupported train pipeline'
        cfg.train_pipeline[3] = dict(type='Resize', img_scale=im_sz, keep_ratio=True)
        if conf.multi_loss_mask:
            cfg.model.train_cfg.assigner.type = 'APTHungarianAssigner'


    assert (cfg.test_pipeline[1].type == 'MultiScaleFlipAug'), 'Unsupported test pipeline'
    cfg.test_pipeline[1].img_scale = im_sz

    cfg.train_pipeline[-1]['keys'].append('gt_bboxes_ignore')
    cfg.data.train.pipeline[-1]['keys'].append('gt_bboxes_ignore')

    default_samples_per_gpu = cfg.data.samples_per_gpu
    cfg.data.samples_per_gpu = conf.batch_size
    cfg.optimizer.lr = cfg.optimizer.lr * conf.learning_rate_multiplier * conf.batch_size/default_samples_per_gpu/8

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
            self.td_data['train_loss'].append(runner.log_buffer.output['loss'].copy())
        else:
            self.td_data['train_loss'].append(runner.log_buffer.output['all_loss'].copy())
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


class Pose_detect_mmdetect(PoseCommon_pytorch):

    def __init__(self,conf,name,**kwargs):
        super().__init__(conf,name)
        self.conf = conf
        self.name = name
        mmdetect_net = conf.mmdetect_net
        if mmdetect_net == 'frcnn':
            self.cfg_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
        elif mmdetect_net == 'detr':
            self.cfg_file = 'configs/detr/detr_r50_8x2_150e_coco.py'
        elif mmdetect_net == 'test':
            self.cfg_file = 'configs/APT/roian.py'

        else:
            assert False, 'Unknown mmpose net type'

        poseConfig.conf = conf
        self.cfg = create_mmdetect_cfg(self.conf,self.cfg_file,name)


    def get_td_file(self):
        if self.name =='deepnet':
            td_name = os.path.join(self.conf.cachedir,'traindata')
        else:
            td_name = os.path.join(self.conf.cachedir, self.conf.expname + '_' + self.name + '_traindata')
        return td_name


    def train_wrapper(self,restore=False, model_file=None):

        # From mmdetection/tools/train.py
        cfg = self.cfg
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        dataset = [build_dataset(cfg.data.train)]

        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            dataset.append(build_dataset(val_dataset))

        if cfg.checkpoint_config is not None:
            # save mmpose version, config file content
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmpose_version=__version__ + get_git_hash(digits=7),
                config=cfg.pretty_text,
            )


        # Rest is from mmdetection/apis/train.py:train_detector

        validate = False
        distributed = len(cfg.gpu_ids)>1
        if distributed:
            init_dist('pytorch', **cfg.dist_params)

        meta = None
        logger = logging.getLogger()
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if model_file is not None:
            cfg.load_from = model_file
        elif restore:
            cfg.resume_from = self.get_latest_model_file()

        dataloader_setting = dict(
            samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
        dataloader_setting = dict(dataloader_setting, **cfg.data.get('train_dataloader', {}))

        data_loaders = [ build_dataloader(ds, **dataloader_setting) for ds in dataset]

        # determine wether use adversarial training precess or not
        use_adverserial_train = cfg.get('use_adversarial_train', False)

        # put model on gpus
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', True)
            model = MMDistributedDataParallel(
                model.cuda(),
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)
        if self.conf.get('mmpose_use_epoch_runner', False):
            get_runner = EpochBasedRunner
            steps = self.conf.dl_steps // len(data_loaders[0])
        else:
            get_runner = IterBasedRunner
            steps = self.conf.dl_steps

        runner = get_runner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
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
        runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config, cfg.get('momentum_config', None))
        if distributed:
            runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
        if validate:
            eval_cfg = cfg.get('evaluation', {})
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            dataloader_setting = dict(
                # samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
                samples_per_gpu=1,
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

        td_hook = TraindataHook(self.get_td_file(),self.conf,self.conf.display_step)
        runner.register_hook(td_hook)

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow, steps)


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

        assert conf.is_multi, 'Object detection has to be multi-animal'

        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        model_file = self.get_latest_model_file() if model_file is None else model_file
        model = inference.init_detector(cfg,model_file)
        logging.info(f'Loaded model from {model_file}')

        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)
        device = next(model.parameters()).device

        def pred_fn(ims,retrawpred=False,show=False):

            pose_results = np.ones([ims.shape[0],conf.max_n_animals,2,2])*np.nan
            conf_res = np.zeros([ims.shape[0],conf.max_n_animals,2])

            if ims.shape[-1] ==1:
                ims = np.tile(ims,[1,1,1,3])
            datas = []
            for img in ims:
                # prepare data
                data = dict(img=img)
                # build the data pipeline
                data = test_pipeline(data)
                datas.append(data)

            data = collate(datas, samples_per_gpu=len(ims))
            # just get the actual data from DataContainer
            data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
            data['img'] = [img.data[0] for img in data['img']]
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                for m in model.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'

            # forward the model
            with torch.no_grad():
                results = model(return_loss=False, rescale=True, **data)

            if show:
                from mmdet.core.visualization.image import imshow_det_bboxes
                from matplotlib import pyplot as plt
                plt.ion()
                for ndx in range(ims.shape[0]):
                    # bboxes = np.vstack(results[ndx])
                    # labels = [np.full(bbox.shape[0], i, dtype=np.int32)
                    #     for i, bbox in enumerate(results[ndx])]
                    # labels = np.concatenate(labels)
                    # imshow_det_bboxes(ims[ndx], bboxes, labels,score_thr=0.3)
                    # plt.show()
                    model.CLASSES = ['animal']
                    ii_out = model.show_result(
                    ims[ndx],
                    results[ndx],
                    show=False,
                    score_thr=0.3)
                    plt.imshow(ii_out)
                    plt.show()

            for b,res in enumerate(results):
                pose_results[b,:len(res[0]),:,:] = res[0][:,:4].copy().reshape([-1,2,2])
                conf_res[b,:len(res[0]),:] = res[0][:,4:].copy()

            ret_dict = {'locs':pose_results,'conf':conf_res}
            return ret_dict

        def close_fn():
            torch.cuda.empty_cache()

        return pred_fn, close_fn, model_file