import pathlib
import os
import sys
# from mmcv import Config

import mmcv
import torch
from packaging import version

import mmdet
assert version.parse(mmdet.__version__).major == 3

from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.config import Config
from mmpose.datasets.datasets import BaseCocoStyleDataset as TopDownCocoDataset
from mmpose.apis import init_model
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate
from mmdet.models.task_modules.assigners import MaxIoUAssigner, HungarianAssigner, AssignResult

from mmengine.dataset import Compose, pseudo_collate
from mmdet.apis import inference
from mmcv.ops import RoIPool
from mmdet.models import DETRHead
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.models.task_modules import BBOX_ASSIGNERS
from mmdet.registry import MODELS, TASK_UTILS
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmdet.utils import collect_env
from mmengine.hooks import DistSamplerSeedHook, Hook
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmengine.runner import load_checkpoint
from mmdet.models.utils import multi_apply
from mmdet.apis import DetInferencer

# from mmcv import DictAction
# from mmcv.runner import init_dist
# from mmengine.runner import train_detector
# from mmdet.datasets import build_dataset
# from mmdet.models import build_detector
# from mmdet.utils import get_root_logger
# from mmengine.hooks import EpochBasedRunner, OptimizerHook, IterBasedRunner,Fp16OptimizerHook,build_optimizer,build_runner
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
#
# from mmengine.hooks import DistEvalHook, EvalHook, build_assigner, build_sampler, multi_apply,reduce_mean
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)
# from mmengine.dataset import collate, scatter
# from mmcv.runner import force_fp32


from scipy.optimize import linear_sum_assignment
import download_pretrained as down_pre

import urllib.request as urllib
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
import logging

@BBOX_ASSIGNERS.register_module()
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
#        assert gt_bboxes_ignore is None, \
#            'Only case when gt_bboxes_ignore is None is supported.'
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
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module(force=True)
class APTHungarianAssignerMask(HungarianAssigner):

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

        # originally ..
        # assign all indices to backgrounds first
        # assigned_gt_inds[:] = 0

        # for masking, first ignore all the prediction bboxes by assigning them to -1
        assigned_gt_inds[:] = -1

        overlap_tr = 0.2
        # Find the un-matched pred bboxes that overlap with the gt_bboxes.
        overlaps = bbox_overlaps(gt_bboxes, bboxes.detach())
        overlaps, _ = overlaps.max(dim=0)
        overlaps[matched_row_inds] = 0. # Ignore the matched bboxes.
        masked_negs = overlaps>overlap_tr

        # Add bboxes that completely contain gt_bboxes and are 2x the size of gt_bboxes as neg.
        overlaps_f = bbox_overlaps(gt_bboxes, bboxes, mode='iof')
        overlaps_f, _ = overlaps_f.max(dim=0)
        big_boxes = (overlaps_f>0.95) & (overlaps<0.5)
        # Big boxes should have boxes that contain most of the gt_bboxes but are at least 1/0.5 = 2x in size.
        masked_negs = masked_negs | big_boxes

        # Find predictions that overlap with ignored gt_bboxes
        if gt_bboxes_ignore.shape[0]>0:
            ignore_overlaps, _ = bbox_overlaps(bboxes.detach(), gt_bboxes_ignore, mode='iof').max(dim=1)
            ignore_overlaps[matched_row_inds] = 0.
            masked_negs = masked_negs | (ignore_overlaps>overlap_tr)
        min_num_negs = 5
        num_negs = np.count_nonzero(masked_negs.cpu().numpy())

        if False:
            if num_negs < min_num_negs:
                masked_negs[np.random.choice(range(masked_negs.shape[0]),min_num_negs-num_negs)] = True

        assigned_gt_inds[masked_negs] = 0


        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(
            num_gts, assigned_gt_inds, overlaps, labels=assigned_labels)


@MODELS.register_module(force=True)
class APT_DETRHead(DETRHead):
    def __init__(self,*args,**kwargs):
        k = super(APT_DETRHead,self)
        self.__class__ = DETRHead
        k.__init__(*args,**kwargs)
        self.__class__ = APT_DETRHead

    def _load_from_state_dict(self,*args,**kwargs):
        k = super(APT_DETRHead,self)
        self.__class__ = DETRHead
        k._load_from_state_dict(*args,**kwargs)
        self.__class__ = APT_DETRHead

    # @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        'Updated to accept gt_bboxes_ignore'
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        # assert gt_bboxes_ignore is None, \
        #     'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Updated to accept gt_bboxes_ignore
        """
        # assert gt_bboxes_ignore_list is None, \
        #     'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [
                gt_bboxes_ignore_list for _ in range(num_imgs)
            ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)



@MODELS.register_module(force=True)
class APT_DETRHeadMask(DETRHead):
    def __init__(self,*args,**kwargs):
        k = super(APT_DETRHeadMask,self)
        self.__class__ = DETRHead
        k.__init__(*args,**kwargs)
        self.__class__ = APT_DETRHeadMask

    def _load_from_state_dict(self,*args,**kwargs):
        k = super(APT_DETRHeadMask,self)
        self.__class__ = DETRHead
        k._load_from_state_dict(*args,**kwargs)
        self.__class__ = APT_DETRHeadMask

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        # if assign_result.max_overlaps is not None:
        #     lpos_inds = pos_inds[assign_result.max_overlaps[pos_inds]>0.5]
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_zeros(num_bboxes)
        label_weights[pos_inds] = 1.
        label_weights[neg_inds] = 1.

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    # @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """ Updated so that it accepts gt_bboxes_ignore"""
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        # assert gt_bboxes_ignore is None, \
        #     'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Updated to accept gt_bboxes_ignore
        """
        # assert gt_bboxes_ignore_list is None, \
        #     'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [
                gt_bboxes_ignore_list for _ in range(num_imgs)
            ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)


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



def create_mmdetect_cfg(conf,mmdet_config_file,run_name):
    # curdir = pathlib.Path(__file__).parent.absolute()
    # mmdir = os.path.join(curdir,'mmdetection')
    # mmdet_config_file_path = os.path.join(mmdir,mmdetection_config_file)
    mmdet_init_file_path = mmdet.__file__
    mmdet_dir = os.path.dirname(mmdet_init_file_path)
    dot_mim_folder_path = os.path.join(mmdet_dir, '.mim')  # this feels not-robust
    mmdet_config_file_path = os.path.join(dot_mim_folder_path, mmdet_config_file)
    data_bdir = conf.cachedir

    cfg = Config.fromfile(mmdet_config_file_path)
    cfg.default_hooks.checkpoint.interval = conf.save_step
    cfg.default_hooks.checkpoint.by_epoch = False
    cfg.default_hooks.checkpoint.max_keep_ckpts = conf.maxckpt
    cfg.default_hooks.checkpoint.filename_tmpl = run_name + '-{}'


    assert len(cfg.param_scheduler) == 1, 'Works only for single multisteplr for now'
    assert cfg.param_scheduler[0].type == 'MultiStepLR', 'Works only for multisteplr for now'
    if cfg.param_scheduler[0].type == 'MultiStepLR':
        def_epochs = cfg.max_epochs
        def_steps = cfg.param_scheduler[0].milestones
        cfg.param_scheduler[0].milestones = [int(dd/def_epochs*conf.dl_steps) for dd in def_steps]
        cfg.max_epochs = conf.dl_steps
        cfg.param_scheduler[0].end = conf.dl_steps
        cfg.param_scheduler[0].by_epoch = False

    # cfg.gpu_ids = [0]
    cfg.seed = None
    cfg.work_dir = conf.cachedir

    cfg.dataset_type = 'COCODataset'
    # if conf.get('coco_classes',None) is not None:
    #     cfg.classes = conf.get('coco_classes',('fly','neg_box'))
    # else:
    #     cfg.classes = ('fly','neg_box')

    im_sz = tuple(int(c / conf.rescale) for c in conf.imsz[::-1])  # conf.imsz[0]
    default_samples_per_gpu = cfg.train_dataloader.batch_size
    cfg.train_dataloader.batch_size = conf.batch_size
    # cfg.train_pipeline[2].img_scale = im_sz
    # cfg.test_pipeline[1].img_scale = im_sz
    # cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    # cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline
    # cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline

    tr_file = os.path.join(data_bdir, f'{conf.trainfilename}.json')
    A = PoseTools.json_load(tr_file)
    cls = tuple([aa['name'] for aa in A['categories']])
    for ttype in ['train', 'val', 'test']:
        name = ttype if ttype != 'test' else 'val'
        fname = conf.trainfilename if ttype == 'train' else conf.valfilename
        file = os.path.join(data_bdir, f'{fname}.json')
        if not os.path.exists(file):
            file = tr_file
        cfg[ttype + '_dataloader'].dataset.ann_file = file

        cfg[ttype+ '_dataloader'].dataset.data_prefix = dict(img='')
        cfg[ttype+ '_dataloader'].dataset.data_root = ''
        cfg[ttype+ '_dataloader'].dataset.metainfo = dict(classes=cls,pallete=(0,0,0))
        # Can't clear the test_dataloader b/c for tracking the DetInferencer complains if there's no test_dataloader.  -- AL, 2025-05-02
        if ttype in ['val']:
            # cfg[ttype+ '_evaluator'].ann_file = file
            cfg[ttype+ '_evaluator'] = None
            cfg[ttype+'_cfg'] = None
            cfg[ttype+ '_dataloader'] = None

    if conf.mmdetect_net == 'frcnn': # TODO!!
        for ttype in ['train', 'val', 'test']:
            name = ttype if ttype != 'test' else 'val'
            fname = conf.trainfilename if ttype == 'train' else conf.valfilename
            cfg[ttype+ '_dataloader'].ann_file = os.path.join(data_bdir, f'{fname}.json')
            file = os.path.join(data_bdir, f'{fname}.json')
            cfg[ttype+ '_dataloader'].img_prefix = os.path.join(data_bdir, name)
            cfg[ttype+ '_dataloader'].classes = cfg.classes
            if not conf.get('mmdetect_use_default_sz', True):
                if ttype == 'train':
                    cfg.data[ttype].pipeline[2].img_scale = im_sz
                else:
                    cfg.data[ttype].pipeline[1].img_scale = im_sz

        cfg.model.roi_head.bbox_head.num_classes = 1
        if conf.get('mmdetection_oversample',False):
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
        cfg.model.test_cfg.rcnn.max_per_img = conf.max_n_animals
        cfg.optimizer.lr = cfg.optimizer.lr * conf.learning_rate_multiplier * conf.batch_size/default_samples_per_gpu/8
        cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

    elif conf.mmdetect_net == 'detr':

        cfg.model.bbox_head.num_classes = 1
        # assert (cfg.train_pipeline[3].type == 'AutoAugment'), 'Unsupported train pipeline'
        # assert (cfg.test_pipeline[1].type == 'MultiScaleFlipAug'), 'Unsupported test pipeline'

        assert conf.get('mmdetect_use_default_sz', True), 'Not supported. IN current version very hard to overrride the default size'

        if conf.multi_loss_mask:
            cfg.model.train_cfg.assigner.type = 'APTHungarianAssignerMask'
            cfg.model.bbox_head.type = 'APT_DETRHeadMask'
#            cfg.model.bbox_head.num_query = 300
#         else: should not be required anymore
#             cfg.model.train_cfg.assigner.type = 'APTHungarianAssigner'
#             cfg.model.bbox_head.type = 'APT_DETRHead'

        script_dir = os.path.dirname(os.path.realpath(__file__))
        wt_dir = os.path.join(script_dir,'pretrained')
        url =  'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
        # wt_file = os.path.join(wt_dir,'detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth')
        # if not os.path.exists(wt_file):
        #     if not os.path.exists(wt_dir):
        #         os.makedirs(wt_dir)
        #     urllib.urlretrieve(url,wt_file)

        cfg.load_from = url

    # Make sure the config honors the dl_steps (max iters) set in the conf.
    # Also make sure we're using iteration-based training, instead of epoch-based.
    del cfg.train_cfg['max_epochs']
    cfg.train_cfg.type = 'IterBasedTrainLoop'
    cfg.train_cfg.max_iters = conf.dl_steps
    cfg.train_cfg.val_interval = conf.dl_steps+1
    # cfg.train_cfg.by_epoch = False

    # cfg.train_pipeline[-1]['keys'].append('gt_bboxes_ignore')
    # cfg.data.train.pipeline[-1]['keys'].append('gt_bboxes_ignore')

    return cfg

class TraindataHook(Hook):
    def __init__(self,out_file,conf,interval=50):
        self.interval = interval
        self.out_file = out_file
        self.conf = conf
        self.td_data = {'train_loss':[],'train_dist':[],'step':[],'val_loss':[],'val_dist':[]}

    def after_train_iter(self, runner,batch_idx=0,data_batch=None,outputs=dict(loss=0)):
        if not self.every_n_train_iters(runner,self.interval):
            return
        self.td_data['step'].append(batch_idx + 1)
        self.td_data['train_loss'].append(outputs['loss'].detach().cpu().numpy().copy())
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
        logging.info(f'Step:{runner.iter+1}, Train loss:{self.td_data["train_loss"][-1]}')


class Pose_detect_mmdetect(PoseCommon_pytorch):

    def __init__(self,conf,name,**kwargs):
        super().__init__(conf,name)
        self.conf = conf
        self.name = name
        mmdetect_net = conf.get('mmdetect_net','detr')
        if mmdetect_net == 'frcnn':
            self.cfg_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
        elif mmdetect_net == 'detr':
            self.cfg_file = 'configs/detr/detr_r50_8xb2-150e_coco.py'
        # elif mmdetect_net == 'test':
        #     self.cfg_file = 'configs/APT/roian.py'

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


    def train_wrapper(self,restore=False, model_file=None, debug=False):

        # From mmdetection/tools/train.py

        # Reduce the number of repeated compilations and improve
        # training speed.
        # setup_cache_size_limit_of_dynamo()

        # load config
        cfg = self.cfg
        cfg.launcher = 'none'

        # enable automatic-mixed-precision training
        # if args.amp is True:
        #     cfg.optim_wrapper.type = 'AmpOptimWrapper'
        #     cfg.optim_wrapper.loss_scale = 'dynamic'

        # enable automatically scaling LR
        # if args.auto_scale_lr:
        #     if 'auto_scale_lr' in cfg and \
        #             'enable' in cfg.auto_scale_lr and \
        #             'base_batch_size' in cfg.auto_scale_lr:
        #         cfg.auto_scale_lr.enable = True
        #     else:
        #         raise RuntimeError('Can not find "auto_scale_lr" or '
        #                            '"auto_scale_lr.enable" or '
        #                            '"auto_scale_lr.base_batch_size" in your'
        #                            ' configuration file.')

        # resume is determined in this priority: resume from > auto_resume
        # if args.resume == 'auto':
        #     cfg.resume = True
        #     cfg.load_from = None
        # elif args.resume is not None:
        #     cfg.resume = True
        #     cfg.load_from = args.resume
        if model_file is not None:
            cfg.load_from = model_file
            cfg.resume = False
        elif restore:
            cfg.resume = True
            # cfg.resume_from = self.get_latest_model_file()
            cfg.load_from = None

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        td_hook = TraindataHook(self.get_td_file(),self.conf,self.conf.display_step)
        runner.register_hook(td_hook)

        # start training
        runner.train()


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
        # from mmdetection.demo.image_demo.py
        cfg = self.cfg
        conf = self.conf

        assert conf.is_multi, 'Object detection has to be multi-animal'
        detr_nms = 0.7
        detr_thr = 0.2

        model_file = self.get_latest_model_file() if model_file is None else model_file
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # init_args = {'weights':model_file,'device':device}
        inferencer = DetInferencer(weights=model_file, device=device)
        max_n = conf.max_n_animals
        min_n = conf.min_n_animals

        def pred_fn(ims,retrawpred=False,show=False):

            batch_size = ims.shape[0]
            pose_results = np.ones([batch_size,conf.max_n_animals,2,2])*np.nan
            conf_res = np.zeros([batch_size,conf.max_n_animals,2])

            if ims.shape[-1] ==1:
                ims = np.tile(ims,[1,1,1,3])

            # This code was not working with mmdetect 3.3 or 3.2.  Replaced it with code below. 
            #   -- ALT, 2025-05-03
            # results = inferencer(ims, batch_size=batch_size, draw_pred=False)
            # for b,res in enumerate(results):
            #     if conf.mmdetect_net == 'detr':
            #         cur_max = max_n
            #         cur_res = np.ones([max_n, 5]) * np.nan
            #         bbs = torch.tensor(res[0][:,:4])
            #         overlaps = bbox_overlaps(bbs,bbs)
            #         overlaps = overlaps.numpy()
            #         overlaps[np.diag_indices(overlaps.shape[0])] = 0.
            #         done_count = 0
            #         cur_ix = -1
            #         while done_count<max_n:
            #             cur_ix = cur_ix + 1
            #             if cur_ix>=overlaps.shape[0]: break
            #             if (cur_ix> 0) and (any(overlaps[cur_ix,:cur_ix]>detr_nms)): continue
            #             if (res[0][cur_ix,4]<detr_thr) and done_count>=min_n: continue
            #             cur_res[done_count,:] = res[0][cur_ix]
            #             done_count += 1
            #     else:
            #         cur_res = res[0]
            #         cur_max = len(res[0])
            #     pose_results[b,:cur_max,:,:] = cur_res[:,:4].copy().reshape([-1,2,2])
            #     conf_res[b,:cur_max,:] = cur_res[:cur_max,4:].copy()

            ims_as_list = [ims[i] for i in range(batch_size)]
            results = inferencer(ims_as_list, batch_size=batch_size, draw_pred=False)
            predictions = results['predictions']
            for b,prediction in enumerate(predictions):
                bboxes = prediction['bboxes']  # list of lists of length 4
                scores = prediction['scores']  # list of scalars
                if conf.mmdetect_net == 'detr':
                    cur_max = max_n
                    cur_bboxes = np.ones([max_n, 4]) * np.nan
                    cur_scores = np.ones([max_n]) * np.nan
                    bbs = torch.tensor(bboxes)
                    overlaps = bbox_overlaps(bbs,bbs)
                    overlaps = overlaps.numpy()
                    overlaps[np.diag_indices(overlaps.shape[0])] = 0.
                    done_count = 0
                    cur_ix = -1
                    while done_count<max_n:
                        cur_ix = cur_ix + 1
                        if cur_ix>=overlaps.shape[0]: break
                        if (cur_ix> 0) and (any(overlaps[cur_ix,:cur_ix]>detr_nms)): continue
                        if (scores[cur_ix]<detr_thr) and done_count>=min_n: continue
                        cur_bboxes[done_count,:] = bboxes[cur_ix]
                        cur_scores[done_count] = scores[cur_ix]
                        done_count += 1
                else:
                    cur_max = len(scores)
                    cur_bboxes = np.array(bboxes)
                    cur_scores = np.array(scores)
                pose_results[b,:cur_max,:,:] = cur_bboxes.copy().reshape([-1,2,2])
                conf_res[b,:cur_max,:] = cur_scores.copy()

            ret_dict = {'locs':pose_results,'conf':conf_res}
            return ret_dict

        def close_fn():
            torch.cuda.empty_cache()

        return pred_fn, close_fn, model_file
