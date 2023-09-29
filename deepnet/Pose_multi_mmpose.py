from Pose_mmpose import Pose_mmpose
from mmpose.models import build_posenet
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, IterBasedRunner, Hook, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmpose.core import wrap_fp16_model
from mmpose.datasets.pipelines import Compose
import numpy as np
from mmcv.parallel import collate, scatter
import torch
import PoseTools
from mmpose.datasets.pipelines.shared_transform import ToTensor, NormalizeTensor
import logging
import pickle
import os
import xtcocotools
from xtcocotools.coco import COCO

## Bottomup dataset

from mmpose.datasets.builder import DATASETS,PIPELINES
from mmpose.datasets.datasets.bottom_up.bottom_up_coco import BottomUpCocoDataset

@DATASETS.register_module()
class BottomUpAPTDataset(BottomUpCocoDataset):
    def __init__(self,**kwargs):
        super(BottomUpCocoDataset,self).__init__(**kwargs)

        # From BottomUpCocoDataset init. It fails.

        # joint index starts from 1
        self.ann_info['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13],
                                     [12, 13], [6, 12], [7, 13], [6, 7],
                                     [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                                     [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                                     [5, 7]]

        self.coco = COCO(kwargs['ann_file'])
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
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

        print(f'=> num_images: {self.num_images}')

        # End BottomUpCocoDataset init

        # APT overrides
        import poseConfig
        conf = poseConfig.conf
        flip_idx = list(range(conf.n_classes))
        for kk in conf.flipLandmarkMatches.keys():
            flip_idx[int(kk)] = conf.flipLandmarkMatches[kk]
        self.ann_info['flip_index'] = flip_idx
        self.ann_info['joint_weights'] = np.ones([conf.n_classes])
        self.sigmas = np.ones([conf.n_classes])*0.6/10.0
        self.ann_info['joint_weights'] = np.ones([self.ann_info['num_joints'],1])
        self.conf = conf


    def _get_mask(self, anno, idx):
        # Masks are created during image generation.
        conf = self.conf
        coco = self.coco
        img_info = coco.loadImgs(self.img_ids[idx])[0]
        m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
        if not conf.multi_loss_mask:
            return m<0.5

        for obj in anno:
            if 'segmentation' in obj:
                rles = xtcocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'],
                    img_info['width'])
                for rle in rles:
                    m += xtcocotools.mask.decode(rle)
                # if obj['iscrowd']:
                #     rle = xtcocotools.mask.frPyObjects(obj['segmentation'],
                #                                        img_info['height'],
                #                                        img_info['width'])
                #     m += xtcocotools.mask.decode(rle)
                # else:

        return m > 0.5

# TODO: Fixing a bug where mmpose code uses np.int instead of np.int32. Remove this when updating mmpose
@PIPELINES.register_module(force=True)
class BottomUpRandomFlip:
    """Data augmentation with random image flip for bottom-up.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        image, mask, joints = results['img'], results['mask'], results[
            'joints']
        self.flip_index = results['ann_info']['flip_index']
        self.output_size = results['ann_info']['heatmap_size']

        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if np.random.random() < self.flip_prob:
            image = image[:, ::-1].copy() - np.zeros_like(image)
            for i, _output_size in enumerate(self.output_size):
                if not isinstance(_output_size, np.ndarray):
                    _output_size = np.array(_output_size)
                if _output_size.size > 1:
                    assert len(_output_size) == 2
                else:
                    _output_size = np.array([_output_size, _output_size],
                                            dtype=np.int32)
                mask[i] = mask[i][:, ::-1].copy()
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size[0] - joints[i][:, :, 0] - 1
                if i == 0 and 'bboxes' in results:
                    bbox = results['bboxes']
                    bbox = bbox[:, [1, 0, 3, 2]]
                    bbox[:, :, 0] = _output_size[0] - bbox[:, :, 0] - 1
                    results['bboxes'] = bbox
        results['img'], results['mask'], results[
            'joints'] = image, mask, joints
        return results



# TODO: Fixing a bug where mmpose FocalHeatmapLoss class generates NaNs when no animal in GT. Remove this when updating mmpose
@LOSSES.register_module(force=True)
class FocalHeatmapLoss(nn.Module):

    def __init__(self, alpha=2, beta=4):
        super(FocalHeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask=None):
        """Modified focal loss.

        Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
        """
        pos_inds = gt.eq(1).bool()
        neg_inds = gt.lt(1).bool()

        if mask is not None:
            mask_bool = mask.bool()
            pos_inds = pos_inds & mask_bool
            neg_inds = neg_inds & mask_bool

        neg_weights = torch.pow(1 - gt, self.beta)

        # loss = 0

        pos_loss = torch.where(pos_inds, torch.log(pred    ) * torch.pow(1 - pred, self.alpha)              , 0.0)
        neg_loss = torch.where(neg_inds, torch.log(1 - pred) * torch.pow(pred    , self.alpha) * neg_weights, 0.0)

        num_pos = pos_inds.float().sum()
        pos_loss_sum = pos_loss.sum()
        neg_loss_sum = neg_loss.sum()

        if num_pos == 0:
            loss = 0.0 - neg_loss_sum
        else:
            loss = 0.0 - (pos_loss_sum + neg_loss_sum) / num_pos
        return loss



# TODO: Fixing a bug where mmpose CIDHead::forward_train() would throw div-by-zero if zero animals in GT. Remove this when updating mmpose
@HEADS.register_module(force=True)
class CIDHead(nn.Module):
    """CID head. paper ref: Dongkai Wang et al. "Contextual Instance Decouple
    for Robust Multi-Person Pose Estimation".

    Args:
        in_channels (int): Number of input channels.
        gfd_channels (int): Number of instance feature map channels
        num_joints (int): Number of joints
        multi_hm_loss_factor (float): loss weight for multi-person heatmap
        single_hm_loss_factor (float): loss weight for single person heatmap
        contrastive_loss_factor (float): loss weight for contrastive loss
        max_train_instances (int): limit the number of instances
        during training to avoid
        prior_prob (float): focal loss bias initialization
    """

    def __init__(self,
                 in_channels,
                 gfd_channels,
                 num_joints,
                 multi_hm_loss_factor=1.0,
                 single_hm_loss_factor=4.0,
                 contrastive_loss_factor=1.0,
                 max_train_instances=200,
                 prior_prob=0.01):
        super().__init__()
        self.multi_hm_loss_factor = multi_hm_loss_factor
        self.single_hm_loss_factor = single_hm_loss_factor
        self.contrastive_loss_factor = contrastive_loss_factor
        self.max_train_instances = max_train_instances
        self.prior_prob = prior_prob

        # iia module
        self.keypoint_center_conv = nn.Conv2d(in_channels, num_joints + 1, 1,
                                              1, 0)
        # gfd module
        self.conv_down = nn.Conv2d(in_channels, gfd_channels, 1, 1, 0)
        self.c_attn = ChannelAtten(in_channels, gfd_channels)
        self.s_attn = SpatialAtten(in_channels, gfd_channels)
        self.fuse_attn = nn.Conv2d(gfd_channels * 2, gfd_channels, 1, 1, 0)
        self.heatmap_conv = nn.Conv2d(gfd_channels, num_joints, 1, 1, 0)

        # loss
        self.heatmap_loss = build_loss(dict(type='FocalHeatmapLoss'))
        self.contrastive_loss = ContrastiveLoss()

        # initialize
        self.init_weights()

    def forward(self, features, forward_info=None):
        """Forward function."""
        assert isinstance(features, list)
        x0_h, x0_w = features[0].size(2), features[0].size(3)

        features = torch.cat([
            features[0],
            F.interpolate(
                features[1],
                size=(x0_h, x0_w),
                mode='bilinear',
                align_corners=False),
            F.interpolate(
                features[2],
                size=(x0_h, x0_w),
                mode='bilinear',
                align_corners=False),
            F.interpolate(
                features[3],
                size=(x0_h, x0_w),
                mode='bilinear',
                align_corners=False)
        ], 1)

        if self.training:
            return self.forward_train(features, forward_info)
        else:
            return self.forward_test(features, forward_info)

    def forward_train(self, features, labels):
        gt_multi_heatmap, gt_multi_mask, gt_instance_coord,\
            gt_instance_heatmap, gt_instance_mask, gt_instance_valid = labels

        pred_multi_heatmap = _sigmoid(self.keypoint_center_conv(features))

        # multi-person heatmap loss
        multi_heatmap_loss = self.heatmap_loss(pred_multi_heatmap,
                                               gt_multi_heatmap, gt_multi_mask)

        contrastive_loss = 0
        total_instances = 0
        instances = defaultdict(list)
        for i in range(features.size(0)):
            if torch.sum(gt_instance_valid[i]) < 0.5:
                continue
            instance_coord = gt_instance_coord[i][
                gt_instance_valid[i] > 0.5].long()
            instance_heatmap = gt_instance_heatmap[i][
                gt_instance_valid[i] > 0.5]
            instance_mask = gt_instance_mask[i][gt_instance_valid[i] > 0.5]
            instance_imgid = i * torch.ones(
                instance_coord.size(0),
                dtype=torch.long,
                device=features.device)
            instance_param = self._sample_feats(features[i], instance_coord)
            contrastive_loss += self.contrastive_loss(instance_param)
            total_instances += instance_coord.size(0)

            instances['instance_coord'].append(instance_coord)
            instances['instance_imgid'].append(instance_imgid)
            instances['instance_param'].append(instance_param)
            instances['instance_heatmap'].append(instance_heatmap)
            instances['instance_mask'].append(instance_mask)

        if total_instances == 0:
            contrastive_loss = torch.zeros_like(multi_heatmap_loss)
            single_heatmap_loss = torch.zeros_like(multi_heatmap_loss)
        else:
            contrastive_loss = contrastive_loss / total_instances

            for k, v in instances.items():
                instances[k] = torch.cat(v, dim=0)

            # limit max instances in training
            if 0 <= self.max_train_instances < instances['instance_param'].size(0):
                inds = torch.randperm(
                    instances['instance_param'].size(0),
                    device=features.device).long()
                for k, v in instances.items():
                    instances[k] = v[inds[:self.max_train_instances]]

            # single person heatmap loss
            global_features = self.conv_down(features)
            instance_features = global_features[instances['instance_imgid']]
            instance_params = instances['instance_param']
            c_instance_feats = self.c_attn(instance_features, instance_params)
            s_instance_feats = self.s_attn(instance_features, instance_params,
                                           instances['instance_coord'])
            cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats),
                                            dim=1)
            cond_instance_feats = self.fuse_attn(cond_instance_feats)
            cond_instance_feats = F.relu(cond_instance_feats)

            pred_instance_heatmaps = _sigmoid(
                self.heatmap_conv(cond_instance_feats))

            gt_instance_heatmaps = instances['instance_heatmap']
            gt_instance_masks = instances['instance_mask']
            single_heatmap_loss = self.heatmap_loss(pred_instance_heatmaps,
                                                    gt_instance_heatmaps,
                                                    gt_instance_masks)

        losses = dict()
        losses['multi_heatmap_loss'] = multi_heatmap_loss *\
            self.multi_hm_loss_factor
        losses['single_heatmap_loss'] = single_heatmap_loss *\
            self.single_hm_loss_factor
        losses['contrastive_loss'] = contrastive_loss *\
            self.contrastive_loss_factor

        return losses

    def forward_test(self, features, test_cfg):
        flip_test = test_cfg.get('flip_test', False)
        center_pool_kernel = test_cfg.get('center_pool_kernel', 3)
        max_proposals = test_cfg.get('max_num_people', 30)
        keypoint_thre = test_cfg.get('detection_threshold', 0.01)

        # flip back feature map
        if flip_test:
            features[1, :, :, :] = features[1, :, :, :].flip([2])

        instances = {}
        pred_multi_heatmap = _sigmoid(self.keypoint_center_conv(features))
        W = pred_multi_heatmap.size()[-1]

        if flip_test:
            center_heatmap = pred_multi_heatmap[:, -1, :, :].mean(
                dim=0, keepdim=True)
        else:
            center_heatmap = pred_multi_heatmap[:, -1, :, :]

        center_pool = F.avg_pool2d(center_heatmap, center_pool_kernel, 1,
                                   (center_pool_kernel - 1) // 2)
        center_heatmap = (center_heatmap + center_pool) / 2.0
        maxm = self.hierarchical_pool(center_heatmap)
        maxm = torch.eq(maxm, center_heatmap).float()
        center_heatmap = center_heatmap * maxm
        scores = center_heatmap.view(-1)
        scores, pos_ind = scores.topk(max_proposals, dim=0)
        select_ind = (scores > (keypoint_thre)).nonzero()

        if len(select_ind) == 0:
            return [], []

        scores = scores[select_ind].squeeze(1)
        pos_ind = pos_ind[select_ind].squeeze(1)
        x = pos_ind % W
        y = pos_ind // W
        instance_coord = torch.stack((y, x), dim=1)
        instance_param = self._sample_feats(features[0], instance_coord)
        instance_imgid = torch.zeros(
            instance_coord.size(0), dtype=torch.long).to(features.device)
        if flip_test:
            instance_param_flip = self._sample_feats(features[1],
                                                     instance_coord)
            instance_imgid_flip = torch.ones(
                instance_coord.size(0), dtype=torch.long).to(features.device)
            instance_coord = torch.cat((instance_coord, instance_coord), dim=0)
            instance_param = torch.cat((instance_param, instance_param_flip),
                                       dim=0)
            instance_imgid = torch.cat((instance_imgid, instance_imgid_flip),
                                       dim=0)

        instances['instance_coord'] = instance_coord
        instances['instance_imgid'] = instance_imgid
        instances['instance_param'] = instance_param

        global_features = self.conv_down(features)
        instance_features = global_features[instances['instance_imgid']]
        instance_params = instances['instance_param']
        c_instance_feats = self.c_attn(instance_features, instance_params)
        s_instance_feats = self.s_attn(instance_features, instance_params,
                                       instances['instance_coord'])
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats),
                                        dim=1)
        cond_instance_feats = self.fuse_attn(cond_instance_feats)
        cond_instance_feats = F.relu(cond_instance_feats)

        instance_heatmaps = _sigmoid(self.heatmap_conv(cond_instance_feats))

        if flip_test:
            instance_heatmaps, instance_heatmaps_flip = torch.chunk(
                instance_heatmaps, 2, dim=0)
            instance_heatmaps_flip = instance_heatmaps_flip[:, test_cfg[
                'flip_index'], :, :]
            instance_heatmaps = (instance_heatmaps +
                                 instance_heatmaps_flip) / 2.0

        return instance_heatmaps, scores

    def _sample_feats(self, features, pos_ind):
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return feats.permute(1, 0)

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > 300:
            maxm = F.max_pool2d(heatmap, 7, 1, 3)
        elif map_size > 200:
            maxm = F.max_pool2d(heatmap, 5, 1, 2)
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1)
        return maxm

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        # focal loss init
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.keypoint_center_conv.bias, bias_value)
        torch.nn.init.constant_(self.heatmap_conv.bias, bias_value)



class Pose_multi_mmpose(Pose_mmpose):

    def __init__(self, conf, name='deepnet',is_multi=True,**kwargs):
        mmpose_net = conf.mmpose_net
        super().__init__(conf,name,mmpose_net=mmpose_net,is_multi=True, **kwargs)

    def get_pred_fn(self, model_file=None,max_n=None,imsz=None):
        # Pred fn is sufficiently different in top-down and bottom-up to have separate fns for both.

        cfg = self.cfg
        conf = self.conf

        assert conf.is_multi, 'This pred function is only for multi-animal (bottom-up)'

        if max_n is not None:

            cfg.model.test_cfg.max_num_people = max_n
            max_n_animals = max_n
        else:
            max_n_animals = conf.max_n_animals

        model = build_posenet(cfg.model)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        model = build_posenet(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        model_file = self.get_latest_model_file() if model_file is None else model_file
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        _ = load_checkpoint(model, model_file, map_location='cpu')
        logging.info(f'Loaded model from {model_file}')
        model = MMDataParallel(model,device_ids=[0])
        # build part of the pipeline to do the same preprocessing as training
        test_pipeline = cfg.test_pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        device = next(model.parameters()).device

        # for multi animal
        flip_idx = list(range(conf.n_classes))
        for kk in conf.flipLandmarkMatches.keys():
            flip_idx[int(kk)] = conf.flipLandmarkMatches[kk]

        to_tensor_trans = ToTensor()
        nt = cfg.test_pipeline[-2]['transforms'][-1]
        norm_trans = NormalizeTensor(nt['mean'],nt['std'])

        match_dist_factor = conf.multi_match_dist_factor

        def pred_fn(in_ims,retrawpred=False):

            pose_results = np.ones([in_ims.shape[0],max_n_animals,conf.n_classes,2])*np.nan
            conf_res = np.zeros([in_ims.shape[0],max_n_animals,conf.n_classes])

            # ims, _ = PoseTools.preprocess_ims(in_ims.copy(),np.zeros([in_ims.shape[0],conf.n_classes,2]),conf,False,conf.rescale)
            ims = in_ims
            tt = test_pipeline
            for b in range(ims.shape[0]):
                if ims.shape[3] == 1:
                    ii = np.tile(ims[b,...],[1,1,3])
                else:
                    ii = ims[b,...]
                # prepare data
                data = {'img': ii.astype('uint8'),
                    'dataset': 'coco',
                    'ann_info': {
                        'image_size':
                            cfg.data_cfg['image_size'],
                        'num_joints':
                            cfg.data_cfg['num_joints'],
                        'flip_index': flip_idx,
                        'image_file':''
                    }
                }

                # data = to_tensor_trans(data)
                # data = norm_trans(data)
                # data['img_metas'] = data['ann_info']
                # data['img'] = torch.unsqueeze(data['img'], 0)
                # data['img_metas']['aug_data'] = [ data['img']]
                # assert False, 'Test this pipeline to include APT scaling'

                data = test_pipeline(data)
                data = collate([data], samples_per_gpu=1)
                if next(model.parameters()).is_cuda:
                    # scatter to specified GPU
                    data = scatter(data, [device])[0]
                else:
                    # just get the actual data from DataContainer
                    data['img_metas'] = data['img_metas'].data[0]

                # forward the model
                with torch.no_grad():
                    model_out = model(return_loss=False, img=data['img'], img_metas=data['img_metas'],return_heatmap=retrawpred)

                all_preds = model_out['preds']
                scores = model_out['scores']
                heatmap = model_out['output_heatmap']
                # remove duplicates
                n_preds = len(all_preds)
                all_array = np.array(all_preds)
                # First sort of find the animal size to set a threshold

                # Find average animal size for predictions
                pred_sz = np.mean(all_array[0].max(axis=-2)-all_array[0].min(axis=-1))
                nms_dist = pred_sz*match_dist_factor

                # find the match indexes
                dd_inter = np.linalg.norm(all_array[None, ..., :2] - all_array[:, None, ..., :2], axis=-1).mean(-1)
                dd_inter = dd_inter.flatten()
                dd_inter[::n_preds+1] = 10000
                dd_inter = dd_inter.reshape([n_preds,n_preds])
                xx, yy = np.where(dd_inter<nms_dist)
                to_remove = []
                for ndx,curx in enumerate(xx):
                    if curx in to_remove:
                        continue
                    else:
                        to_remove.append(yy[ndx])

                count = 0
                for ndx in np.argsort(scores)[::-1]:
                    pred = all_preds[ndx]
                    if count>= conf.max_n_animals:
                        break
                    if ndx in to_remove:
                        continue
                    pose_results[b,count,:,:] = pred[:,:2].copy()*conf.rescale
                    conf_res[b,count,:] = pred[:,2].copy()
                    count = count+1
                    # pose_results.append({
                    #     'keypoints': pred[:, :3],
                    # })

            ret_dict = {'locs':pose_results,'conf':conf_res}
            if retrawpred:
                ret_dict['hmaps'] = heatmap
            return ret_dict

        def close_fn():
            torch.cuda.empty_cache()

        return pred_fn, close_fn, model_file

