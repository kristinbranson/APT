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

from mmpose.datasets.builder import DATASETS
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

            ims, _ = PoseTools.preprocess_ims(in_ims.copy(),np.zeros([in_ims.shape[0],conf.n_classes,2]),conf,False,conf.rescale)
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

