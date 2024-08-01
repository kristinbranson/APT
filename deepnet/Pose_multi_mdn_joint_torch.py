import PoseCommon_pytorch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN
import numpy as np
import PoseTools
from torchvision.ops import misc as misc_nn_ops
import pickle
import os
import cv2
from torchvision import transforms
from torch.nn.parameter import Parameter


class pred_layers(nn.Module):

    def __init__(self, n_in, n_out):
        super(pred_layers,self).__init__()
        momentum=0.1
        self.conv1 = nn.Conv2d(n_in,n_in, 3, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.bn1  = nn.BatchNorm2d(n_in,momentum=momentum)
        # self.bn1  = my_bn(n_in,momentum=momentum)
        self.conv2 = nn.Conv2d(n_in,n_in, 3, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.bn2  = nn.BatchNorm2d(n_in,momentum=momentum)
        self.conv3 = nn.Conv2d(n_in,n_in, 3, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        self.bn3  = nn.BatchNorm2d(n_in,momentum=momentum)
        self.conv_out = nn.Conv2d(n_in,n_out, 1, padding=0)
        torch.nn.init.xavier_normal_(self.conv_out.weight)

    def forward(self, x):
        x1 = self.bn1(self.conv1(F.relu(x)))
        x = self.bn2(self.conv2(F.relu(x1)))
        x = self.bn3(self.conv3(F.relu(x)))
        x = F.relu(x + x1)
        x = self.conv_out(x)
        return x


class my_bn(nn.Module):
    def __init__(self,num_features,**kwargs):
        super(my_bn,self).__init__()
        self.bn = nn.BatchNorm2d(num_features,**kwargs)
        self.rmean = Parameter(torch.Tensor(1,num_features,1,1),requires_grad=False)
        self.rvar = Parameter(torch.Tensor(1,num_features,1,1),requires_grad=False)
        self.rmean.zero_()
        self.rvar.zero_()


    def forward(self,x):
        if self.bn.training:
            with torch.no_grad():
                xm = x.mean(dim=[0, 2, 3], keepdim=True)
                xsq = (x**2).mean(dim=[0,2,3],keepdim=True)
                nb = self.bn.num_batches_tracked
                new_rmean = xm/(nb+1) + self.rmean*nb/(nb+1)
                new_rvar = xsq/(nb+1) + self.rvar*nb/(nb+1)
                self.rmean.copy_(new_rmean)
                self.rvar.copy_(new_rvar)
        cur_var = self.rvar - self.rmean**2
        x = (x-self.rmean)/(torch.sqrt(cur_var)+self.bn.eps)
        with torch.no_grad():
            y = self.bn(x)
        return x



def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # m.track_running_stats = False
        m.eval()


def nanmean(v,*args,**kwargs):
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0.
    return v.sum(*args,**kwargs)/(~is_nan).float().sum(*args,**kwargs)

def nansum(v,*args,**kwargs):
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0.
    return v.sum(*args,**kwargs)

def nanmin(v):
    k = v[~torch.isnan(v)]
    if k.size()[0] == 0:
        return torch.ones(device=v.device)*np.nan
    else:
        return torch.min(k)


def my_resnet_fpn_backbone(backbone_name, pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):
    """
    From torchvision backbone utils.
    Modified to support more channels
    
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Arguments:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # select layers that wont be frozen

    backbone = models.resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = backbone.inplanes//4
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


class mdn_joint(nn.Module):

    def __init__(self, npts, device, pretrain_freeze_bnorm=True, k_j=4, k_r=3, wt_offset=-5,fpn_joint_layer=3,fpn_ref_layer=0,pred_occluded=False):
        super(mdn_joint,self).__init__()

        bn_layer = misc_nn_ops.FrozenBatchNorm2d if pretrain_freeze_bnorm else None
        # Use already available fpn. woohoo.
        backbone = my_resnet_fpn_backbone('resnet50',pretrained=True,trainable_layers=5,norm_layer=bn_layer)
        n_ftrs = backbone.fpn.layer_blocks[0].out_channels

        # else:
        #     backbone = models.resnet50(pretrained=True)
        #     n_ftrs = backbone.layer4[2].conv3.weight.shape[0]
        #
        #     backbone = nn.Sequential(*list(backbone.children())[:-2])
        #     if pretrain_freeze_bnorm:
        #         backbone.apply(freeze_bn)
        #     # self.bn_backone = nn.BatchNorm2d(n_ftrs)
        #     # self.bn_backone_fpn = None

        self.backbone = backbone

        self.locs_joint = pred_layers(n_ftrs, 2*npts*k_j)
        self.wts_joint = pred_layers(n_ftrs, k_j)
        self.locs_ref = pred_layers(n_ftrs,npts*2*k_r)
        self.wts_ref = pred_layers(n_ftrs,npts*k_r)
        self.pred_occluded = pred_occluded
        if pred_occluded:
            self.p_occ = pred_layers(n_ftrs,npts*k_j)
        self.npts= npts
        self.device = device
        self.k_r = k_r
        self.k_j = k_j
        self.im_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).to(self.device)
        self.im_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).to(self.device)
        self.wt_offset = wt_offset
        self.fpn_joint_layer = fpn_joint_layer
        self.fpn_ref_layer = fpn_ref_layer

    def forward(self, input):
        x = input['images']
        x = x.float()
        if not (x.device.type == 'cuda'):
            x = x.to(self.device)
        # Normalize according torchvision models
        im_mean = self.im_mean.to(x.device)
        im_std = self.im_std.to(x.device)
        x = x - im_mean
        x = x / im_std

        x = self.backbone(x)
        x_j = x[f'{self.fpn_joint_layer}']
        x_r = x[f'{self.fpn_ref_layer}']

        locs_j = self.locs_joint(x_j)
        wts_j = self.wts_joint(x_j) + self.wt_offset
        locs_r = self.locs_ref(x_r)
        wts_r = self.wts_ref(x_r)

        js = locs_j.shape
        locs_j = locs_j.reshape(js[0:1] + (self.npts,2,self.k_j) + js[2:])
        y_off_j, x_off_j = torch.meshgrid([torch.arange(js[-2],device=self.device),torch.arange(js[-1],device=self.device)])
        # Add the offsets. NOTE: Torch meshgrid behaves differently than numpy meshgrid!!
        locs_j_x = locs_j[...,0,:,:,:] + x_off_j
        locs_j_y = locs_j[...,1,:,:,:] + y_off_j
        locs_j_off = torch.stack([locs_j_x,locs_j_y],2)

        rs = locs_r.shape
        locs_r = locs_r.reshape(rs[0:1] + (self.npts,2,self.k_r) + rs[2:])
        y_off_r, x_off_r = torch.meshgrid([torch.arange(rs[-2],device=self.device),torch.arange(rs[-1],device=self.device)])
        locs_r_x = locs_r[...,0,:,:,:] + x_off_r
        locs_r_y = locs_r[...,1,:,:,:] + y_off_r
        locs_r_off = torch.stack([locs_r_x,locs_r_y],2)

        wr = wts_r.shape
        wts_r = torch.reshape(wts_r,wr[0:1] + (self.npts,self.k_r) + wr[2:])

        if self.pred_occluded:
            pred_occ = self.p_occ(x_j)
            pred_occ = pred_occ.reshape(js[0:1]+(self.npts,self.k_j)+js[2:])
        else:
            pred_occ = None

        return locs_j_off, wts_j, locs_r_off, wts_r, pred_occ


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

class Pose_multi_mdn_joint_torch(PoseCommon_pytorch.PoseCommon_pytorch):

    def __init__(self,conf,**kwargs):
        super(Pose_multi_mdn_joint_torch, self).__init__(conf, **kwargs)
        self.fpn_joint_layer = self.conf.get('mdn_joint_layer_num',3)
        self.fpn_ref_layer  = self.conf.get('mdn_joint_ref_layer_num',0)

        self.offset = 4*(2**self.fpn_joint_layer)
        self.ref_scale = 4*(2**self.fpn_ref_layer)
        self.locs_noise = self.conf.get('mdn_joint_ref_noise',0.1)
        # self.k_j = 4 if self.fpn_joint_layer ==3 else 1
        self.k_j = 1
        self.k_r = 3
        self.wt_offset = self.conf.get('mdn_joint_wt_offset',-5)
        self.version = 2.
        # version 1 has k_j = 4

    def create_model(self):
        return mdn_joint(self.conf.n_classes, self.device,pretrain_freeze_bnorm=self.conf.pretrain_freeze_bnorm, k_j=self.k_j, k_r=self.k_r, wt_offset=self.wt_offset,fpn_joint_layer=self.fpn_joint_layer,fpn_ref_layer=self.fpn_ref_layer,pred_occluded=self.conf.predict_occluded)

    def loss_slow(self, preds, labels):
        n_classes = self.conf.n_classes
        offset = self.offset
        locs_joint, wts_joint, locs_ref, wts_ref, pred_occ = preds
        labels = labels.to(self.device)

        ll_joint = torch.sigmoid(wts_joint)
        ls = locs_joint.shape
        locs_joint_flat = locs_joint.reshape([locs_joint.shape[0],1,n_classes,2,self.k_j*ls[-2]*ls[-1]]).permute([0,1,4,2,3])
        ll_joint_flat = ll_joint.reshape([ll_joint.shape[0],self.k_j*ls[-1]*ls[-2]])
        valid = torch.all(torch.all(labels>-1000,dim=3),dim=2)

        dd = torch.norm(locs_joint_flat*offset-labels.unsqueeze(2),dim=-1)
        qq = torch.unsqueeze(labels[...,0],2)
        qq = qq.repeat([1,1,ls[-1]*ls[-2]*self.k_j,1])
        dd = torch.where(qq>-1000,dd,1000*torch.ones_like(dd))
        # dd has the distance between all the predictions and all the labels
        dd_all = dd.sum(-1)
        p_assign = torch.softmax(-dd_all.detach()/n_classes,axis=1)

        # loss between predictions and labels
        cur_pred_loss = 0
        cur_wt_loss = 0
        # assign each prediction to the label based on its distance. The compute the weighted loss between the prediction and that label
        ll_detach = ll_joint_flat
        for b in range(locs_joint.shape[0]):
            for lpts in torch.where(valid[b,:])[0]:

                assign = p_assign[b,lpts,:]*ll_detach[b,:]
                assign_norm = assign/(assign.sum())
                dloss = assign_norm * dd_all[b,lpts,:]
                cur_pred_loss = cur_pred_loss + dloss.sum()
                cur_wt_loss = cur_wt_loss + (1.-assign.sum())**2

        joint_loss = cur_pred_loss / self.offset
        wt_loss = cur_wt_loss*n_classes

        # Loss to ensure that the number of predictions match the number of labeled animals. This weight needs to be upweighted otherwise the training converges to degenerate case where  ll_joint is predicted as zero, which would make joint_loss 0.
        # w_assign = p_assign * ll_joint_flat.unsqueeze(1)
        # wt_loss = (w_assign.sum(dim=-1) - valid.type(torch.float32))**2
        # wt_loss = n_classes * wt_loss.sum()

        cur_loss = 0
        locs_noise_mag = self.locs_noise
        locs_noise_mag = torch.tensor(locs_noise_mag).to(self.device)
        if locs_noise_mag > 0.001:
            locs_noise = locs_joint_flat + (torch.rand(locs_joint_flat.shape,device=self.device)-0.5)*2*locs_noise_mag
        else:
            locs_noise = locs_joint
        locs_noise = locs_noise * self.offset/self.ref_scale

        for b in range(ll_joint.shape[0]):
            for g in range(labels.shape[1]):
                if not valid[b,g]:
                    continue
                assign = p_assign[b,g,:]*ll_joint_flat[b,:]
                selex = torch.argmax(assign)
                # torch supports histogramming. Maybe instead of argmax that can be used.

                idx = torch.round(locs_noise[b, 0, selex, :, :]).int()
                # ids are predicted as x,y to match input locs.
                idx_y = torch.clamp(idx[:,1], 0, locs_ref.shape[-2] - 1)
                idx_x = torch.clamp(idx[:,0], 0, locs_ref.shape[-1] - 1)
                for cls in range(labels.shape[2]):
                    if (labels[b,g,cls,0] < -1000) or torch.isnan(labels[b,g,cls,0]):
                        continue

                    pp = labels[b, g, cls:cls + 1, :]
                    cur_ref = locs_ref[b, cls, :, :, idx_y[cls], idx_x[cls]] * self.ref_scale

                    dd_ref = torch.norm(pp-cur_ref.T,dim=-1)
                    ll_ref = torch.softmax(wts_ref[b,cls,:,idx_y[cls],idx_x[cls]],0)
                    cur_loss_ref = torch.sum(dd_ref*ll_ref)
                    cur_loss = cur_loss + cur_loss_ref

        ref_loss = cur_loss

        tot_loss = wt_loss  + joint_loss + ref_loss
        return tot_loss / n_classes


    def loss(self, preds, labels_dict):
        labels = labels_dict['locs'].float()
        occ = labels_dict['occ'].float()
        labels = labels.to(self.device)
        occ = occ.to(self.device)
        n_classes = self.conf.n_classes
        offset = self.offset
        locs_joint, wts_joint, locs_ref, wts_ref, occ_pred = preds
        j_wt_factor = max(0,(self.step[1]*0.5-self.step[0])/(self.step[1]*0.5))
        wts_joint = wts_joint - self.wt_offset*j_wt_factor

        # ll_joint has the weight logits
        ll_joint = torch.sigmoid(wts_joint)

        # Mask the predictions
        if self.conf.multi_loss_mask:
            mask_down = labels_dict['mask'][:,::offset,::offset].to(self.device)
            ll_joint = torch.where(mask_down[:,None,:,:]>0,ll_joint,torch.zeros_like(ll_joint))

        ls = locs_joint.shape
        locs_joint_flat = locs_joint.reshape(
            [locs_joint.shape[0], 1, n_classes, 2, self.k_j * ls[-2] * ls[-1]]).permute([0, 1, 4, 2, 3])
        ll_joint_flat = ll_joint.reshape([ll_joint.shape[0], self.k_j * ls[-1] * ls[-2]])
        valid = torch.any(torch.all(labels > -1000, dim=3), dim=2)
        valid_lbl = labels[...,0] > -1000
        missing = torch.all(~valid,1)

        dd = torch.norm(locs_joint_flat * offset - labels.unsqueeze(2), dim=-1)
        # dd has the distance between all the predictions and all the labels

        qq = torch.unsqueeze(labels[..., 0], 2)
        qq = qq.repeat([1, 1, ls[-1] * ls[-2] * self.k_j, 1])
        dd = torch.where(qq > -1000, dd, torch.zeros_like(dd))
        # Set the distances to 0 where labels are invalid

        dd_all = dd.sum(-1)  # all is all the landmarks
        qq_all = torch.unsqueeze(valid,2)
        qq_all = qq_all.repeat([1,1,ls[-1]*ls[-2]*self.k_j])
        # Set distances to invalid instances to a very high value.
        dd_all = torch.where(qq_all,dd_all,10000*torch.ones_like(dd_all)*self.conf.n_classes)

        p_assign = torch.softmax(-dd_all.detach(), axis=1)

        # assign each prediction to the label based on its distance. Then compute the weighted loss between the prediction and that label
        assign = p_assign * torch.unsqueeze(ll_joint_flat, 1)
        assign_sum = torch.where(valid,assign.sum(axis=-1),torch.ones_like(assign[:,:,0]))
        assign_norm = assign/ torch.unsqueeze(assign_sum+1e-10,dim=-1)
        dloss = (assign_norm*dd_all).sum(axis=-1)
        cur_pred_loss = torch.where(valid,dloss,torch.zeros_like(dloss)).sum(axis=-1)

        # We want only one prediction for each label. We penalize multiple predictions by taking the total weight that gets assigned to a label. This should sum to 1. However this leads to corner case where three close by pixels can have values < 0, but the sum of their sigmoids is 1. To avoid this have that they total weight for each label is 2. With this even for corner cases we will have that one pixels value is greater than 0. Also, this will force close by pixels too to predict the pose.
        wt_loss_all = (2.-assign.sum(axis=-1))**2
        cur_wt_loss = torch.where(valid,wt_loss_all,torch.zeros_like(wt_loss_all)).sum(axis=-1)

        # Predict occluded loss
        if self.conf.predict_occluded:
            occ_flat = occ_pred.reshape([occ_pred.shape[0], 1, n_classes, self.k_j * ls[-2] * ls[-1]]).permute(
                [0, 1, 3, 2])
            dd_occ = (occ_flat - occ[:,:,None]) ** 2
            dd_occ = torch.where(valid[:, :, None, None], dd_occ, torch.zeros_like(dd_occ))
            # Set the loss 0 where labels are missing
            dd_occ = dd_occ.sum(-1)
            docc_loss = (assign_norm * dd_occ).sum(axis=-1)
            cur_occ_loss = torch.where(valid, docc_loss, torch.zeros_like(docc_loss)).sum(axis=-1)
            cur_pred_loss = cur_pred_loss + cur_occ_loss*10
            # occ loss is roughly equal to missing a pose by 10px


        # when an example has no animal, use a different path.
        logit_sum = ll_joint_flat.sum(-1)
        logit_err = logit_sum**2
        cur_wt_loss = torch.where(missing,logit_err,cur_wt_loss)

        loss_wt = 1- cur_wt_loss/(valid.sum(axis=-1)+0.01)
        loss_wt = torch.clamp(loss_wt,0.01,1).detach()

        joint_loss = cur_pred_loss*loss_wt #/ self.offset
        wt_loss = cur_wt_loss * n_classes * 10


        # Loss to ensure that the number of predictions match the number of labeled animals. This weight needs to be upweighted otherwise the training converges to degenerate case where  ll_joint is predicted as zero, which would make joint_loss 0.

        locs_noise_mag = self.locs_noise
        locs_noise_mag = torch.tensor(locs_noise_mag).to(self.device)
        if locs_noise_mag > 0.001:
            locs_noise = locs_joint_flat + (
                        torch.rand(locs_joint_flat.shape, device=self.device) - 0.5) * 2 * locs_noise_mag
        else:
            locs_noise = locs_joint_flat
        locs_noise = locs_noise * self.offset/self.ref_scale

        assign_ndx = torch.argmax(assign, axis=-1)
        bsz = labels.shape[0]
        n_max = labels.shape[1]
        npts = labels.shape[2]
        locs_noise_dim = locs_noise.repeat([1, n_max, 1, 1, 1])
        i1, i2 = torch.meshgrid(torch.arange(0, bsz, device=self.device), torch.arange(0, n_max, device=self.device))
        idx = torch.round(locs_noise_dim[i1, i2, assign_ndx, :, :]).long()
        idx_y = torch.clamp(idx[..., 1], 0, locs_ref.shape[-2] - 1)
        idx_x = torch.clamp(idx[..., 0], 0, locs_ref.shape[-1] - 1)

        locs_ref_dim = torch.unsqueeze(locs_ref, 1).repeat([1, n_max, 1, 1, 1, 1, 1])
        wts_ref_dim = torch.unsqueeze(wts_ref, 1).repeat([1, n_max, 1, 1, 1, 1])
        i1, i2, i3 = torch.meshgrid(torch.arange(bsz, device=self.device), torch.arange(n_max, device=self.device),
                                    torch.arange(npts, device=self.device))
        ref_pred = locs_ref_dim[i1, i2, i3, :, :, idx_y, idx_x]
        ref_pred = ref_pred * self.ref_scale
        ref_wts = wts_ref_dim[i1, i2, i3, :, idx_y, idx_x]
        ref_wts = torch.softmax(ref_wts, 3)
        ref_dist = torch.norm(ref_pred - torch.unsqueeze(labels, -1), dim=-2)
        ref_loss_all = torch.sum(ref_wts * ref_dist, dim=-1)
        ref_loss_sel = torch.where(valid_lbl, ref_loss_all, torch.zeros_like(ref_loss_all))
        ref_loss = ref_loss_sel.sum(axis=(-1,-2))

        # downweight refine loss if the animal detection is not working.
        ref_loss = ref_loss*loss_wt

        tot_loss = wt_loss  + joint_loss + ref_loss
        return tot_loss / n_classes


    def create_targets(self, inputs):
        target_dict = {'locs':inputs['locs']}
        if 'mask' in inputs.keys():
            target_dict['mask'] = inputs['mask']
        target_dict['occ'] = inputs['occ']
        return target_dict

    def get_joint_pred(self,preds):
        n_max = self.conf.max_n_animals
        n_min = self.conf.min_n_animals
        locs_joint, logits_joint, locs_ref, logits_ref, occ_out = preds
        locs_joint = locs_joint
        bsz = locs_joint.shape[0]
        n_classes = locs_joint.shape[1]
        n_x_j = locs_joint.shape[-1]; n_y_j = locs_joint.shape[-2]
        n_x_r = locs_ref.shape[-1]; n_y_r = locs_ref.shape[-2]
        locs_offset = self.offset
        k_ref = locs_ref.shape[-3]
        k_joint = locs_joint.shape[-3]
        ll_joint_flat = logits_joint.reshape([-1,k_joint*n_x_j*n_y_j])
        locs_ref = locs_ref * self.ref_scale

        preds_ref = torch.ones([bsz,n_max, n_classes,2],device=self.device) * np.nan
        conf_ref = torch.ones([bsz,n_max,n_classes],device=self.device)*-100
        preds_joint = torch.ones([bsz,n_max, n_classes,2],device=self.device) * np.nan
        pred_occ = torch.ones([bsz,n_max, n_classes],device=self.device) * np.nan
        conf_joint = torch.ones([bsz,n_max],device=self.device)*-100
        match_dist_factor = self.conf.multi_match_dist_factor
        assert ll_joint_flat.shape[1] >= n_min, f'The max number of animals with image size {self.conf.imsz} is {ll_joint_flat.shape[1]} while the minimum animals set is {n_min}'
        k = np.clip(n_max*5,n_min,ll_joint_flat.shape[1])
        for ndx in range(bsz):
            # n_preds = np.count_nonzero(ll_joint_flat[ndx,:]>0)
            # n_preds = np.clip(n_preds,n_min,np.inf)
            ids = ll_joint_flat[ndx,:].topk(k)[1]
            done_count = 0
            cur_n = 0
            while (done_count < n_max) and (cur_n<len(ids)):
                sel_ex = ids[cur_n]
                cur_n += 1

                if (ll_joint_flat[ndx,sel_ex] < 0) and (done_count >= n_min):
                    break

                idx = unravel_index(sel_ex, [k_joint,n_y_j, n_x_j])
                curp = locs_joint[ndx,...,idx[0],idx[1],idx[2]] * locs_offset
                dprev = torch.norm(preds_joint[ndx,...]-curp[None,...],dim=-1).mean(-1)
                # Find the animal size as the mean length of the bounding box
                cur_sz =  torch.mean(curp.max(axis=-2)[0]-curp.min(axis=-2)[0])
                nms_dist = cur_sz * match_dist_factor
                if ( not torch.all(torch.isnan(dprev))) and (nanmin(dprev) < nms_dist):
                    continue
                preds_joint[ndx,done_count,...] = locs_joint[ndx,...,idx[0],idx[1],idx[2]] * locs_offset
                if self.conf.predict_occluded:
                    pred_occ[ndx,done_count,...] = occ_out[ndx,...,idx[0],idx[1],idx[2]]
                conf_joint[ndx,done_count] = logits_joint[ndx,idx[0],idx[1],idx[2]]
                for cls in range(n_classes):
                    rpred = locs_joint[ndx, cls, :, idx[0], idx[1], idx[2]] * self.offset/self.ref_scale
                    mm = torch.round(rpred).int()
                    mm_y = torch.clamp(mm[1],0,n_y_r-1)
                    mm_x = torch.clamp(mm[0],0,n_x_r-1)
                    pt_selex = logits_ref[ndx,cls,:,mm_y,mm_x].argmax()
                    cur_pred = locs_ref[ndx,cls,:,pt_selex,mm_y,mm_x]
                    preds_ref[ndx,done_count,cls,:] = cur_pred
                    conf_ref[ndx,done_count,cls] = logits_ref[ndx,cls,pt_selex,mm_y,mm_x]

                done_count += 1

        preds_ref = preds_ref.detach().cpu().numpy().copy()
        preds_joint = preds_joint.detach().cpu().numpy().copy()
        conf_ref = conf_ref.detach().cpu().numpy().copy()
        conf_joint = conf_joint.detach().cpu().numpy().copy()
        pred_occ = pred_occ.detach().cpu().numpy().copy()


        return {'ref':preds_ref,'joint':preds_joint,'conf_joint':conf_joint,'conf_ref':conf_ref,'pred_occ':pred_occ}

    def get_joint_pred_up(self,preds,up_sample=8):
        n_max = self.conf.max_n_animals
        n_min = self.conf.min_n_animals
        locs_joint, logits_joint_in, locs_ref, logits_ref, occ_out = preds
        bsz = locs_joint.shape[0]
        n_classes = locs_joint.shape[1]

        l_sh = logits_joint_in.shape
        n_sh = (l_sh[2]*up_sample,l_sh[3]*up_sample)
        logits_joint = torch.nn.functional.interpolate(logits_joint_in,size=n_sh,mode='bicubic')

        # Do nms on logits_joint to reduce the number of candidates
        min_logit = logits_joint.min()
        l_pad = torch.nn.functional.pad(logits_joint, [1, 1, 1, 1], value=min_logit)
        lx1 = l_pad[:, :, 1:-1, 0:-2]
        lx2 = l_pad[:, :, 1:-1, 2:]
        ly2 = l_pad[:, :, 2:, 1:-1]
        ly1 = l_pad[:, :, 0:-2, 1:-1]
        l_nms = (logits_joint > lx1) & (logits_joint > lx2) & (logits_joint > ly1) & (logits_joint > ly2)
        logits_debug = logits_joint.clone()
        logits_joint[~l_nms] = min_logit

        n_x_j = locs_joint.shape[-1]; n_y_j = locs_joint.shape[-2]
        n_x_r = locs_ref.shape[-1]; n_y_r = locs_ref.shape[-2]
        locs_offset = self.offset
        k_ref = locs_ref.shape[-3]
        k_joint = locs_joint.shape[-3]
        ll_joint_flat = logits_joint.reshape([-1,k_joint*n_sh[0]*n_sh[1]])
        locs_ref = locs_ref * self.ref_scale

        preds_ref = torch.ones([bsz,n_max, n_classes,2],device=self.device) * np.nan
        conf_ref = torch.ones([bsz,n_max,n_classes],device=self.device)
        preds_joint = torch.ones([bsz,n_max, n_classes,2],device=self.device) * np.nan
        pred_occ = torch.ones([bsz,n_max, n_classes],device=self.device) * np.nan
        conf_joint = torch.ones([bsz,n_max],device=self.device)
        match_dist_factor = self.conf.multi_match_dist_factor
        assert ll_joint_flat.shape[1] > n_min, f'The max number of animals with image size {self.conf.imsz} is {ll_joint_flat.shape[1]} while the minimum animals set is {n_min}'

        k = np.clip(n_max*5,n_min,ll_joint_flat.shape[1])
        for ndx in range(bsz):
            # n_preds = np.count_nonzero(ll_joint_flat[ndx,:]>0)
            # n_preds = np.clip(n_preds,n_min,np.inf)
            ids = ll_joint_flat[ndx,:].topk(k)[1]
            done_count = 0
            cur_n = 0
            while (done_count < n_max) and (cur_n<len(ids)):
                sel_ex = ids[cur_n]
                cur_n += 1

                if (ll_joint_flat[ndx,sel_ex] < 0) and (done_count >= n_min):
                    break

                idx = unravel_index(sel_ex, [k_joint,n_sh[0], n_sh[1]])
                idx = list(idx)
                idx[1] = torch.round(torch.true_divide(idx[1],up_sample)).int()
                idx[2] = torch.round(torch.true_divide(idx[2],up_sample)).int()
                curp = locs_joint[ndx,...,idx[0],idx[1],idx[2]] * locs_offset
                dprev = torch.norm(preds_joint[ndx,...]-curp[None,...],dim=-1).mean(-1)

                # Find the animal size as the mean length of the bounding box
                cur_sz =  torch.mean(curp.max(axis=-2)[0]-curp.min(axis=-2)[0])
                nms_dist = cur_sz * match_dist_factor
                if ( not torch.all(torch.isnan(dprev))) and (nanmin(dprev) < nms_dist):
                    continue
                preds_joint[ndx,done_count,...] = locs_joint[ndx,...,idx[0],idx[1],idx[2]] * locs_offset
                if self.conf.predict_occluded:
                    pred_occ[ndx,done_count,...] = occ_out[ndx,...,idx[0],idx[1],idx[2]]
                conf_joint[ndx,done_count] = logits_joint[ndx,idx[0],idx[1],idx[2]]
                for cls in range(n_classes):
                    rpred = locs_joint[ndx, cls, :, idx[0], idx[1], idx[2]] * self.offset/self.ref_scale
                    mm = torch.round(rpred).int()
                    mm_y = torch.clamp(mm[1],0,n_y_r-1)
                    mm_x = torch.clamp(mm[0],0,n_x_r-1)
                    pt_selex = logits_ref[ndx,cls,:,mm_y,mm_x].argmax()
                    cur_pred = locs_ref[ndx,cls,:,pt_selex,mm_y,mm_x]
                    preds_ref[ndx,done_count,cls,:] = cur_pred
                    conf_ref[ndx,done_count,cls] = logits_ref[ndx,cls,pt_selex,mm_y,mm_x]

                done_count += 1

        preds_ref = preds_ref.detach().cpu().numpy().copy()
        preds_joint = preds_joint.detach().cpu().numpy().copy()
        conf_ref = conf_ref.detach().cpu().numpy().copy()
        conf_joint = conf_joint.detach().cpu().numpy().copy()
        pred_occ = pred_occ.detach().cpu().numpy().copy()


        return {'ref':preds_ref,'joint':preds_joint,'conf_joint':conf_joint,'conf_ref':conf_ref,'pred_occ':pred_occ}

    def compute_dist(self, output, labels):
        locs = labels['locs'].numpy().copy()
        locs[locs<-1000] = np.nan
        is_valid_nan = np.any(np.invert(np.isnan(locs)),(-1,-2))
        is_valid_low = np.any(locs>-10000,(-1,-2))
        is_valid = is_valid_low & is_valid_nan
        pp = self.get_joint_pred(output)['ref']
        dd_all = np.linalg.norm(locs[:,np.newaxis,...]-pp[:,:,np.newaxis,...],axis=-1)
        dd = np.nanmean(dd_all,axis=-1)
        mm = np.nanmin(dd,axis=1) # closest prediction to each label
        mean_d = np.mean(mm[is_valid])

        return mean_d


    def match_preds(self,locs,olocs,match_dist_factor):
        # Match predictions with offseted predictions
        # Prefix o mean offset predictions
        # To match the predictions, we first find the distances between predictions. If this distances is smaller than 10% (or so) of the average side size of the predictions bounding-box then we merge them.

        matched = {}
        hsz = self.offset // 2
        conf = self.conf

        # Initialize all the arrays.
        olocs_orig = olocs['ref'] + hsz
        locs_orig = locs['ref']
        valid_locs = ~np.all(np.isnan(locs_orig),axis=(2,3))
        valid_olocs = ~np.all(np.isnan(olocs_orig),axis=(2,3))
        cur_pred = np.ones_like(olocs_orig) * np.nan
        cur_joint_conf = np.ones_like(olocs_orig[..., 0]) * -100
        cur_ref_conf = np.ones_like(olocs_orig[..., 0]) * -100
        cur_occ_pred = np.ones_like(olocs_orig[..., 0]) * np.nan
        dd = olocs_orig[:, :, np.newaxis, ...] - locs_orig[:, np.newaxis, ...]
        dd = np.linalg.norm(dd, axis=-1).mean(-1)
        # dd has offset predictions along axis 1 and normal predictions along axis 2

        # Bounding box sizes to compare the distances between predictions too.
        bb = locs_orig.max(axis=-2) - locs_orig.min(axis=-2)
        bb_sz = (bb[...,0]+bb[...,1])/2 # Ideally should be geometric mean but algebraic is better behaved.
        obb = olocs_orig.max(axis=-2) - olocs_orig.min(axis=-2)
        obb_sz = (obb[...,0]+obb[...,1])/2

        conf_margin = 4
        # Average both predictions if the confidences of the predictions are withing this of each other. Else pick the dominant one.

        # match predictions from offset pred and normal preds
        for b in range(dd.shape[0]):
            # loop over examples
            done_offset = np.zeros(dd.shape[1])
            done_locs = np.zeros(dd.shape[1])
            mpred = []
            for ix in range(dd.shape[1]):

                if not valid_locs[b,ix]: continue
                if np.all(np.isnan(dd[b, :, ix])):
                    # no predictions at all in  offset predictions
                    mpred.append((locs_orig[b,ix],locs['conf_joint'][b,ix],locs['pred_occ'][b,ix]))
                    done_locs[ix] = 1
                    continue

                olocs_ndx = np.nanargmin(dd[b, :, ix])
                # Find the closest one in offset prediction.

                bb_avg = (bb_sz[b,ix]+obb_sz[b,olocs_ndx])/2
                if dd[b, olocs_ndx, ix] < bb_avg*match_dist_factor:

                    # Select the one with higher confidence unless they both are close.
                    if locs['conf_joint'][b, ix] < olocs['conf_joint'][b, olocs_ndx] - conf_margin:
                        cc = olocs_orig[b, olocs_ndx, ...]
                        oo = olocs['pred_occ'][b, olocs_ndx, ...]
                    elif locs['conf_joint'][b, ix] - conf_margin > olocs['conf_joint'][b, olocs_ndx]:
                        cc = locs_orig[b, ix, ...]
                        oo = locs['pred_occ'][b, ix, ...]
                    else:
                        cc = (olocs_orig[b, olocs_ndx, ...] + locs_orig[b, ix, ...]) / 2
                        oo = (olocs['pred_occ'][b, olocs_ndx, ...] + locs['pred_occ'][b, ix, ...]) / 2

                    # code below is for selecting based on ref confidences.
                    # cc = np.ones([conf.n_classes,2])*np.nan
                    # for cls in range(conf.n_classes):
                    #     if olocs['conf_ref'][b,olocs_ndx,cls] > locs['conf_ref'][b,ix,cls]:
                    #         cc[cls,:] = olocs_orig[b,olocs_ndx,cls,:]
                    #     else:
                    #         cc[cls,:] = locs_orig[b,ix,cls,...]

                    done_offset[olocs_ndx] = 1
                    done_locs[ix] = 1
                    mconf = max(olocs['conf_joint'][b, olocs_ndx], locs['conf_joint'][b, ix])
                    # print(f'Matched {ix} with {olocs_ndx}')
                else:
                    cc = locs_orig[b, ix, ...]
                    oo = locs['pred_occ'][b, ix]
                    done_locs[ix] = 1
                    mconf = locs['conf_joint'][b, ix]
                mpred.append((cc, mconf, oo))

            for ix in np.where(done_offset < 0.5)[0]:
                if not valid_olocs[b,ix]:
                    continue
                cc = olocs_orig[b, ix, ...]
                oo = olocs['pred_occ'][b, ix]
                mconf = olocs['conf_joint'][b, ix]
                mpred.append((cc, mconf, oo))

            pconf = np.array([m[1] for m in mpred])
            ord = np.flip(np.argsort(pconf))[:conf.max_n_animals]
            if len(ord) > conf.min_n_animals:
                neg_ndx = np.where(pconf[ord[conf.min_n_animals:]] < 0)[0]
                if neg_ndx.size > 0:
                    ord = ord[:(conf.min_n_animals + neg_ndx[0])]

            bpred = [mpred[ix][0] for ix in ord]
            bpred = np.array(bpred)
            opred = [mpred[ix][2] for ix in ord]
            opred = np.array(opred)
            npred = bpred.shape[0]
            if npred > 0:
                cur_pred[b, :npred, ...] = bpred
                cur_occ_pred[b, :npred, ...] = opred
                cur_joint_conf[b, :npred, ...] = pconf[ord, None]

        matched['ref'] = cur_pred
        return matched, cur_joint_conf, cur_occ_pred

    def set_version(self,model_file):
        ckpt = torch.load(model_file, map_location=torch.device('cpu'))
        k_j = ckpt['model_state_params']['module.wts_joint.conv_out.weight'].shape[0]
        if k_j==4:
            self.version = 1

        self.k_j = k_j


    def get_pred_fn_2pass(self, model_file=None,max_n=None,imsz=None):
        if max_n is not None:
            self.conf.max_n_animals = max_n
        if imsz is not None:
            self.conf.imsz = imsz
        model = self.create_model()
        model = torch.nn.DataParallel(model)

        if model_file is None:
            latest_model_file = self.get_latest_model_file()
        else:
            latest_model_file = model_file

        self.restore(latest_model_file,model)
        model.to(self.device)
        model.eval()
        self.model = model
        conf = self.conf
        # conf.batch_size = 1

        # TODO: Change this to multi_match_dist_factor
        match_dist_factor = self.conf.get('multi_match_dist_factor',0.2)

        def pred_fn(ims_in, retrawpred=False):
            locs_sz = (conf.batch_size, conf.n_classes, 2)
            locs_dummy = np.zeros(locs_sz)
            ims_in, _ = PoseTools.preprocess_ims(ims_in,locs_dummy,conf,False,conf.rescale)
            ret_dict = {}
            ret_dict['locs'] = []
            ret_dict['conf'] = []
            ret_dict['occ'] = []
            if retrawpred:
                ret_dict['preds'] = [[],[]]
                ret_dict['raw_locs'] = [[],[]]

            for ndx, ims in enumerate(ims_in):
                # do prediction on half grid cell size offset images. o is for offset
                hsz = self.offset//2
                ims = torch.tensor(ims[None]).to(self.device).permute([0,3,1,2])/255.
                oims = torch.nn.functional.pad(ims, [0, hsz,0, hsz])[:,:, hsz:, hsz:]
                with torch.no_grad():
                    preds = model({'images':ims})
                    opreds = model({'images':oims})
                    locs = self.get_joint_pred(preds)
                    olocs = self.get_joint_pred(opreds)


                matched, cur_joint_conf, cur_occ_pred = self.match_preds(locs,olocs,match_dist_factor)
                ret_dict['locs'].append(matched['ref'][0] * conf.rescale)
                ret_dict['conf'].append(1/(1+np.exp(-cur_joint_conf[0])))
                if self.conf.predict_occluded:
                    ret_dict['occ'].append(cur_occ_pred[0])
                else:
                    ret_dict['occ'].append(np.ones_like(cur_occ_pred[0])*np.nan)

                if retrawpred:
                    ret_dict['preds'][0].append(preds)
                    ret_dict['preds'][1].append(opreds)
                    ret_dict['raw_locs'][0].append(locs)
                    ret_dict['raw_locs'][1].append(olocs)
            ret_dict['locs'] = np.array(ret_dict['locs'] )
            ret_dict['conf'] = np.array(ret_dict['conf'])
            ret_dict['occ'] = np.array(ret_dict['occ'])
            return ret_dict

        def close_fn():
            torch.cuda.empty_cache()

        return pred_fn, close_fn, latest_model_file



    def get_pred_fn_fast(self, model_file=None,max_n=None,imsz=None):
        if max_n is not None:
            self.conf.max_n_animals = max_n
        if imsz is not None:
            self.conf.imsz = imsz

        if model_file is None:
            latest_model_file = self.get_latest_model_file()
        else:
            latest_model_file = model_file


        model = self.create_model()
        model = torch.nn.DataParallel(model)


        self.restore(latest_model_file,model)
        model.to(self.device)
        model = model.eval()
        self.model = model
        conf = self.conf
        # conf.batch_size = 1
        match_dist_factor = self.conf.get('multi_match_dist_factor',0.2)

        def pred_fn(ims_in, retrawpred=False):
            locs_sz = (conf.batch_size, conf.n_classes, 2)
            locs_dummy = np.zeros(locs_sz)
            ims_in, _ = PoseTools.preprocess_ims(ims_in,locs_dummy,conf,False,conf.rescale)
            ret_dict = {}
            ret_dict['locs'] = []
            ret_dict['conf'] = []
            ret_dict['occ'] = []
            if retrawpred:
                ret_dict['preds'] = [[]]
                ret_dict['raw_locs'] = [[]]

            for ndx, ims in enumerate(ims_in):
                # do prediction on half grid cell size offset images. o is for offset
                hsz = self.offset//2
                ims = torch.tensor(ims[None]).to(self.device).permute([0,3,1,2])/255.
                # oims = torch.nn.functional.pad(ims, [0, hsz,0, hsz])[:,:, hsz:, hsz:]
                with torch.no_grad():
                    preds = model({'images':ims})
                    locs = self.get_joint_pred(preds)

                ret_dict['locs'].append(locs['ref'][0] * conf.rescale)
                conf_joint = 1/(1+np.exp(-locs['conf_joint']))
                conf_ref = 1/(1+np.exp(-locs['conf_ref']))
                pred_conf = conf_joint[...,None]*conf_ref
                ret_dict['conf'].append(pred_conf[0])
                if self.conf.predict_occluded:
                    ret_dict['occ'].append(locs['pred_occ'][0])
                else:
                    ret_dict['occ'].append(np.ones_like(locs['ref'][0][..., 0]) * np.nan)

                if retrawpred:
                    ret_dict['preds'].append(preds)
                    ret_dict['raw_locs'].append(locs)
            ret_dict['locs'] = np.array(ret_dict['locs'])
            ret_dict['conf'] = np.array(ret_dict['conf'])
            ret_dict['occ'] = np.array(ret_dict['occ'])
            return ret_dict

        def close_fn():
            del self.model
            torch.cuda.empty_cache()

        return pred_fn, close_fn, latest_model_file


    def get_pred_fn(self,model_file,**kwargs):
        if model_file is None:
            latest_model_file = self.get_latest_model_file()
        else:
            latest_model_file = model_file
        self.set_version(latest_model_file)
        return self.get_pred_fn_fast(model_file,**kwargs)
        # return self.get_pred_fn_2pass(model_file,**kwargs)