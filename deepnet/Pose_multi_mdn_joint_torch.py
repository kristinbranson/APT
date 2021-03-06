import PoseCommon_pytorch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN
import numpy as np
import PoseTools
from torchvision.ops import misc as misc_nn_ops

class pred_layers(nn.Module):

    def __init__(self, n_in, n_out):
        super(pred_layers,self).__init__()
        self.conv1 = nn.Conv2d(n_in,n_in, 3, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.bn1  = nn.BatchNorm2d(n_in)
        self.conv2 = nn.Conv2d(n_in,n_in, 3, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.bn2  = nn.BatchNorm2d(n_in)
        self.conv3 = nn.Conv2d(n_in,n_in, 3, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        self.bn3  = nn.BatchNorm2d(n_in)
        self.conv_out = nn.Conv2d(n_in,n_out, 1, padding=0)
        torch.nn.init.xavier_normal_(self.conv_out.weight)

    def forward(self, x):
        x1 = self.bn1(self.conv1(F.relu(x)))
        x = self.bn2(self.conv2(F.relu(x1)))
        x = self.bn3(self.conv3(F.relu(x)))
        x = F.relu(x + x1)
        x = self.conv_out(x)
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

def my_resnet_fpn_backbone(backbone_name, pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):
    backbone = models.resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
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

    def __init__(self, npts, device, use_fpn=True,k_j=4,k_r=3):
        super(mdn_joint,self).__init__()
        if use_fpn:
            # Use already available fpn. woohoo.
            backbone = my_resnet_fpn_backbone('resnet50',pretrained=True,trainable_layers=5)
            n_ftrs = backbone.fpn.layer_blocks[0].out_channels
        else:
            backbone = models.resnet50(pretrained=True)
            n_ftrs = backbone.layer4[2].conv3.weight.shape[0]

            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone.apply(freeze_bn)

        self.backbone = backbone

        self.locs_joint = pred_layers(n_ftrs, 2*npts*k_j)
        self.wts_joint = pred_layers(n_ftrs, k_j)
        self.locs_ref = pred_layers(n_ftrs,npts*2*k_r)
        self.wts_ref = pred_layers(n_ftrs,npts*k_r)
        self.npts= npts
        self.device = device
        self.k_r = k_r
        self.k_j = k_j
        self.use_fpn = use_fpn
        self.im_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).to(self.device)
        self.im_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).to(self.device)

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
        if self.use_fpn:
            x_j = x['3']
            x_r = x['0']
        else:
            x_j = x
            x_r = x

        locs_j = self.locs_joint(x_j)
        wts_j = self.wts_joint(x_j)
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
        return locs_j_off, wts_j, locs_r_off, wts_r


class Pose_multi_mdn_joint_torch(PoseCommon_pytorch.PoseCommon_pytorch):

    def __init__(self,conf,**kwargs):
        super(Pose_multi_mdn_joint_torch,self).__init__(conf,**kwargs)
        self.offset = 32
        self.locs_noise = self.conf.get('mdn_joint_ref_noise',0.3)
        self.k_j = 4
        self.k_r = 3

    def create_model(self):
        self.offset = 32
        use_fpn = self.conf.get('mdn_joint_use_fpn',True)
        self.ref_scale = 8 if use_fpn else 1
        return mdn_joint(self.conf.n_classes, self.device, use_fpn=use_fpn,k_j=self.k_j,k_r=self.k_r)

    def loss_slow(self, preds, labels):
        n_classes = self.conf.n_classes
        offset = self.offset
        locs_joint, wts_joint, locs_ref, wts_ref = preds
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
        locs_noise = locs_noise*self.ref_scale

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
                    cur_ref = locs_ref[b,cls,:,:,idx_y[cls],idx_x[cls]] * self.offset/self.ref_scale

                    dd_ref = torch.norm(pp-cur_ref.T,dim=-1)
                    ll_ref = torch.softmax(wts_ref[b,cls,:,idx_y[cls],idx_x[cls]],0)
                    cur_loss_ref = torch.sum(dd_ref*ll_ref)
                    cur_loss = cur_loss + cur_loss_ref

        ref_loss = cur_loss

        tot_loss = wt_loss  + joint_loss + ref_loss
        return tot_loss / n_classes


    def loss(self, preds, labels):
        n_classes = self.conf.n_classes
        offset = self.offset
        locs_joint, wts_joint, locs_ref, wts_ref = preds
        labels = labels.to(self.device)

        ll_joint = torch.sigmoid(wts_joint)
        ls = locs_joint.shape
        locs_joint_flat = locs_joint.reshape(
            [locs_joint.shape[0], 1, n_classes, 2, self.k_j * ls[-2] * ls[-1]]).permute([0, 1, 4, 2, 3])
        ll_joint_flat = ll_joint.reshape([ll_joint.shape[0], self.k_j * ls[-1] * ls[-2]])
        valid = torch.any(torch.all(labels > -1000, dim=3), dim=2)
        valid_lbl = labels[...,0] > -1000
        # assert torch.all(torch.any(valid,dim=1)), 'Some inputs dont have any labels'

        dd = torch.norm(locs_joint_flat * offset - labels.unsqueeze(2), dim=-1)
        # dd has the distance between all the predictions and all the labels

        qq = torch.unsqueeze(labels[..., 0], 2)
        qq = qq.repeat([1, 1, ls[-1] * ls[-2] * self.k_j, 1])
        dd = torch.where(qq > -1000, dd, torch.zeros_like(dd))
        # Set the distances to 0 where labels are invalid

        dd_all = dd.sum(-1)
        qq_all = torch.unsqueeze(valid,2)
        qq_all = qq_all.repeat([1,1,ls[-1]*ls[-2]*self.k_j])
        dd_all = torch.where(qq_all,dd_all,10000*torch.ones_like(dd_all)*self.conf.n_classes)
        # Set distances to invalid instances to a very high value.
        p_assign = torch.softmax(-dd_all.detach(), axis=1)

        # assign each prediction to the label based on its distance. The compute the weighted loss between the prediction and that label
        assign = p_assign * torch.unsqueeze(ll_joint_flat, 1)
        assign_sum = torch.where(valid,assign.sum(axis=-1),torch.ones_like(assign[:,:,0]))
        assign_norm = assign/ torch.unsqueeze(assign_sum+0.0001,dim=-1)
        dloss = (assign_norm*dd_all).sum(axis=-1)
        cur_pred_loss = torch.where(valid,dloss,torch.zeros_like(dloss)).sum()
        wt_loss_all = (1.-assign.sum(axis=-1))**2
        cur_wt_loss = torch.where(valid,wt_loss_all,torch.zeros_like(wt_loss_all)).sum()
        joint_loss = cur_pred_loss #/ self.offset
        wt_loss = cur_wt_loss * n_classes * 10

        # Loss to ensure that the number of predictions match the number of labeled animals. This weight needs to be upweighted otherwise the training converges to degenerate case where  ll_joint is predicted as zero, which would make joint_loss 0.

        locs_noise_mag = self.locs_noise
        locs_noise_mag = torch.tensor(locs_noise_mag).to(self.device)
        if locs_noise_mag > 0.001:
            locs_noise = locs_joint_flat + (
                        torch.rand(locs_joint_flat.shape, device=self.device) - 0.5) * 2 * locs_noise_mag
        else:
            locs_noise = locs_joint
        locs_noise = locs_noise * self.ref_scale

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
        ref_pred = ref_pred*self.offset/self.ref_scale
        ref_wts = wts_ref_dim[i1, i2, i3, :, idx_y, idx_x]
        ref_wts = torch.softmax(ref_wts, 3)
        ref_dist = torch.norm(ref_pred - torch.unsqueeze(labels, -1), dim=-2)
        ref_loss_all = torch.sum(ref_wts * ref_dist, dim=-1)
        ref_loss_sel = torch.where(valid_lbl, ref_loss_all, torch.zeros_like(ref_loss_all))
        ref_loss = ref_loss_sel.sum()

        # cur_loss = 0
        # for b in range(ll_joint.shape[0]):
        #     for g in range(labels.shape[1]):
        #         if not valid[b,g]:
        #             continue
        #         assign = p_assign[b,g,:]*ll_joint_flat[b,:]
        #         selex = torch.argmax(assign)
        #         # torch supports histogramming. Maybe instead of argmax that can be used.
        #
        #         idx = torch.round(locs_noise[b, 0, selex, :, :]).int()
        #         # ids are predicted as x,y to match input locs.
        #         idx_y = torch.clamp(idx[:,1], 0, locs_ref.shape[-2] - 1)
        #         idx_x = torch.clamp(idx[:,0], 0, locs_ref.shape[-1] - 1)
        #         for cls in range(labels.shape[2]):
        #             if (labels[b,g,cls,0] < -1000) or torch.isnan(labels[b,g,cls,0]):
        #                 continue
        #
        #             pp = labels[b, g, cls:cls + 1, :]
        #             cur_ref = locs_ref[b,cls,:,:,idx_y[cls],idx_x[cls]] * self.offset/self.ref_scale
        #
        #             dd_ref = torch.norm(pp-cur_ref.T,dim=-1)
        #             ll_ref = torch.softmax(wts_ref[b,cls,:,idx_y[cls],idx_x[cls]],0)
        #             cur_loss_ref = torch.sum(dd_ref*ll_ref)
        #             cur_loss = cur_loss + cur_loss_ref
        #
        # ref_loss1 = cur_loss

        tot_loss = wt_loss  + joint_loss + ref_loss
        return tot_loss / n_classes



    def create_targets(self, inputs):
        return inputs['locs']

    def get_joint_pred(self,preds):
        n_max = self.conf.max_n_animals
        locs_joint, logits_joint, locs_ref, logits_ref = self.to_numpy(preds)
        locs_joint = locs_joint
        bsz = locs_joint.shape[0]
        n_classes = locs_joint.shape[1]
        n_x_j = locs_joint.shape[-1]; n_y_j = locs_joint.shape[-2]
        n_x_r = locs_ref.shape[-1]; n_y_r = locs_ref.shape[-2]
        locs_offset = self.offset
        k_ref = locs_ref.shape[-3]
        k_joint = locs_joint.shape[-3]
        ll_joint_flat = np.reshape(logits_joint,[-1,k_joint*n_x_j*n_y_j])
        locs_ref = locs_ref * locs_offset / self.ref_scale

        preds_ref = np.ones([bsz,n_max, n_classes,2]) * np.nan
        preds_joint = np.ones([bsz,n_max, n_classes,2]) * np.nan

        for ndx in range(bsz):
            n_preds = np.count_nonzero(ll_joint_flat[ndx,:]>0)
            n_preds = np.clip(n_preds,1,n_max)
            ids = np.argsort(-ll_joint_flat[ndx,:])
            for cur_n in range(n_preds):
                sel_ex = ids[cur_n]
                idx = np.unravel_index(sel_ex, [k_joint,n_y_j, n_x_j])
                preds_joint[ndx,cur_n,...] = locs_joint[ndx,...,idx[0],idx[1],idx[2]] * locs_offset
                for cls in range(n_classes):
                    mm = np.round(locs_joint[ndx,cls,:,idx[0],idx[1], idx[2]]*self.ref_scale).astype('int')
                    mm_y = np.clip(mm[1],0,n_y_r-1)
                    mm_x = np.clip(mm[0],0,n_x_r-1)
                    pt_selex = np.argmax(logits_ref[ndx,cls,:,mm_y,mm_x])
                    cur_pred = locs_ref[ndx,cls,:,pt_selex,mm_y,mm_x]
                    preds_ref[ndx,cur_n,cls,:] = cur_pred
        return {'ref':preds_ref,'joint':preds_joint}

    def compute_dist(self, output, labels):
        locs = labels.numpy().copy()
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


    def get_pred_fn(self, model_file=None,max_n=None,imsz=None):
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

        def pred_fn(ims):
            locs_sz = (conf.batch_size, conf.n_classes, 2)
            locs_dummy = np.zeros(locs_sz)

            ims, _ = PoseTools.preprocess_ims(ims,locs_dummy,conf,False,conf.rescale)
            with torch.no_grad():
                preds = model({'images':torch.tensor(ims).permute([0,3,1,2])/255.})
            locs = self.get_joint_pred(preds)
            ret_dict = {}
            ret_dict['locs'] = locs['ref'] * conf.rescale
            ret_dict['locs_joint'] = locs['joint'] * conf.rescale
            return ret_dict

        def close_fn():
            torch.cuda.empty_cache()

        return pred_fn, close_fn, latest_model_file