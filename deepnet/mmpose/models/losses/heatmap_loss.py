# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)

    def forward(self, output, target, target_weight):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output * target_weight.unsqueeze(-1),
                                  target * target_weight.unsqueeze(-1))
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@LOSSES.register_module()
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
