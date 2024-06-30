import torch
import torch.nn as nn

import mmcv

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def consistency_loss(cls_scores,
              ious,
              weight=None,
              eps=0.6,
              reduction='mean',
              avg_factor=None):
    """Calculate consistency beteen class score and iou.

    Args:
        cls_score (torch.Tensor): The prediction, has a shape (n, *)
        iou (torch.Tensor): same shape of cls_score.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    loss = (1 - cls_scores * ious) * ((cls_scores + eps) / (ious + eps) + (ious + eps) / (cls_scores + eps) - 1.7)
    return loss


@LOSSES.register_module()
class ConsistencyLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=0.6):
        """`Dice Loss, which is proposed in
        `V-Net: Fully Convolutional Neural Networks for Volumetric
         Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self,
                cls_scores,
                ious,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            cls_scores (torch.Tensor): The prediction, has a shape (n, *).
            ious (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * consistency_loss(
            cls_scores,
            ious,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss