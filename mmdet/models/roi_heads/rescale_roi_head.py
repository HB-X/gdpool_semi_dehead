# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class RescaleRoIHead(StandardRoIHead):

    def __init__(self, 
                 fc_in_channels=12544,
                 compressed_channels=256,
                 out_channels=2, 
                 sigma=0.5, 
                 gamma=0.1,
                 **kwargs):
        super(RescaleRoIHead, self).__init__(**kwargs)
        self.sigma = sigma
        self.gamma = gamma
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in_channels, compressed_channels),
            nn.ReLU(inplace=True))
        self.rescale_fc = nn.Sequential(
            nn.Linear(compressed_channels, compressed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_channels, out_channels),
            nn.Sigmoid())
        self.offset_fc = nn.Sequential(
            nn.Linear(compressed_channels, compressed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_channels, out_channels))

    def roi_transformer(self, rois, offset_factor, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """

        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_cx = cx + offset_factor[:, 0] * w
        new_cy = cy + offset_factor[:, 1] * h
        new_w = scale_factor[:, 0] * w
        new_h = scale_factor[:, 1] * h
        x1 = new_cx - new_w * 0.5
        x2 = new_cx + new_w * 0.5
        y1 = new_cy - new_h * 0.5
        y2 = new_cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing time."""
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_cls_feats = bbox_feats
        bbox_feats_compressed = self.shared_fc(bbox_feats.flatten(1))
        roi_rescale_factor = self.rescale_fc(bbox_feats_compressed)
        roi_rescale_factor = 1.0 + self.sigma * roi_rescale_factor
        roi_offset_factor = self.offset_fc(bbox_feats_compressed)
        roi_offset_factor = self.gamma * roi_offset_factor
        new_rois = self.roi_transformer(rois, roi_offset_factor, roi_rescale_factor)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            new_rois)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results
