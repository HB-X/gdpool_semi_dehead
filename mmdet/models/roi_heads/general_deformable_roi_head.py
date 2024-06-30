# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class GeneralDeformableRoIHead(StandardRoIHead):

    def __init__(self,
                 **kwargs):
        super(GeneralDeformableRoIHead, self).__init__(**kwargs)

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing time."""
        bbox_feats_cls, bbox_feats_reg = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats_cls = self.shared_head(bbox_feats_cls)
            bbox_feats_reg = self.shared_head(bbox_feats_reg)
        cls_score, bbox_pred = self.bbox_head(bbox_feats_cls, bbox_feats_reg)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats_cls)
        return bbox_results
