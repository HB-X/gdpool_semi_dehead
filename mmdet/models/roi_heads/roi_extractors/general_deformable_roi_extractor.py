# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .single_level_roi_extractor import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class GeneralDeformableRoIExtractor(SingleRoIExtractor):
    def __init__(self,
                 **kwargs):
        super(GeneralDeformableRoIExtractor, self).__init__(**kwargs)
    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
            roi_feats_cls = roi_feats
            roi_feats_reg = roi_feats
        else:
            roi_feats_cls = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
            roi_feats_reg = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats_cls.requires_grad = True
            roi_feats_reg.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats_cls, roi_feats_reg
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)


        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i *= mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats_cls.shape)
                roi_feats_cls_t, roi_feats_reg_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_cls_t *= mask_exp
                roi_feats_reg_t *= mask_exp
                roi_feats_cls += roi_feats_cls_t
                roi_feats_reg += roi_feats_reg_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_cls_t, roi_feats_reg_t = self.roi_layers[i](feats[i], rois_)
                roi_feats_cls[inds] = roi_feats_cls_t
                roi_feats_reg[inds] = roi_feats_reg_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats_cls += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
                roi_feats_reg += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        return roi_feats_cls, roi_feats_reg
    