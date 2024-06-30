import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead

import numpy as np
import cv2
import os
from tools.feature_map_visualization import show_feature_map


class DecoupledConvFC(nn.Module):
    def __init__(self,
                 num_cls_fcs,
                 num_reg_convs,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(DecoupledConvFC, self).__init__()
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.cls_fcs = self._add_fc_branch(num_branch_fcs=self.num_cls_fcs)
        self.reg_convs = self._add_conv_branch(num_branch_convs=self.num_reg_convs)

    def _add_fc_branch(self, num_branch_fcs):
        branch_fcs = nn.ModuleList()
        for i in range(num_branch_fcs):
            fc_in_channels = (
                self.in_channels if i == 0 else self.out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.out_channels))
        return branch_fcs

    def _add_conv_branch(self, num_branch_convs):
        branch_convs = nn.ModuleList()
        for i in range(num_branch_convs):
            branch_convs.append(
                ConvModule(
                    256,
                    256,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return branch_convs

    def init_weights(self):
        super(DecoupledConvFC, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_cls, x_reg):
        for cls_fc in self.cls_fcs:
            x_cls = F.relu(cls_fc(x_cls))
        for reg_conv in self.reg_convs:
            x_reg = reg_conv(x_reg)
        return x_cls, x_reg


class TaskInteractive(nn.Module):
    def __init__(self, in_channels_cls, in_channels_reg):
        super(TaskInteractive, self).__init__()
        self.cls2reg = nn.Sequential(
            nn.Linear(in_channels_cls, in_channels_reg),
            nn.ReLU(inplace=True))
        self.reg2cls = nn.Sequential(
            nn.Linear(in_channels_reg, in_channels_cls),
            nn.ReLU(inplace=True))
        self.cls = nn.Sequential(
            nn.Linear(in_channels_cls, in_channels_cls),
            nn.Tanh())
        self.reg = ConvModule(
            256,
            256,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=dict(type='Tanh'))
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, f_cls, f_reg):
        assert len(f_cls) == len(f_reg)
        len_f = len(f_cls)
        n, c, w, h = f_reg[0].size()
        f_base_cls = torch.zeros_like(f_cls[0])
        f_base_reg = torch.zeros_like(f_reg[0])
        for i in range(len_f):
            f_base_cls = f_base_cls + f_cls[i]
            f_base_reg = f_base_reg + f_reg[i]
        f_base_reg = f_base_reg.flatten(1)
        
        f_cls_ = f_base_cls + self.reg2cls(f_base_reg)
        f_reg_ = f_base_reg + self.cls2reg(f_base_cls)
        f_reg_ = f_reg_.view(n, c, w, h)

        w_cls = self.cls(f_cls_)
        w_reg = self.reg(f_reg_)
        f_cls_ = (1 + w_cls) * f_cls_
        f_reg_ = (1 + w_reg) * f_reg_
        f_reg_ = f_reg_.flatten(1)

        return f_cls_, f_reg_


@HEADS.register_module()
class SemiDeConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_ti=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 fc_out_channels=0,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(SemiDeConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_cls_fcs + num_reg_convs > 0)
        if not self.with_cls:
            assert num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0
        self.num_ti = num_ti
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.task_interactive = TaskInteractive(
            in_channels_cls=self.fc_out_channels,
            in_channels_reg=self.roi_feat_area * self.in_channels)
        self.multi_stage_decoupled_modules = nn.ModuleList()
        for i in range(self.num_ti):
            fc_in_channels = (self.roi_feat_area * self.in_channels
                              if i == 0 else self.fc_out_channels)
            self.multi_stage_decoupled_modules.append(DecoupledConvFC(
                num_cls_fcs=self.num_cls_fcs,
                num_reg_convs=self.num_reg_convs,
                in_channels=fc_in_channels,
                out_channels=self.fc_out_channels))

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
            self.fc_reg = nn.Linear(self.roi_feat_area * self.in_channels, out_dim_reg)

    def forward(self, x):
        x_cls = x
        x_reg = x
        x_cls = x_cls.flatten(1)
        f_cls = []
        f_reg = []
        for i in range(self.num_ti):
            x_cls, x_reg = self.multi_stage_decoupled_modules[i](x_cls, x_reg)
            f_cls.append(x_cls)
            f_reg.append(x_reg)
        f_cls_, f_reg_ = self.task_interactive(f_cls, f_reg)
        cls_score = self.fc_cls(f_cls_) if self.with_cls else None
        bbox_pred = self.fc_reg(f_reg_) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class SemiDeConv2FCBBoxHead(SemiDeConvFCBBoxHead):
    def __init__(self, fc_out_channels, *args, **kwargs):
        super(SemiDeConv2FCBBoxHead, self).__init__(
            num_ti=2,
            num_cls_fcs=1,
            num_reg_convs=2,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)