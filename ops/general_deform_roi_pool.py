# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from mmcv.cnn import ConvModule
import numpy as np
from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['general_deform_roi_pool_forward', 'general_deform_roi_pool_backward'])


class GeneralDeformRoIPoolFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, rescale, offset, output_size, spatial_scale,
                 sampling_ratio, gamma, sigma, beta):
        return g.op(
            'mmcv::MMCVGeneralDeformRoIPool',
            input,
            rois,
            rescale,
            offset,
            pooled_height_i=output_size[0],
            pooled_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_f=sampling_ratio,
            gamma_f=gamma,
            sigma_f=sigma,
            beta_f=beta)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                rescale,
                offset,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                gamma=0.1,
                sigma=0.5,
                beta=0.0):
        if rescale is None:
            rescale = input.new_zeros(0)
        if offset is None:
            offset = input.new_zeros(0)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = float(spatial_scale)
        ctx.sampling_ratio = int(sampling_ratio)
        ctx.gamma = float(gamma)
        ctx.sigma = float(sigma)
        ctx.beta = float(beta)

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)

        ext_module.general_deform_roi_pool_forward(
            input,
            rois,
            rescale,
            offset,
            output,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            gamma=ctx.gamma,
            sigma=ctx.sigma,
            beta=ctx.beta)

        ctx.save_for_backward(input, rois, rescale, offset)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, rescale, offset = ctx.saved_tensors
        grad_input = grad_output.new_zeros(input.shape)
        grad_rescale = grad_output.new_zeros(rescale.shape)
        grad_offset = grad_output.new_zeros(offset.shape)

        ext_module.general_deform_roi_pool_backward(
            grad_output,
            input,
            rois,
            rescale,
            offset,
            grad_input,
            grad_rescale,
            grad_offset,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            gamma=ctx.gamma,
            sigma=ctx.sigma,
            beta=ctx.beta)
        if grad_rescale.numel() == 0:
            grad_rescale = None
        if grad_offset.numel() == 0:
            grad_offset = None
        return grad_input, None, grad_rescale, grad_offset, None, None, None, None, None, None


general_deform_roi_pool = GeneralDeformRoIPoolFunction.apply


class GeneralDeformRoIPool(nn.Module):

    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1,
                 sigma=0.5,
                 beta=0.0):
        super(GeneralDeformRoIPool, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.beta = float(beta)

    def forward(self, input, rois, rescale=None, offset=None):
        return general_deform_roi_pool(input, rois, rescale, offset, self.output_size,
                                       self.spatial_scale, self.sampling_ratio,
                                       self.gamma, self.sigma, self.beta)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GeneralDeformRoIPoolPack(GeneralDeformRoIPool):

    def __init__(self,
                 output_size,
                 output_channels,
                 deform_fc_channels=1024,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1,
                 sigma=0.5,
                 beta=0.0):
        super(GeneralDeformRoIPoolPack, self).__init__(output_size, spatial_scale,
                                                       sampling_ratio, gamma, sigma, beta)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels
        self.transformer_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.output_size[0] * self.output_size[1] * 2))
        self.transformer_fc[-1].weight.data.zero_()
        self.transformer_fc[-1].bias.data.zero_()
        self.transformer_fc_cls = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.output_size[0] * self.output_size[1] * 4))
        self.transformer_fc_cls[-1].weight.data.zero_()
        self.transformer_fc_cls[-1].bias.data.zero_()
        self.transformer_fc_reg = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.output_size[0] * self.output_size[1] * 4))
        self.transformer_fc_reg[-1].weight.data.zero_()
        self.transformer_fc_reg[-1].bias.data.zero_()
        self.tanh = nn.Tanh()
        self.senets = nn.ModuleList([SELayer(channel=self.output_channels) for i in range(2)])

    def forward(self, input, rois):
        assert input.size(1) == self.output_channels
        x = general_deform_roi_pool(input, rois, None, None, self.output_size,
                                    self.spatial_scale, self.sampling_ratio,
                                    self.gamma, self.sigma, self.beta)
        rois_num = rois.size(0)
        offset = self.transformer_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0], self.output_size[1])
        x_dpool = general_deform_roi_pool(input, rois, None, offset, self.output_size,
                                          self.spatial_scale, self.sampling_ratio,
                                          self.gamma, self.sigma, self.beta)
        x_dpool_cls = x_dpool + self.senets[0](x_dpool)
        x_dpool_reg = x_dpool + self.senets[1](x_dpool)
        transformer_factor_cls = self.transformer_fc_cls(x_dpool_cls.view(rois_num, -1))
        transformer_factor_cls = transformer_factor_cls.view(
            rois_num, 4, self.output_size[0], self.output_size[1])
        transformer_factor_cls = torch.split(transformer_factor_cls, 2, dim=1)
        rescale_cls = transformer_factor_cls[0].contiguous()
        rescale_cls = self.tanh(rescale_cls)
        offset_cls = offset + transformer_factor_cls[1].contiguous()
        transformer_factor_reg = self.transformer_fc_reg(x_dpool_reg.view(rois_num, -1))
        transformer_factor_reg = transformer_factor_reg.view(
            rois_num, 4, self.output_size[0], self.output_size[1])
        transformer_factor_reg = torch.split(transformer_factor_reg, 2, dim=1)
        rescale_reg = transformer_factor_reg[0].contiguous()
        rescale_reg = self.tanh(rescale_reg)
        offset_reg = offset + transformer_factor_reg[1].contiguous()

        x_cls = general_deform_roi_pool(input, rois, rescale_cls, offset_cls, self.output_size,
                                        self.spatial_scale, self.sampling_ratio,
                                        self.gamma, self.sigma, self.beta)
        x_reg = general_deform_roi_pool(input, rois, rescale_reg, offset_reg, self.output_size,
                                        self.spatial_scale, self.sampling_ratio,
                                        self.gamma, self.sigma, self.beta)

        return x_cls, x_reg



## Copyright (c) OpenMMLab. All rights reserved.
#import torch
#from torch import nn
#from torch.autograd import Function
#from torch.autograd.function import once_differentiable
#from torch.nn.modules.utils import _pair
#from mmcv.cnn import ConvModule
#import numpy as np
#from ..utils import ext_loader
#
#ext_module = ext_loader.load_ext(
#    '_ext', ['general_deform_roi_pool_forward', 'general_deform_roi_pool_backward'])
#
#
#class GeneralDeformRoIPoolFunction(Function):
#
#    @staticmethod
#    def symbolic(g, input, rois, rescale, offset, output_size, spatial_scale,
#                 sampling_ratio, gamma, sigma, beta):
#        return g.op(
#            'mmcv::MMCVGeneralDeformRoIPool',
#            input,
#            rois,
#            rescale,
#            offset,
#            pooled_height_i=output_size[0],
#            pooled_width_i=output_size[1],
#            spatial_scale_f=spatial_scale,
#            sampling_ratio_f=sampling_ratio,
#            gamma_f=gamma,
#            sigma_f=sigma,
#            beta_f=beta)
#
#    @staticmethod
#    def forward(ctx,
#                input,
#                rois,
#                rescale,
#                offset,
#                output_size,
#                spatial_scale=1.0,
#                sampling_ratio=0,
#                gamma=0.1,
#                sigma=0.5,
#                beta=0.0):
#        if rescale is None:
#            rescale = input.new_zeros(0)
#        if offset is None:
#            offset = input.new_zeros(0)
#        ctx.output_size = _pair(output_size)
#        ctx.spatial_scale = float(spatial_scale)
#        ctx.sampling_ratio = int(sampling_ratio)
#        ctx.gamma = float(gamma)
#        ctx.sigma = float(sigma)
#        ctx.beta = float(beta)
#
#        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'
#
#        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
#                        ctx.output_size[1])
#        output = input.new_zeros(output_shape)
#
#        ext_module.general_deform_roi_pool_forward(
#            input,
#            rois,
#            rescale,
#            offset,
#            output,
#            pooled_height=ctx.output_size[0],
#            pooled_width=ctx.output_size[1],
#            spatial_scale=ctx.spatial_scale,
#            sampling_ratio=ctx.sampling_ratio,
#            gamma=ctx.gamma,
#            sigma=ctx.sigma,
#            beta=ctx.beta)
#
#        ctx.save_for_backward(input, rois, rescale, offset)
#        return output
#
#    @staticmethod
#    @once_differentiable
#    def backward(ctx, grad_output):
#        input, rois, rescale, offset = ctx.saved_tensors
#        grad_input = grad_output.new_zeros(input.shape)
#        grad_rescale = grad_output.new_zeros(rescale.shape)
#        grad_offset = grad_output.new_zeros(offset.shape)
#
#        ext_module.general_deform_roi_pool_backward(
#            grad_output,
#            input,
#            rois,
#            rescale,
#            offset,
#            grad_input,
#            grad_rescale,
#            grad_offset,
#            pooled_height=ctx.output_size[0],
#            pooled_width=ctx.output_size[1],
#            spatial_scale=ctx.spatial_scale,
#            sampling_ratio=ctx.sampling_ratio,
#            gamma=ctx.gamma,
#            sigma=ctx.sigma,
#            beta=ctx.beta)
#        if grad_rescale.numel() == 0:
#            grad_rescale = None
#        if grad_offset.numel() == 0:
#            grad_offset = None
#        return grad_input, None, grad_rescale, grad_offset, None, None, None, None, None, None
#
#
#general_deform_roi_pool = GeneralDeformRoIPoolFunction.apply
#
#
#class GeneralDeformRoIPool(nn.Module):
#
#    def __init__(self,
#                 output_size,
#                 spatial_scale=1.0,
#                 sampling_ratio=0,
#                 gamma=0.1,
#                 sigma=0.5,
#                 beta=0.0):
#        super(GeneralDeformRoIPool, self).__init__()
#        self.output_size = _pair(output_size)
#        self.spatial_scale = float(spatial_scale)
#        self.sampling_ratio = int(sampling_ratio)
#        self.gamma = float(gamma)
#        self.sigma = float(sigma)
#        self.beta = float(beta)
#
#    def forward(self, input, rois, rescale=None, offset=None):
#        return general_deform_roi_pool(input, rois, rescale, offset, self.output_size,
#                                       self.spatial_scale, self.sampling_ratio,
#                                       self.gamma, self.sigma, self.beta)
#
#
#class SELayer(nn.Module):
#    def __init__(self, channel, reduction=16):
#        super(SELayer, self).__init__()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.fc = nn.Sequential(
#            nn.Linear(channel, channel // reduction, bias=False),
#            nn.ReLU(inplace=True),
#            nn.Linear(channel // reduction, channel, bias=False),
#            nn.Sigmoid())
#
#    def forward(self, x):
#        b, c, _, _ = x.size()
#        y = self.avg_pool(x).view(b, c)
#        y = self.fc(y).view(b, c, 1, 1)
#        return x * y.expand_as(x)
#
#
#class GeneralDeformRoIPoolPack(GeneralDeformRoIPool):
#
#    def __init__(self,
#                 output_size,
#                 output_channels,
#                 deform_fc_channels=1024,
#                 spatial_scale=1.0,
#                 sampling_ratio=0,
#                 gamma=0.1,
#                 sigma=0.5,
#                 beta=0.0):
#        super(GeneralDeformRoIPoolPack, self).__init__(output_size, spatial_scale,
#                                                       sampling_ratio, gamma, sigma, beta)
#
#        self.output_channels = output_channels
#        self.deform_fc_channels = deform_fc_channels
#        self.transformer_cls_fc = nn.Sequential(
#            nn.Linear(
#                self.output_size[0] * self.output_size[1] *
#                self.output_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels, self.output_size[0] * self.output_size[1] * 4))
#        self.transformer_cls_fc[-1].weight.data.zero_()
#        self.transformer_cls_fc[-1].bias.data.zero_()
#        self.transformer_reg_fc = nn.Sequential(
#            nn.Linear(
#                self.output_size[0] * self.output_size[1] *
#                self.output_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels, self.output_size[0] * self.output_size[1] * 4))
#        self.transformer_reg_fc[-1].weight.data.zero_()
#        self.transformer_reg_fc[-1].bias.data.zero_()
#        self.tanh_cls = nn.Tanh()
#        self.tanh_reg = nn.Tanh()
#        self.se_cls = SELayer(channel=self.output_channels)
#        self.se_reg = SELayer(channel=self.output_channels)
#
#    def forward(self, input, rois):
#        assert input.size(1) == self.output_channels
#        cx = (rois[:, 1] + rois[:, 3]) * 0.5
#        cy = (rois[:, 2] + rois[:, 4]) * 0.5
#        w = rois[:, 3] - rois[:, 1]
#        h = rois[:, 4] - rois[:, 2]
#        new_w = w * 1.3
#        new_h = h * 1.3
#        x1 = cx - new_w * 0.5
#        x2 = cx + new_w * 0.5
#        y1 = cy - new_h * 0.5
#        y2 = cy + new_h * 0.5
#        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
#        x = general_deform_roi_pool(input, rois, None, None, self.output_size,
#                                    self.spatial_scale, self.sampling_ratio,
#                                    self.gamma, self.sigma, self.beta)
#        x_se_cls = x + self.se_cls(x)
#        x_ = general_deform_roi_pool(input, new_rois, None, None, self.output_size,
#                                     self.spatial_scale, self.sampling_ratio,
#                                     self.gamma, self.sigma, self.beta)
#        x_se_reg = x_ + self.se_reg(x_)
#
#        rois_num = rois.size(0)
#        transformer_factor_cls = self.transformer_cls_fc(x_se_cls.view(rois_num, -1))
#        transformer_factor_reg = self.transformer_reg_fc(x_se_reg.view(rois_num, -1))
#        rescale_cls = transformer_factor_cls[:, :self.output_size[0] * self.output_size[1] * 2].contiguous()
#        rescale_cls = self.tanh_cls(rescale_cls)
#        offset_cls = transformer_factor_cls[:, self.output_size[0] * self.output_size[1] * 2:].contiguous()
#        rescale_reg = transformer_factor_reg[:, :self.output_size[0] * self.output_size[1] * 2].contiguous()
#        rescale_reg = self.tanh_reg(rescale_reg)
#        offset_reg = transformer_factor_reg[:, self.output_size[0] * self.output_size[1] * 2:].contiguous()
#
#        rescale_cls = rescale_cls.view(rois_num, 2, self.output_size[0], self.output_size[1])
#        offset_cls = offset_cls.view(rois_num, 2, self.output_size[0], self.output_size[1])
#        rescale_reg = rescale_reg.view(rois_num, 2, self.output_size[0], self.output_size[1])
#        offset_reg = offset_reg.view(rois_num, 2, self.output_size[0], self.output_size[1])
#        x_cls = general_deform_roi_pool(input, rois, rescale_cls, offset_cls, self.output_size,
#                                        self.spatial_scale, self.sampling_ratio,
#                                        self.gamma, self.sigma, self.beta)
#        x_reg = general_deform_roi_pool(input, new_rois, rescale_reg, offset_reg, self.output_size,
#                                        self.spatial_scale, self.sampling_ratio,
#                                        self.gamma, self.sigma, self.beta)
#        return x_cls, x_reg