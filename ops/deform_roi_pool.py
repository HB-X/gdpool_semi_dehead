# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['deform_roi_pool_forward', 'deform_roi_pool_backward'])


class DeformRoIPoolFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, offset, output_size, spatial_scale,
                 sampling_ratio, gamma):
        return g.op(
            'mmcv::MMCVDeformRoIPool',
            input,
            rois,
            offset,
            pooled_height_i=output_size[0],
            pooled_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_f=sampling_ratio,
            gamma_f=gamma)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                offset,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                gamma=0.1):
        if offset is None:
            offset = input.new_zeros(0)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = float(spatial_scale)
        ctx.sampling_ratio = int(sampling_ratio)
        ctx.gamma = float(gamma)

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)

        ext_module.deform_roi_pool_forward(
            input,
            rois,
            offset,
            output,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            gamma=ctx.gamma)

        ctx.save_for_backward(input, rois, offset)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset = ctx.saved_tensors
        grad_input = grad_output.new_zeros(input.shape)
        grad_offset = grad_output.new_zeros(offset.shape)

        ext_module.deform_roi_pool_backward(
            grad_output,
            input,
            rois,
            offset,
            grad_input,
            grad_offset,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            gamma=ctx.gamma)
        if grad_offset.numel() == 0:
            grad_offset = None
        return grad_input, None, grad_offset, None, None, None, None


deform_roi_pool = DeformRoIPoolFunction.apply


class DeformRoIPool(nn.Module):

    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1):
        super(DeformRoIPool, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.gamma = float(gamma)

    def forward(self, input, rois, offset=None):
        return deform_roi_pool(input, rois, offset, self.output_size,
                               self.spatial_scale, self.sampling_ratio,
                               self.gamma)


class DeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size,
                 output_channels,
                 deform_fc_channels=1024,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1):
        super(DeformRoIPoolPack, self).__init__(output_size, spatial_scale,
                                                sampling_ratio, gamma)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels

        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

    def forward(self, input, rois):
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        rois_num = rois.size(0)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0],
                             self.output_size[1])
        return deform_roi_pool(input, rois, offset, self.output_size,
                               self.spatial_scale, self.sampling_ratio,
                               self.gamma)

class GeDeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size,
                 output_channels,
                 deform_fc_channels=1024,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1,
                 sigma=0.5):
        super(GeDeformRoIPoolPack, self).__init__(output_size, spatial_scale,
                                                  sampling_ratio, gamma)
        self.sigma = sigma
        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels
        self.rescale_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.output_size[0] * self.output_size[1] * 2),
            nn.Sigmoid())
        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

    def rescale_roi(self, rois, scale_factor):
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
        new_cx = cx + scale_factor[:, 0]
        new_cy = cy + scale_factor[:, 1]
        new_w = w * scale_factor[:, 2]
        new_h = h * scale_factor[:, 3]
        x1 = new_cx - new_w * 0.5
        x2 = new_cx + new_w * 0.5
        y1 = new_cy - new_h * 0.5
        y2 = new_cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    def forward(self, input, rois):
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        x_cls = x
        rois_num = rois.size(0)
        rescale = self.rescale_fc(x.view(rois_num, -1))
        rescale = 1.0 + self.sigma * rescale
        new_rois = self.rescale_roi(rois, rescale)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0], self.output_size[1])
        x_reg = deform_roi_pool(input, new_rois, offset, self.output_size, 
                                self.spatial_scale, self.sampling_ratio, self.gamma)

        return x_cls, x_reg


class ModulatedDeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size,
                 output_channels,
                 deform_fc_channels=1024,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1):
        super(ModulatedDeformRoIPoolPack,
              self).__init__(output_size, spatial_scale, sampling_ratio, gamma)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels

        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

        self.mask_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 1),
            nn.Sigmoid())
        self.mask_fc[2].weight.data.zero_()
        self.mask_fc[2].bias.data.zero_()

    def forward(self, input, rois):
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        rois_num = rois.size(0)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0],
                             self.output_size[1])
        mask = self.mask_fc(x.view(rois_num, -1))
        mask = mask.view(rois_num, 1, self.output_size[0], self.output_size[1])
        d = deform_roi_pool(input, rois, offset, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        return d * mask




## Copyright (c) OpenMMLab. All rights reserved.
#from torch import nn
#from torch.autograd import Function
#from torch.autograd.function import once_differentiable
#from torch.nn.modules.utils import _pair
#
#from ..utils import ext_loader
#
#ext_module = ext_loader.load_ext(
#    '_ext', ['deform_roi_pool_forward', 'deform_roi_pool_backward'])
#
#
#class DeformRoIPoolFunction(Function):
#
#    @staticmethod
#    def symbolic(g, input, rois, offset, output_size, spatial_scale,
#                 sampling_ratio, gamma):
#        return g.op(
#            'mmcv::MMCVDeformRoIPool',
#            input,
#            rois,
#            offset,
#            pooled_height_i=output_size[0],
#            pooled_width_i=output_size[1],
#            spatial_scale_f=spatial_scale,
#            sampling_ratio_f=sampling_ratio,
#            gamma_f=gamma)
#
#    @staticmethod
#    def forward(ctx,
#                input,
#                rois,
#                offset,
#                output_size,
#                spatial_scale=1.0,
#                sampling_ratio=0,
#                gamma=0.1):
#        if offset is None:
#            offset = input.new_zeros(0)
#        ctx.output_size = _pair(output_size)
#        ctx.spatial_scale = float(spatial_scale)
#        ctx.sampling_ratio = int(sampling_ratio)
#        ctx.gamma = float(gamma)
#
#        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'
#
#        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
#                        ctx.output_size[1])
#        output = input.new_zeros(output_shape)
#
#        ext_module.deform_roi_pool_forward(
#            input,
#            rois,
#            offset,
#            output,
#            pooled_height=ctx.output_size[0],
#            pooled_width=ctx.output_size[1],
#            spatial_scale=ctx.spatial_scale,
#            sampling_ratio=ctx.sampling_ratio,
#            gamma=ctx.gamma)
#
#        ctx.save_for_backward(input, rois, offset)
#        return output
#
#    @staticmethod
#    @once_differentiable
#    def backward(ctx, grad_output):
#        input, rois, offset = ctx.saved_tensors
#        grad_input = grad_output.new_zeros(input.shape)
#        grad_offset = grad_output.new_zeros(offset.shape)
#
#        ext_module.deform_roi_pool_backward(
#            grad_output,
#            input,
#            rois,
#            offset,
#            grad_input,
#            grad_offset,
#            pooled_height=ctx.output_size[0],
#            pooled_width=ctx.output_size[1],
#            spatial_scale=ctx.spatial_scale,
#            sampling_ratio=ctx.sampling_ratio,
#            gamma=ctx.gamma)
#        if grad_offset.numel() == 0:
#            grad_offset = None
#        return grad_input, None, grad_offset, None, None, None, None
#
#
#deform_roi_pool = DeformRoIPoolFunction.apply
#
#
#class DeformRoIPool(nn.Module):
#
#    def __init__(self,
#                 output_size,
#                 spatial_scale=1.0,
#                 sampling_ratio=0,
#                 gamma=0.1):
#        super(DeformRoIPool, self).__init__()
#        self.output_size = _pair(output_size)
#        self.spatial_scale = float(spatial_scale)
#        self.sampling_ratio = int(sampling_ratio)
#        self.gamma = float(gamma)
#
#    def forward(self, input, rois, offset=None):
#        return deform_roi_pool(input, rois, offset, self.output_size,
#                               self.spatial_scale, self.sampling_ratio,
#                               self.gamma)
#
#
#class DeformRoIPoolPack(DeformRoIPool):
#
#    def __init__(self,
#                 output_size,
#                 output_channels,
#                 deform_fc_channels=1024,
#                 spatial_scale=1.0,
#                 sampling_ratio=0,
#                 gamma=0.1):
#        super(DeformRoIPoolPack, self).__init__(output_size, spatial_scale,
#                                                sampling_ratio, gamma)
#
#        self.output_channels = output_channels
#        self.deform_fc_channels = deform_fc_channels
#
#        self.offset_fc = nn.Sequential(
#            nn.Linear(
#                self.output_size[0] * self.output_size[1] *
#                self.output_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels,
#                      self.output_size[0] * self.output_size[1] * 2))
#        self.offset_fc[-1].weight.data.zero_()
#        self.offset_fc[-1].bias.data.zero_()
#
#    def forward(self, input, rois):
#        assert input.size(1) == self.output_channels
#        x = deform_roi_pool(input, rois, None, self.output_size,
#                            self.spatial_scale, self.sampling_ratio,
#                            self.gamma)
#        rois_num = rois.size(0)
#        offset = self.offset_fc(x.view(rois_num, -1))
#        offset = offset.view(rois_num, 2, self.output_size[0],
#                             self.output_size[1])
#        return deform_roi_pool(input, rois, offset, self.output_size,
#                               self.spatial_scale, self.sampling_ratio,
#                               self.gamma)
#
#
#class ModulatedDeformRoIPoolPack(DeformRoIPool):
#
#    def __init__(self,
#                 output_size,
#                 output_channels,
#                 deform_fc_channels=1024,
#                 spatial_scale=1.0,
#                 sampling_ratio=0,
#                 gamma=0.1):
#        super(ModulatedDeformRoIPoolPack,
#              self).__init__(output_size, spatial_scale, sampling_ratio, gamma)
#
#        self.output_channels = output_channels
#        self.deform_fc_channels = deform_fc_channels
#
#        self.offset_fc = nn.Sequential(
#            nn.Linear(
#                self.output_size[0] * self.output_size[1] *
#                self.output_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels,
#                      self.output_size[0] * self.output_size[1] * 2))
#        self.offset_fc[-1].weight.data.zero_()
#        self.offset_fc[-1].bias.data.zero_()
#
#        self.mask_fc = nn.Sequential(
#            nn.Linear(
#                self.output_size[0] * self.output_size[1] *
#                self.output_channels, self.deform_fc_channels),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.deform_fc_channels,
#                      self.output_size[0] * self.output_size[1] * 1),
#            nn.Sigmoid())
#        self.mask_fc[2].weight.data.zero_()
#        self.mask_fc[2].bias.data.zero_()
#
#    def forward(self, input, rois):
#        assert input.size(1) == self.output_channels
#        x = deform_roi_pool(input, rois, None, self.output_size,
#                            self.spatial_scale, self.sampling_ratio,
#                            self.gamma)
#        rois_num = rois.size(0)
#        offset = self.offset_fc(x.view(rois_num, -1))
#        offset = offset.view(rois_num, 2, self.output_size[0],
#                             self.output_size[1])
#        mask = self.mask_fc(x.view(rois_num, -1))
#        mask = mask.view(rois_num, 1, self.output_size[0], self.output_size[1])
#        d = deform_roi_pool(input, rois, offset, self.output_size,
#                            self.spatial_scale, self.sampling_ratio,
#                            self.gamma)
#        return d * mask
