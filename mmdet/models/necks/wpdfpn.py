import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16


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


class SFE(nn.Module):
    def __init__(self, in_channels, compressed_channels):
        super(SFE, self).__init__()
        self.senet = SELayer(channel=in_channels)
        self.encoder = ConvModule(
            in_channels,
            compressed_channels,
            3,
            padding=1,
            dilation=1,
            conv_cfg=dict(type='DCNv2', deform_groups=1),
            norm_cfg=None,
            act_cfg=dict(type='ReLU'))
        self.w_e = nn.Sequential(
            nn.Conv2d(
                compressed_channels,
                in_channels,
                3,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid())
        self.w_f = nn.Sequential(
            nn.Conv2d(
                compressed_channels,
                in_channels,
                3,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid())

    def init_weights(self):
        """Initialize the weights of SFE module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x, y):
        assert x.size() == y.size()
        base_f = x + y
        base_f = base_f + self.senet(base_f)
        base_f = self.encoder(base_f)
        w_e = self.w_e(base_f)
        w_f = self.w_f(base_f)
        return w_e * x + w_f * y


class Weighted_Path_Aggregation(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(Weighted_Path_Aggregation, self).__init__()
        pad = int((kernel_size - 1) * dilation / 2)
        self.td_weights = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            conv_cfg=dict(type='DCNv2', deform_groups=2),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='Sigmoid'),
            inplace=False)
        self.bt_weights = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            conv_cfg=dict(type='DCNv2', deform_groups=2),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='Sigmoid'),
            inplace=False)

    def init_weights(self):
        """Initialize the weights of WPA module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x, y):
        assert x.size() == y.size()
        base = x + y
        td_w = self.td_weights(base)
        bt_w = self.bt_weights(base)
        enhanced_x = td_w * x + (1 - td_w) * y
        enhanced_y = bt_w * x + (1 - bt_w) * y
        return enhanced_x, enhanced_y


from ..builder import NECKS


@NECKS.register_module()
class WPDFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(WPDFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        self.td_sfes = nn.ModuleList()
        self.bu_sfes = nn.ModuleList()
        self.wpas = nn.ModuleList()

        self.td_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.bu_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                padding=0,
                dilation=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            wpa = Weighted_Path_Aggregation(
                in_channels=out_channels,
                kernel_size=3,
                dilation=3)
            if i != self.backbone_end_level - 2:
                d_convs = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                td_sfe = SFE(in_channels=out_channels,
                             compressed_channels=out_channels)
                bu_sfe = SFE(in_channels=out_channels,
                             compressed_channels=out_channels)
                self.downsample_convs.append(d_convs)
                self.td_sfes.append(td_sfe)
                self.bu_sfes.append(bu_sfe)

            self.lateral_convs.append(l_conv)
            self.wpas.append(wpa)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_fpn_convs_td = nn.ModuleList()
        self.extra_fpn_convs_bu = nn.ModuleList()
        for i in range(extra_levels):
            extra_fpn_conv_td = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            extra_fpn_conv_bu = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.extra_fpn_convs_td.append(extra_fpn_conv_td)
            self.extra_fpn_convs_bu.append(extra_fpn_conv_bu)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of PFPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        td_laterals = laterals
        bu_laterals = laterals
        used_backbone_levels = len(td_laterals)
        # build top-down path (semantic enhancement branch)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = td_laterals[i - 1].shape[2:]
            td_laterals[i - 1] = self.td_sfes[i - 1](
                td_laterals[i - 1], F.interpolate(
                    td_laterals[i], size=prev_shape, **self.upsample_cfg))
        # bulid top-down outputs
        td_laterals[-1] = self.td_conv(td_laterals[-1])

        # build bottom-up path (detail enhancement branch)
        for i in range(0, used_backbone_levels - 1):
            bu_laterals[i + 1] = self.bu_sfes[i](
                bu_laterals[i + 1], self.downsample_convs[i](bu_laterals[i]))

        bu_laterals[0] = self.bu_conv(bu_laterals[0])
        td_outs = []
        bu_outs = []

        for i in range(used_backbone_levels):
            td_out, bu_out = self.wpas[i](td_laterals[i], bu_laterals[i])
            td_outs.append(td_out)
            bu_outs.append(bu_out)
        if self.num_outs > len(td_outs):
            for i in range(0, len(self.extra_fpn_convs_td)):
                td_outs.append(self.extra_fpn_convs_td[i](F.relu(td_outs[-1])))
                bu_outs.append(self.extra_fpn_convs_bu[i](F.relu(bu_outs[-1])))

        return tuple(td_outs), tuple(bu_outs)