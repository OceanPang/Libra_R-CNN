import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from .fpn import FPN
from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class PAFPN(FPN):

    def __init__(self, *args, **kwargs):
        super(PAFPN, self).__init__(*args, **kwargs)

        self.stride_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            stride_conv = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                stride=2,
                padding=1,
                normalize=self.normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            pafpn_conv = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                normalize=self.normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            if i < self.backbone_end_level - 1:
                self.stride_convs.append(stride_conv)
            self.pafpn_convs.append(pafpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add pafpn
        for i in range(used_backbone_levels - 1):
            outs[i + 1] += self.stride_convs[i](outs[i])

        paouts = [
            self.pafpn_convs[i](outs[i]) for i in range(used_backbone_levels)
        ]

        # part 3: add extra levels
        if self.num_outs > len(paouts):
            # use max pool to get more levels on top of outputs
            # e.g. (Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    paouts.append(F.max_pool2d(paouts[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    paouts.append(self.extra_convs[0](orig))
                else:
                    paouts.append(self.extra_convs[0](paouts[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    paouts.append(self.extra_convs[i - used_backbone_levels](
                        paouts[-1]))
        return tuple(paouts)
