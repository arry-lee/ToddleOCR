import torch.nn as nn

__all__ = ['ResNet_vd']

from ptocr.modules.backbones.resnet.det_resnet import ResNet
from ptocr.ops import ConvBNLayer


# class ResNet_vd(nn.Module):
#
#     def __init__(self, in_channels=3, layers=50, dcn_stage=None, out_indices=None, **kwargs):
#         super().__init__()
#         supported_layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 12, 48, 3]}
#         depth = supported_layers[layers]
#         num_features = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
#         num_filters = [64, 128, 256, 512]
#         self.dcn_stage = dcn_stage or [False] * 4
#         self.out_indices = out_indices or [0, 1, 2, 3]
#         self.conv = nn.Sequential(
#             ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act='relu'),
#             ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu'),
#             ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu'),
#         )
#         self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.stages = []
#         self.out_channels = []
#         block_class = BottleneckBlock if layers >= 50 else BasicBlock
#         for block in range(len(depth)):
#             block_list = []
#             is_dcn = self.dcn_stage[block]
#             downsample = ConvBNLayer(in_channels=in_channels, out_channels=num_filters[block], kernel_size=1, stride=1, is_vd_mode=True)
#             for i in range(depth[block]):
#                 if block == i == 0:
#                     downsample = None
#                 _block = block_class(in_channels=num_features[block] if i == 0 else num_filters[block] * block_class.expansion, out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, downsample=downsample, is_dcn=is_dcn)
#                 self.add_module('bb_%d_%d' % (block, i), _block)
#                 block_list.append(_block)
#             if block in self.out_indices:
#                 self.out_channels.append(num_filters[block] * block_class.expansion)
#             self.stages.append(nn.Sequential(*block_list))
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pool2d_max(x)
#         out = []
#         for (i, block) in enumerate(self.stages):
#             x = block(x)
#             if i in self.out_indices:
#                 out.append(x)
#         return out

class ResNet_vd(ResNet):
    def convs(self):
        return nn.Sequential(
            ConvBNLayer(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=2, act='relu'),
            ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu'),
            ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu'),
        )
