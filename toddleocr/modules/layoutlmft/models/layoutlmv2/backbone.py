import math
from abc import abstractmethod, ABCMeta
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def c2_xavier_fill(module: nn.Module) -> None:
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

def c2_msra_fill(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width', 'stride'])):

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)

class FrozenBatchNorm2d(nn.Module):
    _version = 3

    def __init__(self, num_features, eps=1e-05):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = torch.zeros_like(self.running_mean)
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = torch.ones_like(self.running_var)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for (name, child) in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': nn.BatchNorm2d, 'SyncBN': nn.SyncBatchNorm, 'FrozenBN': FrozenBatchNorm2d, 'GN': lambda channels: nn.GroupNorm(32, channels)}[norm]
    return norm(out_channels)

class CNNBlockBase(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

# 这里有点脱裤子放屁了
class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(self.norm, torch.nn.SyncBatchNorm), 'SyncBatchNorm does not support empty inputs!'
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def _pair(kernel_size):
    return (kernel_size, kernel_size)

class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return (_NewEmptyTensorOp.apply(grad, shape), None)

class Backbone(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self) -> int:
        return 0

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

class BasicBlock(CNNBlockBase):

    def __init__(self, in_channels, out_channels, *, stride=1, norm='BN'):
        super().__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, norm=get_norm(norm, out_channels))
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, out_channels))
        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:
                c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out

class BottleneckBlock(CNNBlockBase):

    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN', stride_in_1x1=False, dilation=1):
        super().__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        (stride_1x1, stride_3x3) = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, bottleneck_channels))
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, norm=get_norm(norm, bottleneck_channels))
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out

class BasicStem(CNNBlockBase):

    def __init__(self, in_channels=3, out_channels=64, norm='BN'):
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, out_channels))
        c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class ResNet(Backbone):

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes
        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}
        (self.stage_names, self.stages) = ([], [])
        if out_features is not None:
            num_stages = max([{'res2': 1, 'res3': 2, 'res4': 3, 'res5': 4}.get(f, 0) for f in out_features])
            stages = stages[:num_stages]
        for (i, blocks) in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block
            name = 'res' + str(i + 2)
            stage = nn.Sequential(*blocks)
            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = 'linear'
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, 'Available children: {}'.format(', '.join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        assert x.dim() == 4, f'ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!'
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for (name, stage) in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stem.freeze()
        for (idx, stage) in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for (k, v) in kwargs.items():
                if k.endswith('_per_block'):
                    assert len(v) == num_blocks, f"Argument '{k}' of make_stage should have the same length as num_blocks={num_blocks}."
                    newk = k[:-len('_per_block')]
                    assert newk not in kwargs, f'Cannot call make_stage with both {k} and {newk}!'
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v
            blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs))
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        num_blocks_per_stage = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs['bottleneck_channels'] = o // 4
            ret.append(ResNet.make_stage(block_class=block_class, num_blocks=n, stride_per_block=[s] + [1] * (n - 1), in_channels=i, out_channels=o, **kwargs))
        return ret
ResNetBlockBase = CNNBlockBase
'\nAlias for backward compatibiltiy.\n'

def _assert_strides_are_log2_contiguous(strides):
    for (i, stride) in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], 'Strides {} {} are not log2 contiguous'.format(stride, strides[i - 1])

class FPN(Backbone):
    _fuse_type: torch.jit.Final[str]

    def __init__(self, bottom_up, in_features, out_channels, norm='', top_block=None, fuse_type='sum'):
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []
        use_bias = norm == ''
        for (idx, in_channels) in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)
            lateral_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm) # fixme
            output_conv = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module('fpn_lateral{}'.format(stage), lateral_conv)
            self.add_module('fpn_output{}'.format(stage), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        self._out_feature_strides = {'p{}'.format(int(math.log2(s))): s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides['p{}'.format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {'avg', 'sum'}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))
        for (idx, (lateral_conv, output_conv)) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == 'avg':
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))
        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for (f, res) in zip(self._out_features, results)}

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

class LastLevelMaxPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = 'p5'

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

def build_resnet_fpn_backbone(norm='FrozenBN'):
    bottom_up = build_resnet_backbone()
    in_features = ['res2', 'res3', 'res4', 'res5']
    out_channels = 256
    backbone = FPN(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, norm=norm, top_block=LastLevelMaxPool(), fuse_type='sum')
    return backbone

def build_resnet_backbone():
    norm = 'FrozenBN'
    stem = BasicStem(in_channels=3, out_channels=64, norm=norm)
    freeze_at = 2
    out_features = ['res2', 'res3', 'res4', 'res5']
    depth = 101
    num_groups = 32
    width_per_group = 8
    bottleneck_channels = num_groups * width_per_group
    in_channels = 64
    out_channels = 256
    stride_in_1x1 = False
    res5_dilation = 1
    deform_on_per_stage = [False, False, False, False]
    assert res5_dilation in {1, 2}, 'res5_dilation cannot be {}.'.format(res5_dilation)
    num_blocks_per_stage = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
    if depth in [18, 34]:
        assert out_channels == 64, 'Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34'
        assert not any(deform_on_per_stage), 'MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34'
        assert res5_dilation == 1, 'Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34'
        assert num_groups == 1, 'Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34'
    stages = []
    for (idx, stage_idx) in enumerate(range(2, 6)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {'num_blocks': num_blocks_per_stage[idx], 'stride_per_block': [first_stride] + [1] * (num_blocks_per_stage[idx] - 1), 'in_channels': in_channels, 'out_channels': out_channels, 'norm': norm}
        if depth in [18, 34]:
            stage_kargs['block_class'] = BasicBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['stride_in_1x1'] = stride_in_1x1
            stage_kargs['dilation'] = dilation
            stage_kargs['num_groups'] = num_groups
            if deform_on_per_stage[idx]:
                raise ValueError('not DeformBottleneckBlock')
            else:
                stage_kargs['block_class'] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
