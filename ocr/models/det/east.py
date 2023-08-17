import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super().__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            affine=True,
            track_running_stats=True)

        self.name = name

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.if_act and self.act is not None:
            if self.act == 'relu':
                x = nn.ReLU()(x)
            elif self.act == 'sigmoid':
                x = nn.Sigmoid()(x)
            elif self.act == 'tanh':
                x = nn.Tanh()(x)
            elif self.act == 'leaky_relu':
                x = nn.LeakyReLU()(x)
            elif self.act == 'elu':
                x = nn.ELU()(x)
            elif self.act == 'hardswish':
                x = nn.Hardswish()(x)
            else:
                raise NotImplementedError(f"Activation function {self.act} is not supported.")

        return x

class DeConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super().__init__()
        self.if_act = if_act
        self.act = act
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            affine=True,
            track_running_stats=True)

        self.name = name

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)

        if self.if_act and self.act is not None:
            if self.act == 'relu':
                x = nn.ReLU()(x)
            elif self.act == 'sigmoid':
                x = nn.Sigmoid()(x)
            elif self.act == 'tanh':
                x = nn.Tanh()(x)
            elif self.act == 'leaky_relu':
                x = nn.LeakyReLU()(x)
            elif self.act == 'elu':
                x = nn.ELU()(x)
            else:
                raise NotImplementedError(f"Activation function {self.act} is not supported.")

        return x

class EASTFPN(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "large":
            self.out_channels = 128
        else:
            self.out_channels = 64
        self.in_channels = in_channels[::-1]
        self.h1_conv = ConvBNLayer(
            in_channels=self.out_channels + self.in_channels[1],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_1")
        self.h2_conv = ConvBNLayer(
            in_channels=self.out_channels + self.in_channels[2],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_2")
        self.h3_conv = ConvBNLayer(
            in_channels=self.out_channels + self.in_channels[3],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_3")
        self.g0_deconv = DeConvBNLayer(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_0")
        self.g1_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_1")
        self.g2_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_2")
        self.g3_conv = ConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_3")

    def forward(self, x):
        f = x[::-1]

        h = f[0]
        g = self.g0_deconv(h)
        h = torch.cat([g, f[1]], dim=1)
        h = self.h1_conv(h)
        g = self.g1_deconv(h)
        h = torch.cat([g, f[2]], dim=1)
        h = self.h2_conv(h)
        g = self.g2_deconv(h)
        h = torch.cat([g, f[3]], dim=1)
        h = self.h3_conv(h)
        g = self.g3_conv(h)

        return g

class EASTHead(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]

        self.det_conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="det_head1")
        self.det_conv2 = ConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="det_head2")
        self.score_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name="f_score")
        self.geo_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name="f_geo")

    def forward(self, x, targets=None):
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)
        f_score = F.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (F.sigmoid(f_geo) - 0.5) * 2 * 800

        pred = {'f_score': f_score, 'f_geo': f_geo}
        return pred


class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        return x
class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = torch.sigmoid(outputs) * inputs
        return outputs

class MobileNetV3(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_name='large',
                 scale=0.5,
                 disable_se=False,
                 **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super().__init__()

        self.disable_se = disable_se

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 2],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', 2],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', 2],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', 2],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish')

        self.stages = []
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            start_idx = 2 if model_name == 'large' else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl))
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                if_act=True,
                act='hardswish'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))
        for i, stage in enumerate(self.stages):
            self.add_module(name="stage{}".format(i),module=stage)

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class EASTPostProcess:
    """
    The post process for EAST.
    """

    def __init__(self,
                 score_thresh=0.8,
                 cover_thresh=0.1,
                 nms_thresh=0.2,
                 **kwargs):

        self.score_thresh = score_thresh
        self.cover_thresh = cover_thresh
        self.nms_thresh = nms_thresh

    def restore_rectangle_quad(self, origin, geometry):
        """
        Restore rectangle from quadrangle.
        """
        # quad
        origin_concat = np.concatenate(
            (origin, origin, origin, origin), axis=1)  # (n, 8)
        pred_quads = origin_concat - geometry
        pred_quads = pred_quads.reshape((-1, 4, 2))  # (n, 4, 2)
        return pred_quads

    def detect(self,
               score_map,
               geo_map,
               score_thresh=0.8,
               cover_thresh=0.1,
               nms_thresh=0.2):
        """
        restore text boxes from score map and geo map
        """

        score_map = score_map[0]
        geo_map = np.swapaxes(geo_map, 1, 0)
        geo_map = np.swapaxes(geo_map, 1, 2)
        # filter the score map
        xy_text = np.argwhere(score_map > score_thresh)
        if len(xy_text) == 0:
            return []
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        #restore quad proposals
        text_box_restored = self.restore_rectangle_quad(
            xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

        try:
            import lanms
            boxes = lanms.merge_quadrangle_n9(boxes, nms_thresh)
        except:
            print(
                'you should install lanms by pip3 install lanms-nova to speed up nms_locality'
            )
            # boxes = nms_locality(boxes.astype(np.float64), nms_thresh)
        if boxes.shape[0] == 0:
            return []
        # Here we filter some low score boxes by the average score map,
        #   this is different from the orginal paper.
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            # cv2.fillPoly(mask, box[:8].reshape(
            #     (-1, 4, 2)).astype(np.int32) // 4, 1)
            # boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > cover_thresh]
        return boxes

    def sort_poly(self, p):
        """
        Sort polygons.
        """
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4,\
            (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def __call__(self, outs_dict, shape_list):
        score_list = outs_dict['f_score']
        geo_list = outs_dict['f_geo']
        if isinstance(score_list, torch.Tensor):
            score_list = score_list.numpy()
            geo_list = geo_list.numpy()
        img_num = len(shape_list)
        dt_boxes_list = []
        for ino in range(img_num):
            score = score_list[ino]
            geo = geo_list[ino]
            boxes = self.detect(
                score_map=score,
                geo_map=geo,
                score_thresh=self.score_thresh,
                cover_thresh=self.cover_thresh,
                nms_thresh=self.nms_thresh)
            boxes_norm = []
            if len(boxes) > 0:
                h, w = score.shape[1:]
                src_h, src_w, ratio_h, ratio_w = shape_list[ino]
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                for i_box, box in enumerate(boxes):
                    box = self.sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 \
                        or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    boxes_norm.append(box)
            dt_boxes_list.append({'points': np.array(boxes_norm)})
        return dt_boxes_list
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        DiceLoss function.
        """

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = torch.sum(pred * gt * mask)

        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss
class EASTLoss(nn.Module):
    """
    """

    def __init__(self,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        l_score, l_geo, l_mask = labels[1:]
        f_score = predicts['f_score']
        f_geo = predicts['f_geo']

        dice_loss = self.dice_loss(f_score, l_score, l_mask)

        channels = 8
        l_geo_split = torch.split(
            l_geo, channels + 1, 1)
        f_geo_split = torch.split(f_geo, channels, 1)
        smooth_l1 = 0
        for i in range(0, channels):
            geo_diff = l_geo_split[i] - f_geo_split[i]
            abs_geo_diff = torch.abs(geo_diff)
            smooth_l1_sign = torch.lt(abs_geo_diff, l_score)
            smooth_l1_sign = smooth_l1_sign.type(dtype=torch.float32)
            in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + \
                (abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
            out_loss = l_geo_split[-1] / channels * in_loss * l_score
            smooth_l1 += out_loss
        smooth_l1_loss = torch.mean(smooth_l1 * l_score)

        dice_loss = dice_loss * 0.01
        total_loss = dice_loss + smooth_l1_loss
        losses = {"loss":total_loss, \
                  "dice_loss":dice_loss,\
                  "smooth_l1_loss":smooth_l1_loss}
        return losses
