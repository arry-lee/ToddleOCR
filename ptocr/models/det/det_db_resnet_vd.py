import os
import sys
sys.path.append(os.getcwd())
from ptocr.config import ConfigModel, _
from ptocr.modules.backbones.resnet.det_resnet_vd import ResNet_vd
from ptocr.modules.necks.db_fpn import DBFPN
from ptocr.modules.heads.db import DBHead
from ptocr.loss.db import DBLoss
from ptocr.metrics.det import DetMetric
from ptocr.postprocess.db import DBPostProcess
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ptocr.datasets.simple import SimpleDataSet
from ptocr.transforms.operators import ToCHWImage, DetResizeForTest, KeepKeys, NormalizeImage, DecodeImage
from ptocr.transforms.label_ops import DetLabelEncode
from ptocr.transforms.iaa_augment import IaaAugment
from ptocr.transforms.random_crop_data import EastRandomCropData
from ptocr.transforms.make_border_map import MakeBorderMap
from ptocr.transforms.make_shrink_map import MakeShrinkMap

class Model(ConfigModel):
    use_gpu = True
    epoch_num = 1200
    log_window_size = 20
    log_batch_step = 2
    save_epoch_step = 1200
    eval_batch_step = [3000, 2000]
    metric_during_train = False
    pretrained_model = None
    checkpoints = None
    model_type = 'det'
    algorithm = 'DB'
    Backbone = _(ResNet_vd, layers=18)
    Neck = _(DBFPN, out_channels=256)
    Head = _(DBHead, k=50)
    loss = DBLoss(balance_loss=True, main_loss_type='DiceLoss', alpha=5, beta=10, ohem_ratio=3)
    metric = DetMetric(main_indicator='hmean')
    postprocessor = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=2)
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(CosineAnnealingWarmRestarts, T_0=2)

    class Data:
        dataset = SimpleDataSet
        root = './train_data/icdar2015/text_localization/'
        label_file_list: ['train_data/icdar2015/text_localization/test_icdar2015_label.txt'] = ['./train_data/icdar2015/text_localization/train_icdar2015_label.txt']
        ratio_list = [1.0]

    class Loader:
        shuffle: 0 = True
        drop_last: 0 = False
        batch_size: 1 = 8
        num_workers: 2 = 4
    Transforms = _[DecodeImage(img_mode='BGR', channel_first=False), DetLabelEncode():..., IaaAugment(augmenter_args=[{'type': 'Fliplr', 'args': {'p': 0.5}}, {'type': 'Affine', 'args': {'rotate': [-10, 10]}}, {'type': 'Resize', 'args': {'size': [0.5, 3]}}]):, EastRandomCropData(size=[960, 960], max_tries=50, keep_ratio=True):, MakeBorderMap(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):, MakeShrinkMap(shrink_ratio=0.4, min_text_size=8):, :DetResizeForTest():..., NormalizeImage(scale='1./255.', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order='hwc'), ToCHWImage(), KeepKeys('image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask'):KeepKeys('image', 'shape', 'polys', 'ignore_tags'):KeepKeys('image', 'shape')]

def transmodel():
    import torch
    import paddle

    def p2t(tensor) -> torch.Tensor:
        return torch.from_numpy(tensor.numpy())
    m = Model()
    x = m.model.state_dict()
    t_keys = [k for k in x.keys()]
    print('模型参数', len(t_keys))
    paddle_params = paddle.load('D:\\dev\\.model\\paddocr\\ch_ppocr_server_v2.0_det_train\\best_accuracy.pdparams')
    print('文件参数', len(paddle_params))
    keys = list(paddle_params.keys())
    keys.sort()
    print(t_keys)
    print(keys)
    print(len(t_keys), len(keys))
    maps = {'bn.running_mean': '_batch_norm._mean', 'bn.running_var': '_batch_norm._variance', 'bn.bias': '_batch_norm.bias', 'bn.weight': '_batch_norm.weight', 'conv.weight': '_conv.weight'}
    tset = set()
    for k in t_keys:
        (l, m, r) = k.rsplit('.', 2)
        if m + '.' + r in maps:
            k = l + '.' + maps[m + '.' + r]
        tset.add(k)
    pset = set(keys)
    u = pset - tset
    print(u)
    print(len(u))
    d = {}
    transmap = {'_variance': 'running_var', '_mean': 'running_mean'}
    d = dict()
    for (k, v) in paddle_params.items():
        (p, n) = k.rsplit('.', 1)
        if n in transmap:
            o = p + '.' + transmap[n]
            d[o] = p2t(v)
            if n == '_variance':
                d[p + '.num_batches_tracked'] = torch.tensor(200)
        else:
            d[k] = p2t(v)
    torch.save(d, 'D:\\dev\\.model\\paddocr\\ch_ppocr_server_v2.0_det_train\\best_accuracy.pth')
if __name__ == '__main__':
    m = Model()
    m.infer('D:\\dev\\github\\PaddleOCR\\doc\\imgs\\11.jpg', 'D:\\dev\\.model\\paddocr\\ch_ppocr_server_v2.0_det_train\\best_accuracy.pth')