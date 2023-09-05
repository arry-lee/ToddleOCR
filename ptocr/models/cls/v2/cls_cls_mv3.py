# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from ptocr.config import ConfigModel,_
from ptocr.modules.backbones.mobilenetv3.rec_mobilenet_v3 import MobileNetV3Rec
from ptocr.modules.heads.cls import ClsHead
from ptocr.loss.cls import ClsLoss
from ptocr.metrics.cls import ClsMetric
from ptocr.postprocess.cls import ClsPostProcess
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from ptocr.datasets.simple import SimpleDataSet
from ptocr.transforms.operators import DecodeImage, KeepKeys
from ptocr.transforms.label_ops import ClsLabelEncode
from ptocr.transforms.rec_img_aug import ClsResizeImg, BaseDataAugmentation
from ptocr.transforms.randaugment import RandAugment
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 100
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = None
    save_epoch_step = 3
    eval_batch_step = [0, 1000]
    metric_during_train = True
    pretrained_model = "../model/ch_ppocr_mobile_v2.0_cls_train/best_accuracy.pt"
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    label_list = ['0', '180']
    model_type = 'cls'
    algorithm = 'CLS'
    Transform = None
    Backbone = _(MobileNetV3Rec, scale=0.35, model_name="small")
    Neck = None
    Head = _(ClsHead, class_dim=2)
    loss = ClsLoss()
    metric = ClsMetric(main_indicator="acc")
    postprocessor = ClsPostProcess(label_list)
    Optimizer = _(Adam,betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(CosineAnnealingLR,)
    class Data:
        dataset = SimpleDataSet
        root = "train_data/cls"
        label_file_list:"test.txt" = "train.txt"
    class Loader:
        shuffle:False = True
        drop_last:False = True
        batch_size = 512
        num_workers:4 = 8
    Transforms = _[DecodeImage(img_mode="BGR", channel_first=False),ClsLabelEncode(label_list):...,BaseDataAugmentation():,RandAugment():,ClsResizeImg(image_shape=[3, 48, 192]),KeepKeys("image","label"):...:KeepKeys("image")]