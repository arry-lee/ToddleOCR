# This .py is auto generated by the script in the root folder.
from ptocr.config import ConfigModel, _
from ptocr.modules.backbones.mobilenetv3.det_mobilenet_v3 import MobileNetV3
from ptocr.modules.heads.cls import ClsHead
from ptocr.loss.cls import ClsLoss
from ptocr.metrics.cls import ClsMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from ptocr.postprocess.cls import ClsPostProcess
from ptocr.datasets.simple import SimpleDataSet
from ptocr.transforms.operators import KeepKeys, DecodeImage
from ptocr.transforms.label_ops import ClsLabelEncode
from ptocr.transforms.rec_img_aug import BaseDataAugmentation, ClsResizeImg
from ptocr.transforms.randaugment import RandAugment


class Model(ConfigModel):
    use_gpu = True
    epoch_num = 100
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = "./output/cls/mv3/"
    save_epoch_step = 3
    eval_batch_step = [0, 1000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "doc/imgs_words_en/word_10.png"
    label_list = ["0", "180"]
    model_type = "cls"
    algorithm = "CLS"
    Transform = None
    Backbone = _(MobileNetV3, scale=0.35, model_name="small")
    Neck = None
    Head = _(ClsHead, class_dim=2)
    loss = ClsLoss()
    metric = ClsMetric(main_indicator="acc")
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(
        CosineAnnealingLR,
    )
    PostProcessor = _(
        ClsPostProcess,
    )

    class Train:
        Dataset = _(
            SimpleDataSet,
            data_dir="./train_data/cls",
            label_file_list=["./train_data/cls/train.txt"],
        )
        transforms = _[
            DecodeImage(img_mode="BGR", channel_first=False),
            ClsLabelEncode(),
            BaseDataAugmentation(),
            RandAugment(),
            ClsResizeImg(image_shape=[3, 48, 192]),
            KeepKeys(keep_keys=["image", "label"]),
        ]
        DATALOADER = _(
            shuffle=True, batch_size=512, drop_last=True, num_workers=8
        )

    class Eval:
        Dataset = _(
            SimpleDataSet,
            data_dir="./train_data/cls",
            label_file_list=["./train_data/cls/test.txt"],
        )
        transforms = _[
            DecodeImage(img_mode="BGR", channel_first=False),
            ClsLabelEncode(),
            ClsResizeImg(image_shape=[3, 48, 192]),
            KeepKeys(keep_keys=["image", "label"]),
        ]
        DATALOADER = _(
            shuffle=False, drop_last=False, batch_size=512, num_workers=4
        )
