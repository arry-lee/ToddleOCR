# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from toddleocr.config import _, ConfigModel
from toddleocr.datasets.simple import SimpleDataSet
from toddleocr.loss.pse import PSELoss
from toddleocr.metrics.det import DetMetric
from toddleocr.modules.backbones.mobilenetv3.det_mobilenet_v3 import MobileNetV3
from toddleocr.modules.heads.pse import PSEHead
from toddleocr.modules.necks.fpn import FPN
from toddleocr.postprocess.pse import PSEPostProcess
from toddleocr.transforms.ColorJitter import ColorJitter
from toddleocr.transforms.iaa_augment import IaaAugment
from toddleocr.transforms.label_ops import DetLabelEncode
from toddleocr.transforms.make_pse_gt import MakePseGt
from toddleocr.transforms.operators import (
    DecodeImage,
    DetResizeForTest,
    KeepKeys,
    NormalizeImage,
    ToCHWImage,
)
from toddleocr.transforms.random_crop_data import RandomCropImgMask
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

class Model(ConfigModel):
    use_gpu = True
    epoch_num = 600
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = None
    save_epoch_step = 600
    eval_batch_step = [0, 63]
    metric_during_train = False
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    model_type = "det"
    algorithm = "PSE"
    Transform = None
    Backbone = _(MobileNetV3, scale=0.5, model_name="large")
    Neck = _(FPN, out_channels=96)
    Head = _(PSEHead, hidden_dim=96, out_channels=7)
    loss = PSELoss(alpha=0.7, ohem_ratio=3, kernel_sample_mask="pred", reduction="none")
    metric = DetMetric(main_indicator="hmean")
    postprocessor = PSEPostProcess(
        thresh=0, box_thresh=0.85, min_area=16, box_type="quad", scale=1
    )
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(StepLR, step_size=200, gamma=0.1)

    class Data:
        dataset = SimpleDataSet
        root = "train_data/icdar2015/text_localization/"
        label_file_list: "test_icdar2015_label.txt" = "train_icdar2015_label.txt"
        ratio_list = "1.0"

    class Loader:
        shuffle: False = True
        drop_last = False
        batch_size: 1 = 16
        num_workers = 8
    Transforms = _[
        DecodeImage(img_mode="BGR", channel_first=False),
        DetLabelEncode() : ...,
        ColorJitter(brightness=0.12549019607843137, saturation=0.5) :,
        IaaAugment(
            augmenter_args=[
                {"type": "Resize", "args": {"size": [0.5, 3]}},
                {"type": "Fliplr", "args": {"p": 0.5}},
                {"type": "Affine", "args": {"rotate": [-10, 10]}},
            ]
        ) :,
        MakePseGt(kernel_num=7, min_shrink_ratio=0.4, size=640) :,
        RandomCropImgMask(
            size=[640, 640],
            main_key="gt_text",
            crop_keys=["image", "gt_text", "gt_kernels", "mask"],
        ) :,
        : DetResizeForTest(limit_side_len=736, limit_type="min"),
        NormalizeImage(
            scale="1./255.",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order="hwc",
        ),
        ToCHWImage(),
        KeepKeys("image", "gt_text", "gt_kernels", "mask") : KeepKeys(
            "image", "shape", "polys", "ignore_tags"
        ) : ...,
    ]
