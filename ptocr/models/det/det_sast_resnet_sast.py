# This .py is auto generated by the script in the root folder.
from ptocr.config import ConfigModel, _
from ptocr.modules.backbones.resnet.det_resnet_vd_sast import ResNet_SAST
from ptocr.modules.necks.sast_fpn import SASTFPN
from ptocr.modules.heads.sast import SASTHead
from ptocr.loss.sast import SASTLoss
from ptocr.metrics.det import DetMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
from ptocr.postprocess.sast import SASTPostProcess
from ptocr.datasets.simple import SimpleDataSet
from ptocr.transforms.operators import (
    ToCHWImage,
    DetResizeForTest,
    DecodeImage,
    NormalizeImage,
    KeepKeys,
)
from ptocr.transforms.label_ops import DetLabelEncode
from ptocr.transforms.sast_process import SASTProcessTrain


class Model(ConfigModel):
    use_gpu = True
    epoch_num = 5000
    log_window_size = 20
    log_batch_step = 2
    save_model_dir = "./output/sast_r50_vd_tt/"
    save_epoch_step = 1000
    eval_batch_step = [4000, 5000]
    metric_during_train = False
    pretrained_model = "./pretrain_models/ResNet50_vd_ssld_pretrained"
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = None
    save_res_path = "./output/sast_r50_vd_tt/predicts_sast.txt"
    model_type = "det"
    algorithm = "SAST"
    Transform = None
    Backbone = _(ResNet_SAST, layers=50)
    Neck = _(SASTFPN, with_cab=True)
    Head = _(
        SASTHead,
    )
    loss = SASTLoss()
    metric = DetMetric(main_indicator="hmean")
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(
        ConstantLR,
    )
    PostProcessor = _(
        SASTPostProcess,
        score_thresh=0.5,
        sample_pts_num=6,
        nms_thresh=0.2,
        expand_scale=1.2,
        shrink_ratio_of_width=0.2,
    )

    class Train:
        Dataset = _(
            SimpleDataSet,
            data_dir="./train_data/",
            label_file_list=[
                "./train_data/art_latin_icdar_14pt/train_no_tt_test/train_label_json.txt",
                "./train_data/total_text_icdar_14pt/train_label_json.txt",
            ],
            ratio_list=[0.5, 0.5],
        )
        transforms = _[
            DecodeImage(img_mode="BGR", channel_first=False),
            DetLabelEncode(),
            SASTProcessTrain(
                image_shape=[512, 512],
                min_crop_side_ratio=0.3,
                min_crop_size=24,
                min_text_size=4,
                max_text_size=512,
            ),
            KeepKeys(
                keep_keys=[
                    "image",
                    "score_map",
                    "border_map",
                    "training_mask",
                    "tvo_map",
                    "tco_map",
                ]
            ),
        ]
        DATALOADER = _(
            shuffle=True, drop_last=False, batch_size=4, num_workers=4
        )

    class Eval:
        Dataset = _(
            SimpleDataSet,
            data_dir="./train_data/",
            label_file_list=[
                "./train_data/total_text_icdar_14pt/test_label_json.txt"
            ],
        )
        transforms = _[
            DecodeImage(img_mode="BGR", channel_first=False),
            DetLabelEncode(),
            DetResizeForTest(resize_long=768),
            NormalizeImage(
                scale="1./255.",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                order="hwc",
            ),
            ToCHWImage(),
            KeepKeys(keep_keys=["image", "shape", "polys", "ignore_tags"]),
        ]
        DATALOADER = _(
            shuffle=False, drop_last=False, batch_size=1, num_workers=2
        )
