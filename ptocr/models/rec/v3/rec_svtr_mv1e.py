import os
import sys

sys.path.append(os.getcwd())
from ptocr.config import ConfigModel, _
from ptocr.modules.backbones.rec_mv1_enhance import MobileNetV1Enhance
from ptocr.modules.heads.multi import MultiHead
from ptocr.loss.compose import MultiLoss
from ptocr.metrics.rec import RecMetric
from ptocr.postprocess.rec import CTCLabelDecode
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ptocr.datasets.simple import SimpleDataSet
from ptocr.transforms import KeepKeys, DecodeImage
from ptocr.transforms import RecAug, RecConAug, RecResizeImg
from ptocr.transforms import MultiLabelEncode

CHARACTER_DICT_PATH = (
    "D:/dev/github/ToddleOCR/ptocr/utils/ppocr_keys_v1.txt"
)
MAX_TEXT_LENGTH = 25
USE_SPACE_CHAR = True


class Model(ConfigModel):
    model_type = "rec"
    algorithm = "SVTR"
    checkpoints = None
    distributed = False
    epoch_num = 500
    eval_batch_step = [0, 2000]
    log_batch_step = 10
    log_window_size = 20
    max_text_length = 25
    metric_during_train = True
    pretrained_model = None#"../model/ch_PP-OCRv3_rec_train/best_accuracy.pt"
    save_epoch_step = 3
    save_infer_dir = None
    save_model_dir = "./output/v3_en_mobile"
    save_res_path = "./output/rec/predicts_ppocrv3_en.txt"
    use_gpu = False
    use_visualdl = False

    class Data:
        dataset = SimpleDataSet
        root = "train_data"
        label_files: ["train_data/train_list.txt"] = ["train_data/train_list.txt"]

    class Loader:
        shuffle: False = True
        batch_size: 2 = 128
        drop_last: True = True
        num_workers: 0 = 2
        pin_memory: False = True

    Transforms = _[
                 DecodeImage(img_mode="BGR", channel_first=False),
                 RecConAug(
                     prob=0.5,
                     ext_data_num=2,
                     image_shape=[48, 320, 3],
                     max_text_length=25,
                 ):,
                 RecAug(),
                 MultiLabelEncode(
                     MAX_TEXT_LENGTH, CHARACTER_DICT_PATH, USE_SPACE_CHAR
                 ): ...,
                 RecResizeImg(image_shape=[3, 48, 320]): ...: RecResizeImg(
                     image_shape=[3, 48, 320], infer_mode=True
                 ),
                 KeepKeys(
                     "image", "label_ctc", "label_sar", "length", "valid_ratio"
                 ): ...: KeepKeys("image"),
                 ]
    Backbone = _(
        MobileNetV1Enhance,
        scale=0.5,
        last_conv_stride=[1, 2],
        last_pool_type="avg",
    )
    Neck = None
    postprocessor = CTCLabelDecode(CHARACTER_DICT_PATH, USE_SPACE_CHAR)
    Head = _(
        MultiHead,
        head_list=[
            {
                "class": "CTCHead",
                "Neck": {
                    "name": "svtr",
                    "dims": 64,
                    "depth": 2,
                    "hidden_dims": 120,
                    "use_guide": True,
                },
                "Head": {"fc_decay": 1e-05},
            },
            {"class": "SARHead", "enc_dim": 512, "max_text_length": 25},
        ],
        out_channels_list={
            "CTCLabelDecode": len(postprocessor.character),
            "SARLabelDecode": len(postprocessor.character) + 2,
        },
    )
    loss = MultiLoss(loss_config_list=[{"CTCLoss": None}, {"SARLoss": None}])
    metric = RecMetric(main_indicator="acc", ignore_space=False)
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(CosineAnnealingWarmRestarts, T_0=5)
