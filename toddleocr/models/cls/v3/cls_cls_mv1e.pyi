# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from toddleocr.config import _, ConfigModel
from toddleocr.datasets.simple import SimpleDataSet
from toddleocr.loss.cls import ClsLoss
from toddleocr.metrics.cls import ClsMetric
from toddleocr.modules.backbones.rec_mv1_enhance import MobileNetV1Enhance
from toddleocr.modules.heads.cls import ClsHead
from toddleocr.postprocess.cls import ClsPostProcess
from toddleocr.transforms.operators import DecodeImage, KeepKeys
from toddleocr.transforms.randaugment import RandAugment
from toddleocr.transforms.rec_img_aug import BaseDataAugmentation
from toddleocr.transforms.ssl_img_aug import SSLRotateResize
from toddleocr.utils.collate_fn import SSLRotateCollate
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class Model(ConfigModel):
    debug = False
    use_gpu = True
    epoch_num = 100
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = None
    save_epoch_step = 3
    eval_batch_step = [0, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    character_dict_path = "toddleocr/utils/dict/chinese_sim_dict.txt"
    max_text_length = 25
    infer_mode = False
    use_space_char = True
    model_type = "cls"
    algorithm = "CLS"
    Transform = None
    Backbone = _(
        MobileNetV1Enhance, scale=0.5, last_conv_stride=[1, 2], last_pool_type="avg"
    )
    Neck = None
    Head = _(ClsHead, class_dim=4)
    loss = ClsLoss(main_indicator="acc")
    metric = ClsMetric(main_indicator="acc")
    postprocessor = ClsPostProcess()
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(
        CosineAnnealingLR,
    )

    class Data:
        dataset = SimpleDataSet
        root = "train_data"
        label_file_list: "val_list.txt" = "train_list.txt"

    class Loader:
        collate_fn = SSLRotateCollate
        shuffle: False = True
        drop_last: False = True
        batch_size: 64 = 32
        num_workers = 8
    Transforms = _[
        DecodeImage(img_mode="BGR", channel_first=False),
        BaseDataAugmentation() :,
        RandAugment() :,
        SSLRotateResize(image_shape=[3, 48, 320]),
        KeepKeys("image", "label") : ...,
    ]
