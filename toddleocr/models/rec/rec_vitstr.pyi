# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from toddleocr.config import _, ConfigModel
from toddleocr.datasets.lmdb import LMDBDataSet
from toddleocr.loss.ce import CELoss
from toddleocr.metrics.rec import RecMetric
from toddleocr.modules.backbones.rec_vitstr import ViTSTR
from toddleocr.modules.heads.ctc import CTCHead
from toddleocr.modules.necks.rnn import SequenceEncoder
from toddleocr.postprocess.rec import ViTSTRLabelDecode
from toddleocr.transforms.label_ops import ViTSTRLabelEncode
from toddleocr.transforms.operators import DecodeImage, KeepKeys
from toddleocr.transforms.rec_img_aug import GrayRecResizeImg
from torch.optim import Adadelta
from torch.optim.lr_scheduler import ConstantLR

class Model(ConfigModel):
    use_gpu = True
    epoch_num = 20
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = None
    save_epoch_step = 1
    eval_batch_step = [0, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    character_dict_path = "ppocr/utils/dict/en_symbol_dict.txt"
    max_text_length = 25
    infer_mode = False
    use_space_char = False
    model_type = "rec"
    algorithm = "ViTSTR"
    in_channels = 1
    Transform = None
    Backbone = _(ViTSTR, scale="tiny")
    Neck = _(SequenceEncoder, encoder_type="reshape")
    Head = _(
        CTCHead,
    )
    loss = CELoss(with_all=True, ignore_index=0)
    metric = RecMetric(main_indicator="acc")
    postprocessor = ViTSTRLabelDecode()
    Optimizer = _(Adadelta, eps=1e-08, rho=0.95, clip_norm=5.0, lr=1.0)
    LRScheduler = _(
        ConstantLR,
    )

    class Data:
        dataset = LMDBDataSet
        root: "train_data/data_lmdb_release/evaluation/" = (
            "train_data/data_lmdb_release/training/"
        )

    class Loader:
        shuffle: False = True
        drop_last: False = True
        batch_size: 256 = 48
        num_workers: 2 = 8
    Transforms = _[
        DecodeImage(img_mode="BGR", channel_first=False),
        ViTSTRLabelEncode(ignore_index=0) : ...,
        GrayRecResizeImg(
            image_shape=[224, 224],
            resize_type="PIL",
            inter_type="Image.BICUBIC",
            scale=False,
        ),
        KeepKeys("image", "label", "length") : ...,
    ]