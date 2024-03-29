# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from toddleocr.config import _, ConfigModel
from toddleocr.datasets.lmdb import LMDBDataSet
from toddleocr.loss.ce import CELoss
from toddleocr.metrics.rec import RecMetric
from toddleocr.modules.backbones.rec_nrtr_mtb import MTB
from toddleocr.modules.heads.nrtr import Transformer
from toddleocr.postprocess.rec import NRTRLabelDecode
from toddleocr.transforms.label_ops import NRTRLabelEncode
from toddleocr.transforms.operators import DecodeImage, KeepKeys
from toddleocr.transforms.rec_img_aug import GrayRecResizeImg
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class Model(ConfigModel):
    use_gpu = True
    epoch_num = 21
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
    algorithm = "NRTR"
    in_channels = 1
    Transform = None
    Backbone = _(MTB, cnn_num=2)
    Head = _(Transformer, d_model=512, num_encoder_layers=6, beam_size=-1)
    loss = CELoss(smoothing=True)
    metric = RecMetric(main_indicator="acc")
    postprocessor = NRTRLabelDecode()
    Optimizer = _(Adam, betas=[0.9, 0.99], clip_norm=5.0, lr=0.0005)
    LRScheduler = _(CosineAnnealingWarmRestarts, T_0=2)

    class Data:
        dataset = LMDBDataSet
        root: "train_data/data_lmdb_release/evaluation/" = (
            "train_data/data_lmdb_release/training/"
        )

    class Loader:
        shuffle: False = True
        drop_last: False = True
        batch_size: 256 = 512
        num_workers: 4 = 8
        pin_memory = False
    Transforms = _[
        DecodeImage(img_mode="BGR", channel_first=False),
        NRTRLabelEncode() : ...,
        GrayRecResizeImg(image_shape=[100, 32], resize_type="PIL"),
        KeepKeys("image", "label", "length") : ...,
    ]
