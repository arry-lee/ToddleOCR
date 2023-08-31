# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from ptocr.config import ConfigModel,_
from ptocr.modules.backbones.mobilenetv3.rec_mobilenet_v3 import MobileNetV3Rec
from ptocr.modules.necks.rnn import SequenceEncoder
from ptocr.modules.heads.ctc import CTCHead
from ptocr.loss.ctc import CTCLoss
from ptocr.metrics.rec import RecMetric
from ptocr.postprocess.rec import CTCLabelDecode
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
from ptocr.datasets.lmdb import LMDBDataSet
from ptocr.transforms.operators import DecodeImage, KeepKeys
from ptocr.transforms.label_ops import CTCLabelEncode
from ptocr.transforms.rec_img_aug import RecResizeImg
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 72
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
    character_dict_path = None
    max_text_length = 25
    infer_mode = False
    use_space_char = False
    model_type = 'rec'
    algorithm = 'Rosetta'
    Transform = None
    Backbone = _(MobileNetV3Rec, scale=0.5, model_name="large")
    Neck = _(SequenceEncoder, encoder_type="reshape")
    Head = _(CTCHead, fc_decay=0.0004)
    loss = CTCLoss()
    metric = RecMetric(main_indicator="acc")
    postprocessor = CTCLabelDecode()
    Optimizer = _(Adam,betas=[0.9, 0.999], lr=0.0005)
    LRScheduler = _(ConstantLR,)
    class Data:
        dataset = LMDBDataSet
        root:"train_data/data_lmdb_release/validation/" = "train_data/data_lmdb_release/training/"
    class Loader:
        shuffle = False
        drop_last:False = True
        batch_size = 256
        num_workers = 8
    Transforms = _[DecodeImage(img_mode="BGR", channel_first=False),CTCLabelEncode():...,RecResizeImg(image_shape=[3, 32, 100]),KeepKeys("image","label","length"):...]