# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from ptocr.modules.backbones.rec_efficientb3_pren import EfficientNetB3_PREN
from ptocr.modules.necks.pren_fpn import PRENFPN
from ptocr.modules.heads.pren import PRENHead
from ptocr.loss.pren import PRENLoss
from ptocr.metrics.rec import RecMetric
from torch.optim import Adadelta
from ptocr.optim.lr_scheduler import PiecewiseLR
from ptocr.postprocess.rec import PRENLabelDecode
from ptocr.datasets.lmdb_dataset import LMDBDataSet
from ptocr.transforms.operators import DecodeImage, KeepKeys
from ptocr.transforms.label_ops import PRENLabelEncode
from ptocr.transforms.rec_img_aug import PRENResizeImg, RecAug
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 8
    log_window_size = 20
    log_batch_step = 5
    save_model_dir = "./output/rec/pren_new"
    save_epoch_step = 3
    eval_batch_step = [4000, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "doc/imgs_words/ch/word_1.jpg"
    character_dict_path = None
    max_text_length = 25
    infer_mode = False
    use_space_char = False
    save_res_path = "./output/rec/predicts_pren.txt"
    model_type = 'rec'
    algorithm = 'PREN'
    in_channels = 3
    Backbone = _(EfficientNetB3_PREN, )
    Neck = _(PRENFPN, n_r=5, d_model=384, max_len=25, dropout=0.1)
    Head = _(PRENHead, )
    loss = PRENLoss()
    metric = RecMetric(main_indicator="acc")
    Optimizer = _(Adadelta,)
    LRScheduler = _(PiecewiseLR,decay_epochs=[2, 5, 7], values=[0.5, 0.1, 0.01, 0.001])
    PostProcessor = _(PRENLabelDecode,)
    class Train:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/training/")
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), PRENLabelEncode(), RecAug(), PRENResizeImg(image_shape=[64, 256]), KeepKeys(keep_keys=['image', 'label'])]
        DATALOADER = _(shuffle=True, batch_size=128, drop_last=True, num_workers=8)
    class Eval:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/validation/")
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), PRENLabelEncode(), PRENResizeImg(image_shape=[64, 256]), KeepKeys(keep_keys=['image', 'label'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=64, num_workers=8)
