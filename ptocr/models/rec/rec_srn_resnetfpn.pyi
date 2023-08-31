# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from ptocr.config import ConfigModel,_
from ptocr.modules.backbones.resnet.rec_resnet_fpn import ResNetFPN
from ptocr.modules.heads.srn import SRNHead
from ptocr.loss.srn import SRNLoss
from ptocr.metrics.rec import RecMetric
from ptocr.postprocess.rec import SRNLabelDecode
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
from ptocr.datasets.lmdb import LMDBDataSet
from ptocr.transforms.operators import DecodeImage, KeepKeys
from ptocr.transforms.label_ops import SRNLabelEncode
from ptocr.transforms.rec_img_aug import SRNRecResizeImg
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 72
    log_window_size = 20
    log_batch_step = 5
    save_model_dir = None
    save_epoch_step = 3
    eval_batch_step = [0, 5000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    character_dict_path = None
    max_text_length = 25
    num_heads = 8
    infer_mode = False
    use_space_char = False
    model_type = 'rec'
    algorithm = 'SRN'
    in_channels = 1
    Transform = None
    Backbone = _(ResNetFPN, )
    Head = _(SRNHead, max_text_length=25, num_heads=8, num_encoder_TUs=2, num_decoder_TUs=4, hidden_dims=512)
    loss = SRNLoss()
    metric = RecMetric(main_indicator="acc")
    postprocessor = SRNLabelDecode()
    Optimizer = _(Adam,betas=[0.9, 0.999], clip_norm=10.0, lr=0.0001)
    LRScheduler = _(ConstantLR,)
    class Data:
        dataset = LMDBDataSet
        root:"train_data/data_lmdb_release/validation/" = "train_data/data_lmdb_release/training/"
    class Loader:
        shuffle = False
        drop_last = False
        batch_size:32 = 64
        num_workers = 4
    Transforms = _[DecodeImage(img_mode="BGR", channel_first=False),SRNLabelEncode():...,SRNRecResizeImg(image_shape=[1, 64, 256]),KeepKeys("image","label","length","encoder_word_pos","gsrm_word_pos","gsrm_slf_attn_bias1","gsrm_slf_attn_bias2"):...]