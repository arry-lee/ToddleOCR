# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from ppocr.models.backbones.rec_resnet_fpn import ResNetFPN
from ppocr.models.heads.rec_srn_head import SRNHead
from ppocr.losses.rec_srn_loss import SRNLoss
from ppocr.metrics.rec_metric import RecMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
from ppocr.postprocess.rec_postprocess import SRNLabelDecode
from ppocr.data.lmdb_dataset import LMDBDataSet
from ppocr.data.imaug.operators import DecodeImage, KeepKeys
from ppocr.data.imaug.label_ops import SRNLabelEncode
from ppocr.data.imaug.rec_img_aug import SRNRecResizeImg
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 72
    log_window_size = 20
    log_batch_step = 5
    save_model_dir = "./output/rec/srn_new"
    save_epoch_step = 3
    eval_batch_step = [0, 5000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "doc/imgs_words/ch/word_1.jpg"
    character_dict_path = None
    max_text_length = 25
    num_heads = 8
    infer_mode = False
    use_space_char = False
    save_res_path = "./output/rec/predicts_srn.txt"
    model_type = 'rec'
    algorithm = 'SRN'
    in_channels = 1
    Transform = None
    Backbone = _(ResNetFPN, )
    Head = _(SRNHead, max_text_length=25, num_heads=8, num_encoder_TUs=2, num_decoder_TUs=4, hidden_dims=512)
    loss = SRNLoss()
    metric = RecMetric(main_indicator="acc")
    Optimizer = _(Adam,betas=[0.9, 0.999], clip_norm=10.0, lr=0.0001)
    LRScheduler = _(ConstantLR,)
    PostProcessor = _(SRNLabelDecode,)
    class Train:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/training/")
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), SRNLabelEncode(), SRNRecResizeImg(image_shape=[1, 64, 256]), KeepKeys(keep_keys=['image', 'label', 'length', 'encoder_word_pos', 'gsrm_word_pos', 'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'])]
        DATALOADER = _(shuffle=False, batch_size=64, drop_last=False, num_workers=4)
    class Eval:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/validation/")
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), SRNLabelEncode(), SRNRecResizeImg(image_shape=[1, 64, 256]), KeepKeys(keep_keys=['image', 'label', 'length', 'encoder_word_pos', 'gsrm_word_pos', 'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=32, num_workers=4)
