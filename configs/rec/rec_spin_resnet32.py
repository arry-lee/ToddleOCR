# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from torch.nn import GA_SPIN
from ptocr.modules.backbones.rec_resnet_32 import ResNet32
from ptocr.modules.necks.rnn import SequenceEncoder
from ptocr.modules.heads.spin_att import SPINAttentionHead
from ptocr.loss.spin import SPINAttentionLoss
from ptocr.metrics.rec import RecMetric
from torch.optim import AdamW
from ptocr.optim.lr_scheduler import PiecewiseLR
from ptocr.postprocess.rec import SPINLabelDecode
from ptocr.datasets.simple_dataset import SimpleDataSet
from ptocr.transforms.operators import DecodeImage, KeepKeys
from ptocr.transforms.label_ops import SPINLabelEncode
from ptocr.transforms.rec_img_aug import SPINRecResizeImg
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 6
    log_window_size = 50
    log_batch_step = 50
    save_model_dir = "./output/rec/rec_r32_gaspin_bilstm_att/"
    save_epoch_step = 3
    eval_batch_step = [0, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "doc/imgs_words_en/word_10.png"
    character_dict_path = "./ppocr/utils/dict/spin_dict.txt"
    max_text_length = 25
    infer_mode = False
    use_space_char = False
    save_res_path = "./output/rec/predicts_r32_gaspin_bilstm_att.txt"
    model_type = 'rec'
    algorithm = 'SPIN'
    in_channels = 1
    Transform = _(GA_SPIN, offsets=True, default_type=6, loc_lr=0.1, stn=True)
    Backbone = _(ResNet32, out_channels=512)
    Neck = _(SequenceEncoder, encoder_type="cascadernn", hidden_size=256, out_channels=[256, 512], with_linear=True)
    Head = _(SPINAttentionHead, hidden_size=256)
    loss = SPINAttentionLoss(ignore_index=0)
    metric = RecMetric(main_indicator="acc", is_filter=True)
    Optimizer = _(AdamW,beta1=0.9, beta2=0.999)
    LRScheduler = _(PiecewiseLR,decay_epochs=[3, 4, 5], values=[0.001, 0.0003, 9e-05, 2.7e-05], clip_norm=5)
    PostProcessor = _(SPINLabelDecode,use_space_char=False)
    class Train:
        Dataset = _(SimpleDataSet, data_dir="./train_data/ic15_data/", label_file_list=['./train_data/ic15_data/rec_gt_train.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), SPINLabelEncode(), SPINRecResizeImg(image_shape=[100, 32], interpolation=2, mean=[127.5], std=[127.5]), KeepKeys(keep_keys=['image', 'label', 'length'])]
        DATALOADER = _(shuffle=True, batch_size=8, drop_last=True, num_workers=4)
    class Eval:
        Dataset = _(SimpleDataSet, data_dir="./train_data/ic15_data", label_file_list=['./train_data/ic15_data/rec_gt_test.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), SPINLabelEncode(), SPINRecResizeImg(image_shape=[100, 32], interpolation=2, mean=[127.5], std=[127.5]), KeepKeys(keep_keys=['image', 'label', 'length'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=8, num_workers=2)
