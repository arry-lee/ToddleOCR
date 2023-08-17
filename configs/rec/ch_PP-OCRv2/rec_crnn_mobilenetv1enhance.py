# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from ptocr.modules.backbones.rec_mv1_enhance import MobileNetV1Enhance
from ptocr.modules.necks.rnn import SequenceEncoder
from ptocr.modules.heads.ctc import CTCHead
from ptocr.loss.compose import CombinedLoss
from ptocr.metrics.rec import RecMetric
from torch.optim import Adam
from ptocr.optim.lr_scheduler import PiecewiseLR
from ptocr.postprocess.rec import CTCLabelDecode
from ptocr.datasets.simple_dataset import SimpleDataSet
from ptocr.transforms.operators import DecodeImage, KeepKeys
from ptocr.transforms.rec_img_aug import RecResizeImg, RecAug
from ptocr.transforms.label_ops import CTCLabelEncode
class Model(ConfigModel):
    debug = False
    use_gpu = True
    epoch_num = 800
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = "./output/rec_mobile_pp-OCRv2_enhanced_ctc_loss"
    save_epoch_step = 3
    eval_batch_step = [0, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "doc/imgs_words/ch/word_1.jpg"
    character_dict_path = "ppocr/utils/ppocr_keys_v1.txt"
    max_text_length = 25
    infer_mode = False
    use_space_char = True
    distributed = True
    save_res_path = "./output/rec/predicts_mobile_pp-OCRv2_enhanced_ctc_loss.txt"
    model_type = 'rec'
    algorithm = 'CRNN'
    Transform = None
    Backbone = _(MobileNetV1Enhance, scale=0.5)
    Neck = _(SequenceEncoder, encoder_type="rnn", hidden_size=64)
    Head = _(CTCHead, mid_channels=96, fc_decay=2e-05, return_feats=True)
    loss = CombinedLoss(loss_config_list=[{'CTCLoss': {'use_focal_loss': False, 'weight': 1.0}}, {'CenterLoss': {'weight': 0.05, 'num_classes': 6625, 'feat_dim': 96, 'center_file_path': None}}])
    metric = RecMetric(main_indicator="acc")
    Optimizer = _(Adam,betas=[0.9, 0.999])
    LRScheduler = _(PiecewiseLR,decay_epochs=[700], values=[0.001, 0.0001], warmup_epoch=5)
    PostProcessor = _(CTCLabelDecode,)
    class Train:
        Dataset = _(SimpleDataSet, data_dir="./train_data/", label_file_list=['./train_data/train_list.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), RecAug(), CTCLabelEncode(), RecResizeImg(image_shape=[3, 32, 320]), KeepKeys(keep_keys=['image', 'label', 'length', 'label_ace'])]
        DATALOADER = _(shuffle=True, batch_size=128, drop_last=True, num_workers=8)
    class Eval:
        Dataset = _(SimpleDataSet, data_dir="./train_data", label_file_list=['./train_data/val_list.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), CTCLabelEncode(), RecResizeImg(image_shape=[3, 32, 320]), KeepKeys(keep_keys=['image', 'label', 'length'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=128, num_workers=8)
