# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from ptocr.modules.backbones.rec_resnet_31 import ResNet31
from ptocr.modules.heads.robustscanner import RobustScannerHead
from ptocr.loss.sar import SARLoss
from ptocr.metrics.rec import RecMetric
from torch.optim import Adam
from ptocr.optim.lr_scheduler import PiecewiseLR
from ptocr.postprocess.rec import SARLabelDecode
from ptocr.datasets.lmdb_dataset import LMDBDataSet
from ptocr.transforms.operators import DecodeImage, KeepKeys
from ptocr.transforms.label_ops import SARLabelEncode
from ptocr.transforms.rec_img_aug import RobustScannerRecResizeImg
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 5
    log_window_size = 20
    log_batch_step = 20
    save_model_dir = "./output/rec/rec_r31_robustscanner/"
    save_epoch_step = 1
    eval_batch_step = [0, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "doc/imgs_words_en/word_10.png"
    character_dict_path = "ppocr/utils/dict90.txt"
    max_text_length = 40
    infer_mode = False
    use_space_char = False
    rm_symbol = True
    save_res_path = "./output/rec/predicts_robustscanner.txt"
    model_type = 'rec'
    algorithm = 'RobustScanner'
    Transform = None
    Backbone = _(ResNet31, init_type="KaimingNormal")
    Head = _(RobustScannerHead, enc_outchannles=128, hybrid_dec_rnn_layers=2, hybrid_dec_dropout=0, position_dec_rnn_layers=2, start_idx=91, mask=True, padding_idx=92, encode_value=False, max_text_length=40)
    loss = SARLoss()
    metric = RecMetric(is_filter=True)
    Optimizer = _(Adam,betas=[0.9, 0.999])
    LRScheduler = _(PiecewiseLR,decay_epochs=[3, 4], values=[0.001, 0.0001, 1e-05])
    PostProcessor = _(SARLabelDecode,)
    class Train:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/training/")
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), SARLabelEncode(), RobustScannerRecResizeImg(image_shape=[3, 48, 48, 160], width_downsample_ratio=0.25, max_text_length=40), KeepKeys(keep_keys=['image', 'label', 'valid_ratio', 'word_positons'])]
        DATALOADER = _(shuffle=True, batch_size=64, drop_last=True, num_workers=8, use_shared_memory=False)
    class Eval:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/evaluation/")
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), SARLabelEncode(), RobustScannerRecResizeImg(image_shape=[3, 48, 48, 160], max_text_length=40, width_downsample_ratio=0.25), KeepKeys(keep_keys=['image', 'label', 'valid_ratio', 'word_positons'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=64, num_workers=4, use_shared_memory=False)
