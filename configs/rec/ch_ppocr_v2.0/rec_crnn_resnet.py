# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from ppocr.models.backbones.rec_resnet_vd import ResNet
from ppocr.models.necks.rnn import SequenceEncoder
from ppocr.models.heads.rec_ctc_head import CTCHead
from ppocr.losses.rec_ctc_loss import CTCLoss
from ppocr.metrics.rec_metric import RecMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ppocr.postprocess.rec_postprocess import CTCLabelDecode
from ppocr.data.simple_dataset import SimpleDataSet
from ppocr.data.imaug.operators import DecodeImage, KeepKeys
from ppocr.data.imaug.rec_img_aug import RecResizeImg, RecAug
from ppocr.data.imaug.label_ops import CTCLabelEncode
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 500
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = "./output/rec_chinese_common_v2.0"
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
    save_res_path = "./output/rec/predicts_chinese_common_v2.0.txt"
    model_type = 'rec'
    algorithm = 'CRNN'
    Transform = None
    Backbone = _(ResNet, layers=34)
    Neck = _(SequenceEncoder, encoder_type="rnn", hidden_size=256)
    Head = _(CTCHead, fc_decay=4e-05)
    loss = CTCLoss()
    metric = RecMetric(main_indicator="acc")
    Optimizer = _(Adam,betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(CosineAnnealingWarmRestarts,T_0=5)
    PostProcessor = _(CTCLabelDecode,)
    class Train:
        Dataset = _(SimpleDataSet, data_dir="./train_data/", label_file_list=['./train_data/train_list.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), RecAug(), CTCLabelEncode(), RecResizeImg(image_shape=[3, 32, 320]), KeepKeys(keep_keys=['image', 'label', 'length'])]
        DATALOADER = _(shuffle=True, batch_size=256, drop_last=True, num_workers=8)
    class Eval:
        Dataset = _(SimpleDataSet, data_dir="./train_data/", label_file_list=['./train_data/val_list.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), CTCLabelEncode(), RecResizeImg(image_shape=[3, 32, 320]), KeepKeys(keep_keys=['image', 'label', 'length'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=256, num_workers=8)
