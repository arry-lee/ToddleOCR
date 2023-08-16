# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from ppocr.models.backbones.rec_resnet_45 import ResNet45
from ppocr.models.heads.rec_visionlan_head import VLHead
from ppocr.losses.rec_vl_loss import VLLoss
from ppocr.metrics.rec_metric import RecMetric
from torch.optim import Adam
from ppocr.optimizer.lr_scheduler import PiecewiseLR
from ppocr.postprocess.rec_postprocess import VLLabelDecode
from ppocr.data.lmdb_dataset import LMDBDataSet
from ppocr.data.imaug.operators import DecodeImage, KeepKeys
from ppocr.data.imaug.rec_img_aug import ABINetRecAug, VLRecResizeImg
from ppocr.data.imaug.label_ops import VLLabelEncode
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 8
    log_window_size = 200
    log_batch_step = 200
    save_model_dir = "./output/rec/r45_visionlan"
    save_epoch_step = 1
    eval_batch_step = [0, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = True
    infer_img = "doc/imgs_words/en/word_2.png"
    character_dict_path = None
    max_text_length = 25
    training_step = "LA"
    infer_mode = False
    use_space_char = False
    save_res_path = "./output/rec/predicts_visionlan.txt"
    model_type = 'rec'
    algorithm = 'VisionLAN'
    Transform = None
    Backbone = _(ResNet45, strides=[2, 2, 2, 1, 1])
    Head = _(VLHead, n_layers=3, n_position=256, n_dim=512, max_text_length=25, training_step="LA")
    loss = VLLoss(mode="LA", weight_res=0.5, weight_mas=0.5)
    metric = RecMetric(is_filter=True)
    Optimizer = _(Adam,betas=[0.9, 0.999], clip_norm=20.0, group_lr=True, training_step="LA")
    LRScheduler = _(PiecewiseLR,decay_epochs=[6], values=[0.0001, 1e-05])
    PostProcessor = _(VLLabelDecode,)
    class Train:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/training/")
        transforms = _[DecodeImage(img_mode="RGB", channel_first=False), ABINetRecAug(), VLLabelEncode(), VLRecResizeImg(image_shape=[3, 64, 256]), KeepKeys(keep_keys=['image', 'label', 'label_res', 'label_sub', 'label_id', 'length'])]
        DATALOADER = _(shuffle=True, batch_size=220, drop_last=True, num_workers=4)
    class Eval:
        Dataset = _(LMDBDataSet, data_dir="./train_data/data_lmdb_release/validation/")
        transforms = _[DecodeImage(img_mode="RGB", channel_first=False), VLLabelEncode(), VLRecResizeImg(image_shape=[3, 64, 256]), KeepKeys(keep_keys=['image', 'label', 'label_res', 'label_sub', 'label_id', 'length'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=64, num_workers=4)