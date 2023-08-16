# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from ppocr.models.backbones.rec_mv1_enhance import MobileNetV1Enhance
from ppocr.models.heads.rec_multi_head import MultiHead
from ppocr.losses.rec_multi_loss import MultiLoss
from ppocr.metrics.rec_metric import RecMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ppocr.postprocess.rec_postprocess import CTCLabelDecode
from ppocr.data.simple_dataset import SimpleDataSet
from ppocr.data.imaug.operators import DecodeImage, KeepKeys
from ppocr.data.imaug.rec_img_aug import RecResizeImg, RecConAug, RecAug
from ppocr.data.imaug.label_ops import MultiLabelEncode
class Model(ConfigModel):
    debug = False
    use_gpu = True
    epoch_num = 500
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = "./output/v3_te_mobile"
    save_epoch_step = 3
    eval_batch_step = [0, 2000]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "doc/imgs_words/ch/word_1.jpg"
    character_dict_path = "ppocr/utils/dict/te_dict.txt"
    max_text_length = 25
    infer_mode = False
    use_space_char = True
    distributed = True
    save_res_path = "./output/rec/predicts_ppocrv3_te.txt"
    model_type = 'rec'
    algorithm = 'SVTR'
    Transform = None
    Backbone = _(MobileNetV1Enhance, scale=0.5, last_conv_stride=[1, 2], last_pool_type="avg")
    Head = _(MultiHead, head_list=[{'class': 'CTCHead', 'Neck': {'class': 'svtr', 'dims': 64, 'depth': 2, 'hidden_dims': 120, 'use_guide': True}, 'Head': {'fc_decay': 1e-05}}, {'class': 'SARHead', 'enc_dim': 512, 'max_text_length': 25}])
    loss = MultiLoss(loss_config_list=[{'CTCLoss': None}, {'SARLoss': None}])
    metric = RecMetric(main_indicator="acc", ignore_space=False)
    Optimizer = _(Adam,betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(CosineAnnealingWarmRestarts,T_0=5)
    PostProcessor = _(CTCLabelDecode,)
    class Train:
        Dataset = _(SimpleDataSet, data_dir="./train_data/", ext_op_transform_idx=1, label_file_list=['./train_data/train_list.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), RecConAug(prob=0.5, ext_data_num=2, image_shape=[48, 320, 3]), RecAug(), MultiLabelEncode(), RecResizeImg(image_shape=[3, 48, 320]), KeepKeys(keep_keys=['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio'])]
        DATALOADER = _(shuffle=True, batch_size=128, drop_last=True, num_workers=4)
    class Eval:
        Dataset = _(SimpleDataSet, data_dir="./train_data", label_file_list=['./train_data/val_list.txt'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), MultiLabelEncode(), RecResizeImg(image_shape=[3, 48, 320]), KeepKeys(keep_keys=['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=128, num_workers=4)
