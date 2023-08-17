# This .py is auto generated by the script in the root folder.
from configs.config import ConfigModel,_
from torch.nn import MobileNetV3
from ptocr.modules.heads.table_att import TableAttentionHead
from ptocr.loss.table_att import TableAttentionLoss
from ptocr.metrics.table import TableMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
from ptocr.postprocess.table import TableLabelDecode
from ptocr.datasets.pubtab_dataset import PubTabDataSet
from ptocr.transforms.operators import ToCHWImage, DecodeImage, KeepKeys, NormalizeImage
from ptocr.transforms.label_ops import TableBoxEncode, TableLabelEncode
from ptocr.transforms.table_ops import PaddingTableImage, ResizeTableImage
class Model(ConfigModel):
    use_gpu = True
    epoch_num = 400
    log_window_size = 20
    log_batch_step = 5
    save_model_dir = "./output/table_mv3/"
    save_epoch_step = 400
    eval_batch_step = [0, 400]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    infer_img = "ppstructure/docs/table/table.jpg"
    save_res_path = "output/table_mv3"
    character_dict_path = "ppocr/utils/dict/table_structure_dict.txt"
    character_type = "en"
    max_text_length = 500
    box_format = "xyxy"
    infer_mode = False
    model_type = 'table'
    algorithm = 'TableAttn'
    Backbone = _(MobileNetV3, scale=1.0, model_name="small", disable_se=True)
    Head = _(TableAttentionHead, hidden_size=256, max_text_length=500, loc_reg_num=4)
    loss = TableAttentionLoss(structure_weight=100.0, loc_weight=10000.0)
    metric = TableMetric(main_indicator="acc", compute_bbox_metric=False)
    Optimizer = _(Adam,betas=[0.9, 0.999], clip_norm=5.0, lr=0.001)
    LRScheduler = _(ConstantLR,)
    PostProcessor = _(TableLabelDecode,)
    class Train:
        Dataset = _(PubTabDataSet, data_dir="train_data/table/pubtabnet/train/", label_file_list=['train_data/table/pubtabnet/PubTabNet_2.0.0_train.jsonl'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), TableLabelEncode(learn_empty_box=False, merge_no_span_structure=False, replace_empty_cell_token=False, loc_reg_num=4, max_text_length=500), TableBoxEncode(), ResizeTableImage(max_len=488), NormalizeImage(scale="1./255.", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order="hwc"), PaddingTableImage(size=[488, 488]), ToCHWImage(), KeepKeys(keep_keys=['image', 'structure', 'bboxes', 'bbox_masks', 'shape'])]
        DATALOADER = _(shuffle=True, batch_size=48, drop_last=True, num_workers=1)
    class Eval:
        Dataset = _(PubTabDataSet, data_dir="train_data/table/pubtabnet/val/", label_file_list=['train_data/table/pubtabnet/PubTabNet_2.0.0_val.jsonl'])
        transforms = _[DecodeImage(img_mode="BGR", channel_first=False), TableLabelEncode(learn_empty_box=False, merge_no_span_structure=False, replace_empty_cell_token=False, loc_reg_num=4, max_text_length=500), TableBoxEncode(), ResizeTableImage(max_len=488), NormalizeImage(scale="1./255.", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order="hwc"), PaddingTableImage(size=[488, 488]), ToCHWImage(), KeepKeys(keep_keys=['image', 'structure', 'bboxes', 'bbox_masks', 'shape'])]
        DATALOADER = _(shuffle=False, drop_last=False, batch_size=48, num_workers=1)
