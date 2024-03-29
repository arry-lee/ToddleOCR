# This .pyi is auto generated by the script in the root folder.
# only for cache,use .py for changes
from toddleocr.config import _, ConfigModel
from toddleocr.datasets.pubtab import PubTabDataSet
from toddleocr.loss.table_att import SLALoss
from toddleocr.metrics.table import TableMetric
from toddleocr.modules.backbones.det_pp_lcnet import PPLCNet
from toddleocr.modules.heads.table_att import SLAHead
from toddleocr.modules.necks.csp_pan import CSPPAN
from toddleocr.postprocess.table import TableLabelDecode
from toddleocr.transforms.label_ops import TableBoxEncode, TableLabelEncode
from toddleocr.transforms.operators import (
    DecodeImage,
    KeepKeys,
    NormalizeImage,
    ToCHWImage,
)
from toddleocr.transforms.table_ops import PaddingTableImage, ResizeTableImage
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR

class Model(ConfigModel):
    use_gpu = True
    epoch_num = 400
    log_window_size = 20
    log_batch_step = 20
    save_model_dir = None
    save_epoch_step = 400
    eval_batch_step = [0, 331]
    metric_during_train = True
    pretrained_model = None
    checkpoints = None
    save_infer_dir = "./output/SLANet_ch/infer"
    use_visualdl = False
    character_dict_path = "ppocr/utils/dict/table_structure_dict_ch.txt"
    character_type = "en"
    max_text_length = 500
    box_format = "xy4"
    infer_mode = False
    use_sync_bn = True
    model_type = "tab"
    algorithm = "SLANet"
    Backbone = _(PPLCNet, scale=1.0, pretrained=False, use_ssld=True)
    Neck = _(CSPPAN, out_channels=96)
    Head = _(SLAHead, hidden_size=256, max_text_length=500, loc_reg_num=8)
    loss = SLALoss(structure_weight=1.0, loc_weight=2.0, loc_loss="smooth_l1")
    metric = TableMetric(
        main_indicator="acc",
        compute_bbox_metric=False,
        loc_reg_num=8,
        box_format="xy4",
        del_thead_tbody=True,
    )
    postprocessor = TableLabelDecode(merge_no_span_structure=True)
    Optimizer = _(Adam, betas=[0.9, 0.999], clip_norm=5.0, lr=0.001)
    LRScheduler = _(
        ConstantLR,
    )

    class Data:
        dataset = PubTabDataSet
        root: "train_data/table/val/" = "train_data/table/train/"
        label_file_list: "train_data/table/val.txt" = "train_data/table/train.txt"

    class Loader:
        shuffle: False = True
        drop_last: False = True
        batch_size = 48
        num_workers = 1
    Transforms = _[
        DecodeImage(img_mode="BGR", channel_first=False),
        TableLabelEncode(
            learn_empty_box=False,
            merge_no_span_structure=True,
            replace_empty_cell_token=False,
            loc_reg_num=8,
            max_text_length=500,
        ) : ...,
        TableBoxEncode(in_box_format="xy4", out_box_format="xy4"),
        ResizeTableImage(max_len=488),
        NormalizeImage(
            scale="1./255.",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order="hwc",
        ),
        PaddingTableImage(size=[488, 488]),
        ToCHWImage(),
        KeepKeys("image", "structure", "bboxes", "bbox_masks", "shape") : ...,
    ]
