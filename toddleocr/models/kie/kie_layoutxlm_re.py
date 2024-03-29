#  Copyright (c) 2023. Arry Lee, <arry_lee@qq.com>

from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from toddleocr._appdir import DATASET_DIR, MODEL_DIR
from toddleocr.config import _, ConfigModel
from toddleocr.datasets.simple import SimpleDataSet
from toddleocr.loss.basic import LossFromOutput
from toddleocr.metrics.vqa import VQAReTokenMetric
from toddleocr.modules.backbones.vqa_layoutlm import LayoutXLMForRe
from toddleocr.postprocess.vqa import VQAReTokenLayoutLMPostProcess
from toddleocr.transforms import VQATokenLabelEncode
from toddleocr.transforms.operators import (
    DecodeImage,
    KeepKeys,
    NormalizeImage,
    Resize,
    ToCHWImage,
)
from toddleocr.transforms.vqa.token.vqa_re_convert import (
    TensorizeEntitiesRelations,
)
from toddleocr.transforms.vqa.token.vqa_token_chunk import VQAReTokenChunk
from toddleocr.transforms.vqa.token.vqa_token_pad import VQATokenPad
from toddleocr.transforms.vqa.token.vqa_token_relation import VQAReTokenRelation

CLASS_PATH = DATASET_DIR / "XFUND/class_list_xfun.txt"


class Model(ConfigModel):
    use_gpu = True
    epoch_num = 130
    log_window_size = 10
    log_batch_step = 10
    save_model_dir = None
    save_epoch_step = 2000
    eval_batch_step = [0, 19]
    metric_during_train = False
    save_infer_dir = None
    use_visualdl = False
    seed = 2022
    pretrained_model = None
    model_type = "kie"
    algorithm = "LayoutXLM"
    Transform = None
    Backbone = _(
        LayoutXLMForRe,
        pretrained=MODEL_DIR/"layoutxlm-base",
        checkpoints=None,
    )
    loss = LossFromOutput(key="loss", reduction="mean")
    metric = VQAReTokenMetric(main_indicator="hmean")
    postprocessor = VQAReTokenLayoutLMPostProcess()
    Optimizer = _(AdamW, beta1=0.9, beta2=0.999, clip_norm=10, lr=5e-05)
    LRScheduler = _(ConstantLR, warmup_epoch=10)

    class Data:
        dataset = SimpleDataSet
        root: "XFUND/zh_val/image" = "XFUND/zh_train/image"
        label_file_list: "XFUND/zh_val/val.json" = "XFUND/zh_train/train.json"

    class Loader:
        shuffle: False = True
        drop_last = False
        batch_size: 8 = 2
        num_workers = 8

    Transforms = _[
        DecodeImage(),
        VQATokenLabelEncode(
            contains_re=True, class_path=CLASS_PATH
        ) : ... : VQATokenLabelEncode(
            contains_re=True, class_path=CLASS_PATH, infer_mode=True
        ),
        VQATokenPad(),
        VQAReTokenRelation(),
        VQAReTokenChunk(),
        TensorizeEntitiesRelations(),
        Resize(size=[224, 224]),
        NormalizeImage(
            scale=1,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            order="hwc",
        ),
        ToCHWImage(),
        KeepKeys(
            "input_ids",
            "bbox",
            "attention_mask",
            "token_type_ids",
            "image",
            "entities",
            "relations",
        ) : ...,
    ]


def _t():
    import torch
    import paddle

    def endswith(p, ls):
        for s in ls:
            if p.endswith(s):
                return True
        return False

    def transmodel(pdmodel, linear_suffix=()):
        def p2t(tensor) -> torch.Tensor:
            return torch.from_numpy(tensor.numpy())

        # global transpose
        pdmodel = str(pdmodel)
        pd = paddle.load(pdmodel)
        maps = {
            "._mean": ".running_mean",
            "._variance": ".running_var",
            "layoutxlm": "backbone.model.layoutxlm",  # todo,simplify the prefix of backbone
            # 'classifier': 'backbone.model.classifier',
            "extractor": "backbone.model.extractor",
        }
        transpose = "weight"
        new = {}
        for k, v in pd.items():
            tk = k
            for key in maps:
                if key in k:
                    tk = tk.replace(key, maps[key])
            new[tk] = p2t(v)

        for tk in new.keys():
            if tk.endswith(transpose):
                if endswith(tk, linear_suffix):
                    new[tk] = new[tk].T

        torch.save(new, rf'{pdmodel.split(".")[0]}.pt')

    transmodel(
        MODEL_DIR / "re_LayoutXLM_xfun_zh/model_state.pdparams",
        (
            "classifier.weight",
            "query.weight",
            "key.weight",
            "value.weight",
            "dense.weight",
            "visual_proj.weight",
            ".linear.weight",
            "ffnn_head.0.weight",
            "ffnn_head.3.weight",
            "ffnn_tail.0.weight",
            "ffnn_tail.3.weight",
        ),
    )


# if __name__ == '__main__':
# _t()

# for k in m.model.state_dict():
#     print(k)
