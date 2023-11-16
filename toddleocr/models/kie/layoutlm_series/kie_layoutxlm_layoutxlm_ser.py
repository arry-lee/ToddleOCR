import os
import sys

sys.path.append(os.getcwd())

from toddleocr.config import _, ConfigModel
from toddleocr.datasets.simple import SimpleDataSet
from toddleocr.loss.vqa_token_layoutlm import VQASerTokenLayoutLMLoss
from toddleocr.metrics.vqa import VQASerTokenMetric
from toddleocr.modules.backbones.vqa_layoutlm import LayoutXLMForSer
from toddleocr.postprocess.vqa import VQASerTokenLayoutLMPostProcess
from toddleocr.transforms import VQATokenLabelEncode
from toddleocr.transforms.operators import (
    DecodeImage,
    KeepKeys,
    NormalizeImage,
    Resize,
    ToCHWImage,
)
from toddleocr.transforms.vqa.token.vqa_token_chunk import VQASerTokenChunk
from toddleocr.transforms.vqa.token.vqa_token_pad import VQATokenPad
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from toddleocr.config import PROJECT_DIR

print(PROJECT_DIR)
CLASS_PATH = os.path.join(PROJECT_DIR, "../train_data/XFUND/class_list_xfun.txt")
from toddleocr import ToddleOCR


class Model(ConfigModel):
    use_gpu = True
    epoch_num = 200
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
    Backbone = _(LayoutXLMForSer,
                 pretrained="D:/dev/.model/huggingface/layoutxlm-base",
                 checkpoints=None,
                 num_classes=7)
    loss = VQASerTokenLayoutLMLoss(num_classes=7, key="backbone_out")
    metric = VQASerTokenMetric(main_indicator="hmean")
    postprocessor = VQASerTokenLayoutLMPostProcess(
        class_path=CLASS_PATH
    )
    Optimizer = _(AdamW, beta1=0.9, beta2=0.999, lr=5e-05)
    LRScheduler = _(PolynomialLR, total_iters=200, warmup_epoch=2)

    class Data:
        dataset = SimpleDataSet
        root: "train_data/XFUND/zh_val/image" = "train_data/XFUND/zh_train/image"
        label_file_list: "train_data/XFUND/zh_val/val.json" = (
            "train_data/XFUND/zh_train/train.json"
        )

    class Loader:
        shuffle: False = True
        drop_last = False
        batch_size = 8
        num_workers = 4

    kie_det_model_dir = os.path.join(PROJECT_DIR, "../weights/zh_ocr_det_v3")
    kie_rec_model_dir = os.path.join(PROJECT_DIR, "../weights/zh_ocr_rec_v3")

    ocr_engine = ToddleOCR(
        det_model_dir=kie_det_model_dir,
        rec_model_dir=kie_rec_model_dir,
        use_gpu=use_gpu,
        use_angle_cls=False,
    )

    Transforms = _[
                 DecodeImage(img_mode="RGB", channel_first=False),
                 VQATokenLabelEncode(
                     contains_re=False,
                     algorithm="LayoutXLM",
                     class_path=CLASS_PATH,
                 ): ...:VQATokenLabelEncode(
                     contains_re=False,
                     algorithm="LayoutXLM",
                     class_path=CLASS_PATH,
                     ocr_engine=ocr_engine,
                     infer_mode=True,
                 ),
                 VQATokenPad(max_seq_len=512, return_attention_mask=True),
                 VQASerTokenChunk(max_seq_len=512),
                 Resize(size=[224, 224]),
                 NormalizeImage(
                     scale=1,
                     mean=[123.675, 116.28, 103.53],
                     std=[58.395, 57.12, 57.375],
                     order="hwc",
                 ),
                 ToCHWImage(),
                 KeepKeys(
                     "input_ids", "bbox", "attention_mask", "token_type_ids", "image", "labels"
                 ): ...: KeepKeys('input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                                  'image', 'labels', 'segment_offset_id', 'ocr_info',
                                  'entities'),
                 ]

    # @torch.no_grad()
    # def ser_one_image(self, img_or_path,output=None):
    #
    #     if isinstance(img_or_path, str):
    #         img = cv2.imread(img_or_path)
    #
    #     self.model.eval()
    #     data = {'image': img}
    #     batch = self.transforms("infer")(data)  # 可以在此处引入OCR_ENGINE
    #     batch = to_tensor(batch)
    #
    #     preds = self.model(batch)
    #     print(preds)
    #     post_result = self.postprocessor(preds, segment_offset_ids=batch[6], ocr_infos=batch[7])
    #     print(post_result)
    #     if output:
    #         img_res = draw_ser_results(img_or_path, post_result[0])
    #         cv2.imwrite(output, img_res)
    #     return post_result


def _t():
    global endswith
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
        pd = paddle.load(pdmodel)
        maps = {'._mean': '.running_mean',
                '._variance': '.running_var',
                'layoutxlm': 'backbone.model.layoutlmv2', # todo,simplify the prefix of backbone
                'classifier': 'backbone.model.classifier'
                }
        transpose = 'weight'
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

        torch.save(new, fr'{pdmodel.split(".")[0]}.pt')

    transmodel("D:\dev\github\ToddleOCR\model\ser_LayoutXLM_xfun_zh\model_state.pdparams",
               ("classifier.weight", "query.weight", "key.weight", "value.weight", "dense.weight","visual_proj.weight"
                ))


if __name__ == '__main__':
    # m = LayoutXLMForSer(7, pretrained="D:/dev/.model/huggingface/layoutxlm-base")
    # m.load_state_dict(torch.load("D:\dev\github\ToddleOCR\model\ser_LayoutXLM_xfun_zh\model_state.pt"))

    ## todo 推理测试
    # for k in m.state_dict().keys():
    #     print(k)
    #
    # print(len(m.state_dict()))
    #
    # import paddle
    #
    # pd = paddle.load("D:\dev\github\ToddleOCR\model\ser_LayoutXLM_xfun_zh\model_state.pdparams")
    # print(len(pd))
    # for k in pd.keys():
    #     print(k)
    # _t()
    m = Model(pretrained="D:\dev\github\ToddleOCR\model\ser_LayoutXLM_xfun_zh\model_state.pt")
    m.ser_one_image("D:\dev\github\ToddleOCR\docs\imgs\zh_val_42.jpg",output='output.jpg')
