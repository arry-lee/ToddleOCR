__all__ = ["build_backbone"]

from torchvision.models import resnet50


def build_backbone(config, model_type):
    """Build backbone."""
    if model_type == "det":
        from .det_mobilenet_v3 import MobileNetV3
def build_backbone(config, model_type):
    if model_type == "det" or model_type == "table":
        from .det_mobilenet_v3 import MobileNetV3
        from .det_resnet import ResNet
        from .det_resnet_vd import ResNet_vd
        from .det_resnet_vd_sast import ResNet_SAST
        from .det_pp_lcnet import PPLCNet

        support_dict = ["MobileNetV3", "ResNet", "ResNet_vd", "ResNet_SAST", "PPLCNet"]
        if model_type == "table":
            from .table_master_resnet import TableResNetExtra

            support_dict.append("TableResNetExtra")
    elif model_type == "rec" or model_type == "cls":
        from .rec_mobilenet_v3 import MobileNetV3
        from .rec_resnet_vd import ResNet
        from .rec_resnet_fpn import ResNetFPN
        from .rec_mv1_enhance import MobileNetV1Enhance
        from .rec_nrtr_mtb import MTB
        from .rec_resnet_31 import ResNet31
        from .rec_resnet_32 import ResNet32
        from .rec_resnet_45 import ResNet45
        from .rec_resnet_aster import ResNet_ASTER
        from .rec_micronet import MicroNet
        from .rec_efficientb3_pren import EfficientNetB3_PREN
        from .rec_svtrnet import SVTRNet
        from .rec_vitstr import ViTSTR
        from .rec_resnet_rfl import ResNetRFL
        from .rec_densenet import DenseNet

        support_dict = [
            "MobileNetV1Enhance",
            "MobileNetV3",
            "ResNet",
            "ResNetFPN",
            "MTB",
            "ResNet31",
            "ResNet45",
            "ResNet_ASTER",
            "MicroNet",
            "EfficientNetB3_PREN",
            "SVTRNet",
            "ViTSTR",
            "ResNet32",
            "ResNetRFL",
            "DenseNet",
        ]
    elif model_type == "e2e":
        from .e2e_resnet_vd_pg import ResNet

        support_dict = ["ResNet"]
    elif model_type == "kie":
        from .kie_unet_sdmgr import Kie_backbone
        from .vqa_layoutlm import LayoutLMForSer, LayoutLMv2ForSer, LayoutLMv2ForRe, LayoutXLMForSer, LayoutXLMForRe

        support_dict = [
            "Kie_backbone",
            "LayoutLMForSer",
            "LayoutLMv2ForSer",
            "LayoutLMv2ForRe",
            "LayoutXLMForSer",
            "LayoutXLMForRe",
        ]
    elif model_type == "table":

        support_dict = ["ResNet", "MobileNetV3"]
    else:
        raise NotImplementedError

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type, support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
