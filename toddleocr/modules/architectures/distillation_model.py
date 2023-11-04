from torch import nn

from .base_model import BaseModel

__all__ = ["DistillationModel"]


class DistillationModel(nn.Module):
    def __init__(self, config):
        """
        the module for OCR distillation.
        args:
            config (dict): the super parameters for module.
        """
        super().__init__()
        self.model_list = []
        self.model_name_list = []
        for key in config["Models"]:
            model_config = config["Models"][key]
            freeze_params = False
            pretrained = None
            if "freeze_params" in model_config:
                freeze_params = model_config.pop("freeze_params")
            if "pretrained" in model_config:
                pretrained = model_config.pop("pretrained")
            model = BaseModel(model_config)
            # if pretrained is not None:
            #     load_pretrained_params(model, pretrained)
            if freeze_params:
                for param in model.parameters():
                    param.trainable = False
            self.model_list.append(self.add_module(key, model))
            self.model_name_list.append(key)

    def forward(self, x, data=None):
        result_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            result_dict[model_name] = self.model_list[idx](x, data)
        return result_dict
