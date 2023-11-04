from torch import nn


class BaseModel(nn.Module):
    def __init__(
        self, in_channels, backbone, neck, head, transform=None, return_all_feats=False
    ):
        super().__init__()
        if transform:
            self.transform = transform()
        if backbone:
            self.backbone = backbone(in_channels=in_channels)
            in_channels = self.backbone.out_channels
        if neck:
            self.neck = neck(in_channels=in_channels)
            in_channels = self.neck.out_channels
        if head:
            self.head = head(in_channels=in_channels)
        self.return_all_feats = return_all_feats

    def forward(self, x, **kwargs):
        out_dict = {}
        for module_name, module in self.named_children():
            if module is self.head:
                x = module(x, **kwargs)
            else:
                x = module(x)
        return x
        #     if isinstance(x, dict):
        #         out_dict.update(x)
        #         x[f"{module_name}_out"]= x.copy()
        #         # x = y
        #     else:
        #         out_dict[f"{module_name}_out"] = x
        # if self.return_all_feats:
        #     if self.training:
        #         return out_dict
        #     elif isinstance(x, dict):
        #         return x
        #     else:
        #         return {list(out_dict.keys())[-1]: x}
        # else:
        #     return x


class Tranformer:
    def encode(self, data):
        """输入转换器"""
        return data

    def decode(self, data):
        """输出转换器"""
        return data
