import torch
from torch import nn

from toddleocr.modules.backbones.rec_efficientb3_pren import Swish

__all__ = ["SequenceEncoder"]
from toddleocr.modules.backbones.rec_svtrnet import (
    Block,
    ConvBNLayer,
    ones_,
    trunc_normal_,
    zeros_,
)


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.permute(0, 2, 1)  # (NTC)(batch, width, channels)
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, num_layers=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        num_layers=1,
        dropout=0,
        direction=False,
        time_major=False,
        with_linear=False,
    ):
        super().__init__()
        self.with_linear = with_linear
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            direction=direction,
            time_major=time_major,
        )

        # text recognition the specified structure LSTM with linear
        if self.with_linear:
            self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_feature):
        recurrent, _ = self.rnn(
            input_feature
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        if self.with_linear:
            output = self.linear(recurrent)  # batch_size x T x output_size
            return output
        return recurrent


class EncoderWithCascadeRNN(nn.Module):
    def __init__(
        self, in_channels, hidden_size, out_channels, num_layers=2, with_linear=False
    ):
        super().__init__()
        self.out_channels = out_channels[-1]
        self.encoder = nn.ModuleList(
            [
                BidirectionalLSTM(
                    in_channels if i == 0 else out_channels[i - 1],
                    hidden_size,
                    output_size=out_channels[i],
                    num_layers=1,
                    # bidirectional=True,
                    with_linear=with_linear,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        for i, l in enumerate(self.encoder):
            x = l(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size

        self.fc = nn.Linear(in_channels, hidden_size, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        dims=64,
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        qk_scale=None,
    ):  # XS
        super().__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(
            in_channels, in_channels // 8, kernel_size=3, padding=1, act=Swish
        )
        self.conv2 = ConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act=Swish
        )

        self.svtr_block = nn.ModuleList(
            [
                Block(
                    dim=hidden_dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=Swish,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer="nn.LayerNorm",
                    eps=1e-05,
                    prenorm=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act=Swish)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(
            2 * in_channels, in_channels // 8, kernel_size=3, padding=1, act=Swish
        )

        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act=Swish)
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        # for use guide
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.concat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {
                "reshape": Im2Seq,
                "fc": EncoderWithFC,
                "rnn": EncoderWithRNN,
                "svtr": EncoderWithSVTR,
                "cascadernn": EncoderWithCascadeRNN,
            }
            assert encoder_type in support_encoder_dict, "{} must in {}".format(
                encoder_type, support_encoder_dict.keys()
            )
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs
                )
            elif encoder_type == "cascadernn":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size, **kwargs
                )
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size
                )
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != "svtr":
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x
