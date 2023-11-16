# coding=utf-8
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from .backbone import build_resnet_fpn_backbone

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMIntermediate as LayoutLMv2Intermediate
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMOutput as LayoutLMv2Output
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMPooler as LayoutLMv2Pooler
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMSelfOutput as LayoutLMv2SelfOutput
from transformers.utils import logging

from ...modules.decoders.re import REDecoder
from ...utils import ReOutput
from .configuration_layoutlmv2 import LayoutLMv2Config

PIXEL_STD = [57.375, 57.120, 58.395]

PIXEL_MEAN = [103.530, 116.280, 123.675]

logger = logging.get_logger(__name__)

LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlmv2-base-uncased",
    "layoutlmv2-large-uncased",
]


LayoutLMv2LayerNorm = torch.nn.LayerNorm


class LayoutLMv2Embeddings(nn.Module):
    """
    首先，构造函数__init__中初始化了各种嵌入层，包括词嵌入、位置嵌入、二维位置嵌入、标记类型嵌入等。这些嵌入层将用于将输入的文本和布局信息转换为对应的向量表示。

    然后，_cal_spatial_position_embeddings方法根据给定的边界框(bbox)计算空间位置嵌入。
    空间位置嵌入包括左侧位置嵌入、上方位置嵌入、右侧位置嵌入、底部位置嵌入、高度位置嵌入和宽度位置嵌入。
    其中，左、上、右、底位置嵌入使用二维位置嵌入层进行计算，高度和宽度位置嵌入使用不同的嵌入层进行计算。
    最后，_cal_spatial_position_embeddings方法将计算得到的空间位置嵌入拼接在一起，并返回该结果。

    这段代码的作用是将输入的文本和布局信息转化为对应的嵌入表示，以供模型后续的处理和预测。

    这段代码都没有定义 forward 方法，为什么是 nn.Modules


    每一个嵌入表示都有一个参数矩阵要学习

    """

    def __init__(self, config):
        super(LayoutLMv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayoutLMv2LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        # 多余信息
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings


class LayoutLMv2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool), float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LayoutLMv2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LayoutLMv2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2Encoder(nn.Module):
    """
    编码器， Transformer 中最重要的东西
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias: # 相对位置偏置
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias: # 空间位置偏置
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LayoutLMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv2Config
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlmv2"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayoutLMv2LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

from .backbone import FrozenBatchNorm2d
def my_convert_sync_batchnorm(module, process_group=None):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    module_output = module

    if isinstance(module, FrozenBatchNorm2d):
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        module_output.weight = torch.nn.Parameter(module.weight)
        module_output.bias = torch.nn.Parameter(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=module.running_mean.device)
    for name, child in module.named_children():
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
    del module
    return module_output

def add_layoutlmv2_config(cfg):
    _C = cfg
    # -----------------------------------------------------------------------------
    # Config definition
    # -----------------------------------------------------------------------------
    _C.MODEL.MASK_ON = True

    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    _C.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]

    # ---------------------------------------------------------------------------- #
    # Backbone options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"

    # ---------------------------------------------------------------------------- #
    # FPN options
    # ---------------------------------------------------------------------------- #
    # Names of the input feature maps to be used by FPN
    # They must have contiguous power of 2 strides
    # e.g., ["res2", "res3", "res4", "res5"]
    _C.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]

    # ---------------------------------------------------------------------------- #
    # Anchor generator options
    # ---------------------------------------------------------------------------- #
    # Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
    # Format: list[list[float]]. SIZES[i] specifies the list of sizes
    # to use for IN_FEATURES[i]; len(SIZES) == len(IN_FEATURES) must be true,
    # or len(SIZES) == 1 is true and size list SIZES[0] is used for all
    # IN_FEATURES.
    _C.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]

    # ---------------------------------------------------------------------------- #
    # RPN options
    # ---------------------------------------------------------------------------- #
    # Names of the input feature maps to be used by RPN
    # e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
    _C.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    # Number of top scoring RPN proposals to keep before applying NMS
    # When FPN is used, this is *per FPN level* (not total)
    _C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    _C.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    # Number of top scoring RPN proposals to keep after applying NMS
    # When FPN is used, this limit is applied per level and then again to the union
    # of proposals from all levels
    # NOTE: When FPN is used, the meaning of this config is different from Detectron1.
    # It means per-batch topk in Detectron1, but per-image topk here.
    # See the "find_top_rpn_proposals" function for details.
    _C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    _C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    # ---------------------------------------------------------------------------- #
    # ROI HEADS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    # Number of foreground classes
    _C.MODEL.ROI_HEADS.NUM_CLASSES = 5
    # Names of the input feature maps to be used by ROI heads
    # Currently all heads (box, mask, ...) use the same input feature map list
    # e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
    _C.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    # ---------------------------------------------------------------------------- #
    # Box Head
    # ---------------------------------------------------------------------------- #
    # C4 don't use head name option
    # Options for non-C4 models: FastRCNNConvFCHead,
    _C.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    _C.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    _C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14

    # ---------------------------------------------------------------------------- #
    # Mask Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    _C.MODEL.ROI_MASK_HEAD.NUM_CONV = 4  # The number of convs in the mask head
    _C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 7

    # ---------------------------------------------------------------------------- #
    # ResNe[X]t options (ResNets = {ResNet, ResNeXt}
    # Note that parts of a resnet may be used for both the backbone and the head
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RESNETS.DEPTH = 101
    _C.MODEL.RESNETS.SIZES = [[32], [64], [128], [256], [512]]
    _C.MODEL.RESNETS.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    _C.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]  # res4 for C4 backbone, res2..5 for FPN backbone

    # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
    _C.MODEL.RESNETS.NUM_GROUPS = 32

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.RESNETS.WIDTH_PER_GROUP = 8

    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    _C.MODEL.RESNETS.STRIDE_IN_1X1 = False
class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_resnet_fpn_backbone(norm='')
        if (
            config.convert_sync_batchnorm
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            self_rank = torch.distributed.get_rank()
            node_size = torch.cuda.device_count()
            world_size = torch.distributed.get_world_size()
            assert world_size % node_size == 0

            node_global_ranks = [
                list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)
            ]
            sync_bn_groups = [
                torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
            ]
            node_rank = self_rank // node_size
            assert self_rank in node_global_ranks[node_rank]

            self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])

        # assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = 3 #len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(PIXEL_MEAN).view(num_channels, 1, 1),
        )
        self.register_buffer("pixel_std", torch.Tensor(PIXEL_STD).view(num_channels, 1, 1))
        self.out_feature_key = "p2"
        if torch.is_deterministic_algorithms_warn_only_enabled():
            logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def forward(self, images):      
        images_input = ((images if torch.is_tensor(images) else images.tensor) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features


class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
    """LayoutLMv2的核心

    初始化了一系列模型组件，包括嵌入层、视觉处理部分、编码器、池化器等，并调用了init_weights方法来初始化模型权重。
    """
    def __init__(self, config):
        super(LayoutLMv2Model, self).__init__(config)
        self.config = config
        self.use_visual_backbone = config.use_visual_backbone

        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        # 嵌入层
        self.embeddings = LayoutLMv2Embeddings(config)
        # 视觉处理部分
        if self.use_visual_backbone:
            self.visual = VisualBackbone(config)
            self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        if self.has_visual_segment_embedding: # 是否使用视觉嵌入
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        # 层归一化
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)
        # 编码器
        self.encoder = LayoutLMv2Encoder(config)
        # 池化器
        self.pooler = LayoutLMv2Pooler(config)
        # 初始化
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        # 计算文本的嵌入表示

        seq_length = input_ids.size(1)

        # 如果未指定位置索引，则创建默认的位置索引
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # 如果未指定标记类型索引，则创建默认的标记类型索引（全为0）
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 计算词嵌入
        words_embeddings = self.embeddings.word_embeddings(input_ids)

        # 计算位置嵌入
        position_embeddings = self.embeddings.position_embeddings(position_ids)

        # 计算空间位置嵌入
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)

        # 计算标记类型嵌入，区分句子A和句子B
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        # 将各种嵌入相加得到最终的嵌入表示
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings

        # 应用层归一化和dropout操作
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)

        return embeddings


    def _calc_img_embeddings(self, image, bbox, position_ids):
        # 计算图像的嵌入表示



        # 计算位置嵌入
        position_embeddings = self.embeddings.position_embeddings(position_ids)

        # 计算空间位置嵌入
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)

        # 将视觉嵌入、位置嵌入和空间位置嵌入相加得到最终的嵌入表示
        embeddings =  position_embeddings + spatial_position_embeddings
        # 计算视觉嵌入
        if self.use_visual_backbone:
            visual_embeddings = self.visual_proj(self.visual(image)) # todo 使用其他
            embeddings = embeddings + visual_embeddings
        # 如果模型包含视觉分段嵌入，则将其加到嵌入表示上
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding

        # 应用层归一化和dropout操作
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)

        return embeddings


    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        模型的前向传播方法。

        参数:
            input_ids (torch.Tensor, optional): 输入序列的ID。形状: (batch_size, sequence_length)。
            bbox (torch.Tensor, optional): 边界框坐标。形状: (batch_size, sequence_length, 4)。
            image (torch.Tensor, optional): 图像特征。形状: (batch_size, image_feature_shape)。
            attention_mask (torch.Tensor, optional): 注意力遮罩。形状: (batch_size, sequence_length)。
            token_type_ids (torch.Tensor, optional): 标记类型ID。形状: (batch_size, sequence_length)。
            position_ids (torch.Tensor, optional): 位置ID。形状: (batch_size, sequence_length)。
            head_mask (torch.Tensor, optional): 头部遮罩。形状: (num_hidden_layers, num_attention_heads)。
            inputs_embeds (torch.Tensor, optional): 嵌入输入。形状: (batch_size, sequence_length, embedding_size)。
            encoder_hidden_states (torch.Tensor, optional): 编码器隐藏状态。
            encoder_attention_mask (torch.Tensor, optional): 编码器注意力遮罩。
            output_attentions (bool, optional): 是否输出注意力。
            output_hidden_states (bool, optional): 是否输出隐藏状态。
            return_dict (bool, optional): 是否返回字典作为输出。

        返回:
            BaseModelOutputWithPoolingAndCrossAttentions 或 tuple(torch.Tensor): 模型输出。如果 `return_dict` 为 True，
            则返回一个命名元组，包括last_hidden_state、pooler_output、hidden_states、attentions和cross_attentions。
            如果 `return_dict` 为 False，则返回一个元组，包括sequence_output、pooled_output和剩余的编码器输出。
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]#视觉形状是特征的长乘宽。
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]#最终形状是文字的长度和视觉特征的长度之和。
        final_shape = torch.Size(final_shape)

        visual_bbox_x = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[1]
        )
        visual_bbox_y = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[0]
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))
        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            # """
            # attention_mask的作用是在进行注意力计算时，对输入进行掩码操作，以控制模型在哪些位置上需要进行注意力计算，哪些位置上不需要。
            # 在这段代码中，首先判断attention_mask是否为None，如果是None，则通过torch.ones函数创建一个形状与input_shape相同的张量，所有元素的值均为1。这个新创建的张量即为注意力掩码。
            # 注意力掩码一般用于处理可变长度的序列输入。例如，对于一句话的文本序列，如果某个位置是填充字符，则在计算注意力时不希望模型将注意力放在该位置上，
            # 因为填充字符并不包含有意义的信息。通过使用注意力掩码，可以将填充字符的位置掩盖(mask)，使模型在计算注意力时忽略这些位置。这样可以提高模型的效率和准确性。
            # 总结起来，attention_mask的作用是控制模型在注意力计算过程中对输入的注意力权重的分配，从而有效地处理输入中的填充或无关信息。
            # """
        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand_as(input_ids)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # 这行代码的含义是将extended_attention_mask中的1变为0，0
        # 变为 - 10000。这是因为在注意力计算中，通常会使用非常小的负数作为掩码，以便在softmax计算中使得掩码位置的权重趋近于0。乘以 - 10000
        # 是为了让掩码位置的权重趋近于负无穷，这样在经过softmax计算后，其权重就会趋近于0，达到了屏蔽掩码位置的效果。

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        # head_mask 在这里被用来控制多头注意力机制中每个头的注意力权重，在模型计算注意力时起到了控制作用。
        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    base_model = LayoutLMv2Model

    def __init__(self, config):

        super().__init__(config)
        self.num_labels = config.num_labels
        # self.layoutlmv2 = LayoutLMv2Model(config) # todo layoutxlm
        self.register_module(self.base_model_prefix,self.base_model(config))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = getattr(self,self.base_model_prefix)(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMv2ForRelationExtraction(LayoutLMv2PreTrainedModel):
    base_model = LayoutLMv2Model

    def __init__(self, config):
        super().__init__(config)
        self.register_module(self.base_model_prefix,self.base_model(config))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = REDecoder(config)
        self.init_weights()


    def forward(
        self,
        input_ids,
        bbox,
        labels=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        entities=None,
        relations=None,
    ):

        outputs = getattr(self,self.base_model_prefix)(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)

        return ReOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
            hidden_states=outputs[0],
        )
