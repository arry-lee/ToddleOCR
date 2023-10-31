"""
This code is refer from: 
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/encoders/channel_reduction_encoder.py
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/robust_scanner_decoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RobustScannerHead"]


class BaseDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(
        self,
        feat,
        out_enc,
        label=None,
        valid_ratios=None,
        word_positions=None,
        train_mode=True,
    ):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(
                feat, out_enc, label, valid_ratios, word_positions
            )
        return self.forward_test(feat, out_enc, valid_ratios, word_positions)


class ChannelReductionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        nn.init.xavier_normal_(self.layer.weight, gain=1.0)

    def forward(self, feat):
        return self.layer(feat)


def masked_fill(x, mask, value):
    y = torch.full(x.shape, value, dtype=x.dtype)
    return torch.where(mask, y, x)


class DotProductAttentionLayer(nn.Module):
    def __init__(self, dim_model=None):
        super().__init__()
        self.scale = dim_model ** (-0.5) if dim_model is not None else 1.0

    def forward(self, query, key, value, h, w, valid_ratios=None):
        query = query.permute(0, 2, 1)
        logits = torch.matmul(query, key) * self.scale
        (n, c, t) = logits.shape
        logits = torch.reshape(logits, [n, c, h, w])
        if valid_ratios is not None:
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, int(w * valid_ratio + 0.5))
                if valid_width < w:
                    logits[i, :, :, valid_width:] = float("-inf")
        logits = torch.reshape(logits, [n, c, t])
        weights = F.softmax(logits, dim=2)
        value = value.permute(0, 2, 1)
        glimpse = torch.matmul(weights, value)
        glimpse = glimpse.permute(0, 2, 1)
        return glimpse


class SequenceAttentionDecoder(BaseDecoder):
    def __init__(
        self,
        num_classes=None,
        rnn_layers=2,
        dim_input=512,
        dim_model=128,
        max_seq_len=40,
        start_idx=0,
        mask=True,
        padding_idx=None,
        dropout=0,
        return_feature=False,
        encode_value=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.max_seq_len = max_seq_len
        self.start_idx = start_idx
        self.mask = mask
        self.embedding = nn.Embedding(
            self.num_classes, self.dim_model, padding_idx=padding_idx
        )
        self.sequence_layer = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            time_major=False,
            dropout=dropout,
        )
        self.attention_layer = DotProductAttentionLayer()
        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input, pred_num_classes
            )

    def forward_train(self, feat, out_enc, targets, valid_ratios):
        tgt_embedding = self.embedding(targets)
        (n, c_enc, h, w) = out_enc.shape
        assert c_enc == self.dim_model
        (_, c_feat, _, _) = feat.shape
        assert c_feat == self.dim_input
        (_, len_q, c_q) = tgt_embedding.shape
        assert c_q == self.dim_model
        assert len_q <= self.max_seq_len
        (query, _) = self.sequence_layer(tgt_embedding)
        query = query.permute(0, 2, 1)
        key = torch.reshape(out_enc, [n, c_enc, h * w])
        if self.encode_value:
            value = key
        else:
            value = torch.reshape(feat, [n, c_feat, h * w])
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        attn_out = attn_out.permute(0, 2, 1)
        if self.return_feature:
            return attn_out
        out = self.prediction(attn_out)
        return out

    def forward_test(self, feat, out_enc, valid_ratios):
        seq_len = self.max_seq_len
        batch_size = feat.shape[0]
        decode_sequence = (
            torch.ones((batch_size, seq_len), dtype=torch.int64) * self.start_idx
        )
        outputs = []
        for i in range(seq_len):
            step_out = self.forward_test_step(
                feat, out_enc, decode_sequence, i, valid_ratios
            )
            outputs.append(step_out)
            max_idx = torch.argmax(step_out, dim=1, keepdim=False)
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx
        outputs = torch.stack(outputs, 1)
        return outputs

    def forward_test_step(
        self, feat, out_enc, decode_sequence, current_step, valid_ratios
    ):
        embed = self.embedding(decode_sequence)
        (n, c_enc, h, w) = out_enc.shape
        assert c_enc == self.dim_model
        (_, c_feat, _, _) = feat.shape
        assert c_feat == self.dim_input
        (_, _, c_q) = embed.shape
        assert c_q == self.dim_model
        (query, _) = self.sequence_layer(embed)
        query = query.permute(0, 2, 1)
        key = torch.reshape(out_enc, [n, c_enc, h * w])
        if self.encode_value:
            value = key
        else:
            value = torch.reshape(feat, [n, c_feat, h * w])
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        out = attn_out[:, :, current_step]
        if self.return_feature:
            return out
        out = self.prediction(out)
        out = F.softmax(out, dim=-1)
        return out


class PositionAwareLayer(nn.Module):
    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()
        self.dim_model = dim_model
        self.rnn = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            time_major=False,
        )
        self.mixer = nn.Sequential(
            nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, img_feature):
        (n, c, h, w) = img_feature.shape
        rnn_input = img_feature.permute(0, 2, 3, 1)
        rnn_input = torch.reshape(rnn_input, (n * h, w, c))
        (rnn_output, _) = self.rnn(rnn_input)
        rnn_output = torch.reshape(rnn_output, (n, h, w, c))
        rnn_output = rnn_output.permute(0, 3, 1, 2)
        out = self.mixer(rnn_output)
        return out


class PositionAttentionDecoder(BaseDecoder):
    def __init__(
        self,
        num_classes=None,
        rnn_layers=2,
        dim_input=512,
        dim_model=128,
        max_seq_len=40,
        mask=True,
        return_feature=False,
        encode_value=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.mask = mask
        self.embedding = nn.Embedding(self.max_seq_len + 1, self.dim_model)
        self.position_aware_module = PositionAwareLayer(self.dim_model, rnn_layers)
        self.attention_layer = DotProductAttentionLayer()
        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input, pred_num_classes
            )

    def _get_position_index(self, length, batch_size):
        position_index_list = []
        for i in range(batch_size):
            position_index = torch.arange(0, end=length, step=1, dtype=torch.int64)
            position_index_list.append(position_index)
        batch_position_index = torch.stack(position_index_list, dim=0)
        return batch_position_index

    def forward_train(self, feat, out_enc, targets, valid_ratios, position_index):
        (n, c_enc, h, w) = out_enc.shape
        assert c_enc == self.dim_model
        (_, c_feat, _, _) = feat.shape
        assert c_feat == self.dim_input
        (_, len_q) = targets.shape
        assert len_q <= self.max_seq_len
        position_out_enc = self.position_aware_module(out_enc)
        query = self.embedding(position_index)
        query = query.permute(0, 2, 1)
        key = torch.reshape(position_out_enc, (n, c_enc, h * w))
        if self.encode_value:
            value = torch.reshape(out_enc, (n, c_enc, h * w))
        else:
            value = torch.reshape(feat, (n, c_feat, h * w))
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        attn_out = attn_out.permute(0, 2, 1)
        if self.return_feature:
            return attn_out
        return self.prediction(attn_out)

    def forward_test(self, feat, out_enc, valid_ratios, position_index):
        (n, c_enc, h, w) = out_enc.shape
        assert c_enc == self.dim_model
        (_, c_feat, _, _) = feat.shape
        assert c_feat == self.dim_input
        position_out_enc = self.position_aware_module(out_enc)
        query = self.embedding(position_index)
        query = query.permute(0, 2, 1)
        key = torch.reshape(position_out_enc, (n, c_enc, h * w))
        if self.encode_value:
            value = torch.reshape(out_enc, (n, c_enc, h * w))
        else:
            value = torch.reshape(feat, (n, c_feat, h * w))
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        attn_out = attn_out.permute(0, 2, 1)
        if self.return_feature:
            return attn_out
        return self.prediction(attn_out)


class RobustScannerFusionLayer(nn.Module):
    def __init__(self, dim_model, dim=-1):
        super().__init__()
        self.dim_model = dim_model
        self.dim = dim
        self.linear_layer = nn.Linear(dim_model * 2, dim_model * 2)

    def forward(self, x0, x1):
        assert x0.shape == x1.shape
        fusion_input = torch.concat([x0, x1], self.dim)
        output = self.linear_layer(fusion_input)
        output = F.glu(output, self.dim)
        return output


class RobustScannerDecoder(BaseDecoder):
    def __init__(
        self,
        num_classes=None,
        dim_input=512,
        dim_model=128,
        hybrid_decoder_rnn_layers=2,
        hybrid_decoder_dropout=0,
        position_decoder_rnn_layers=2,
        max_seq_len=40,
        start_idx=0,
        mask=True,
        padding_idx=None,
        encode_value=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.encode_value = encode_value
        self.start_idx = start_idx
        self.padding_idx = padding_idx
        self.mask = mask
        self.hybrid_decoder = SequenceAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=hybrid_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_seq_len,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            dropout=hybrid_decoder_dropout,
            encode_value=encode_value,
            return_feature=True,
        )
        self.position_decoder = PositionAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=position_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_seq_len,
            mask=mask,
            encode_value=encode_value,
            return_feature=True,
        )
        self.fusion_module = RobustScannerFusionLayer(
            self.dim_model if encode_value else dim_input
        )
        pred_num_classes = num_classes - 1
        self.prediction = nn.Linear(
            dim_model if encode_value else dim_input, pred_num_classes
        )

    def forward_train(self, feat, out_enc, target, valid_ratios, word_positions):
        hybrid_glimpse = self.hybrid_decoder.forward_train(
            feat, out_enc, target, valid_ratios
        )
        position_glimpse = self.position_decoder.forward_train(
            feat, out_enc, target, valid_ratios, word_positions
        )
        fusion_out = self.fusion_module(hybrid_glimpse, position_glimpse)
        out = self.prediction(fusion_out)
        return out

    def forward_test(self, feat, out_enc, valid_ratios, word_positions):
        seq_len = self.max_seq_len
        batch_size = feat.shape[0]
        decode_sequence = (
            torch.ones((batch_size, seq_len), dtype=torch.int64) * self.start_idx
        )
        position_glimpse = self.position_decoder.forward_test(
            feat, out_enc, valid_ratios, word_positions
        )
        outputs = []
        for i in range(seq_len):
            hybrid_glimpse_step = self.hybrid_decoder.forward_test_step(
                feat, out_enc, decode_sequence, i, valid_ratios
            )
            fusion_out = self.fusion_module(
                hybrid_glimpse_step, position_glimpse[:, i, :]
            )
            char_out = self.prediction(fusion_out)
            char_out = F.softmax(char_out, -1)
            outputs.append(char_out)
            max_idx = torch.argmax(char_out, dim=1, keepdim=False)
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx
        outputs = torch.stack(outputs, 1)
        return outputs


class RobustScannerHead(nn.Module):
    def __init__(
        self,
        out_channels,
        in_channels,
        enc_outchannles=128,
        hybrid_dec_rnn_layers=2,
        hybrid_dec_dropout=0,
        position_dec_rnn_layers=2,
        start_idx=0,
        max_text_length=40,
        mask=True,
        padding_idx=None,
        encode_value=False,
        **kwargs
    ):
        super().__init__()
        self.encoder = ChannelReductionEncoder(
            in_channels=in_channels, out_channels=enc_outchannles
        )
        self.decoder = RobustScannerDecoder(
            num_classes=out_channels,
            dim_input=in_channels,
            dim_model=enc_outchannles,
            hybrid_decoder_rnn_layers=hybrid_dec_rnn_layers,
            hybrid_decoder_dropout=hybrid_dec_dropout,
            position_decoder_rnn_layers=position_dec_rnn_layers,
            max_seq_len=max_text_length,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            encode_value=encode_value,
        )

    def forward(self, inputs, targets=None):
        out_enc = self.encoder(inputs)
        valid_ratios = None
        word_positions = targets[-1]
        if len(targets) > 1:
            valid_ratios = targets[-2]
        if self.training:
            label = targets[0]
            label = torch.Tensor(label, dtype=torch.int64)
            final_out = self.decoder(
                inputs, out_enc, label, valid_ratios, word_positions
            )
        if not self.training:
            final_out = self.decoder(
                inputs,
                out_enc,
                label=None,
                valid_ratios=valid_ratios,
                word_positions=word_positions,
                train_mode=False,
            )
        return final_out
