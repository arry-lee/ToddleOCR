import torch
from torch import nn
from torch.nn import functional as F

gradient_clip = 10


class WrapEncoderForFeature(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        bos_idx=0,
    ):
        super().__init__()

        self.prepare_encoder = PrepareEncoder(
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            bos_idx=bos_idx,
            word_emb_param_name="src_word_emb_table",
        )
        self.encoder = Encoder(
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
        )

    def forward(self, enc_inputs):
        conv_features, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self.prepare_encoder(conv_features, src_pos)
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class WrapEncoder(nn.Module):
    """
    embedder + encoder
    """

    def __init__(
        self,
        src_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        bos_idx=0,
    ):
        super().__init__()

        self.prepare_decoder = PrepareDecoder(
            src_vocab_size, d_model, max_length, prepostprocess_dropout, bos_idx=bos_idx
        )
        self.encoder = Encoder(
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
        )

    def forward(self, enc_inputs):
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self.prepare_decoder(src_word, src_pos)
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class Encoder(nn.Module):
    """
    encoder
    """

    def __init__(
        self,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd="n",
        postprocess_cmd="da",
    ):
        super().__init__()

        self.encoder_layers = list()
        for i in range(n_layer):
            self.encoder_layers.append(
                self.add_module(
                    "layer_%d" % i,
                    EncoderLayer(
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        preprocess_cmd,
                        postprocess_cmd,
                    ),
                )
            )
        self.processer = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )

    def forward(self, enc_input, attn_bias):
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output
        enc_output = self.processer(enc_output)
        return enc_output


class EncoderLayer(nn.Module):
    """
    EncoderLayer
    """

    def __init__(
        self,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd="n",
        postprocess_cmd="da",
    ):
        super().__init__()
        self.preprocesser1 = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )
        self.self_attn = MultiHeadAttention(
            d_key, d_value, d_model, n_head, attention_dropout
        )
        self.postprocesser1 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout
        )

        self.preprocesser2 = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser2 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout
        )

    def forward(self, enc_input, attn_bias):
        attn_output = self.self_attn(
            self.preprocesser1(enc_input), None, None, attn_bias
        )
        attn_output = self.postprocesser1(attn_output, enc_input)
        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.0):
        super().__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = torch.nn.Linear(
            in_features=d_model, out_features=d_key * n_head, bias=False
        )
        self.k_fc = torch.nn.Linear(
            in_features=d_model, out_features=d_key * n_head, bias=False
        )
        self.v_fc = torch.nn.Linear(
            in_features=d_model, out_features=d_value * n_head, bias=False
        )
        self.proj_fc = torch.nn.Linear(
            in_features=d_value * n_head, out_features=d_model, bias=False
        )

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:  # self-attention
            keys, values = queries, queries
            static_kv = False
        else:  # cross-attention
            static_kv = True

        q = self.q_fc(queries)
        q = torch.reshape(q, shape=[0, 0, self.n_head, self.d_key])
        q = q.permute(0, 2, 1, 3)

        if cache is not None and static_kv and "static_k" in cache:
            # for encoder-decoder attention in inference and has cached
            k = cache["static_k"]
            v = cache["static_v"]
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = torch.reshape(k, shape=[0, 0, self.n_head, self.d_key])
            k = k.permute(0, 2, 1, 3)
            v = torch.reshape(v, shape=[0, 0, self.n_head, self.d_value])
            v = v.permute(0, 2, 1, 3)

        if cache is not None:
            if static_kv and not "static_k" in cache:
                # for encoder-decoder attention in inference and has not cached
                cache["static_k"], cache["static_v"] = k, v
            elif not static_kv:
                # for decoder self-attention in inference
                cache_k, cache_v = cache["k"], cache["v"]
                k = torch.concat([cache_k, k], dim=2)
                v = torch.concat([cache_v, v], dim=2)
                cache["k"], cache["v"] = k, v

        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q, k, v = self._prepare_qkv(queries, keys, values, cache)

        # scale dot product attention
        product = torch.matmul(q, k.t())
        product = product * self.d_model**-0.5
        if attn_bias is not None:
            product += attn_bias
        weights = F.softmax(product)
        if self.dropout_rate:
            weights = F.dropout(weights, p=self.dropout_rate)
        out = torch.matmul(weights, v)

        # combine heads
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.proj_fc(out)

        return out


class PrePostProcessLayer(nn.Module):
    """
    PrePostProcessLayer
    """

    def __init__(self, process_cmd, d_model, dropout_rate):
        super().__init__()
        self.process_cmd = process_cmd
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(lambda x, y: x + y if y is not None else x)
            elif cmd == "n":  # add layer normalization
                self.functors.append(
                    self.add_module(
                        "layer_norm_%d" % len(list(self.children())),
                        torch.nn.LayerNorm(
                            normalized_shape=d_model,
                        ),
                    )
                )
            elif cmd == "d":  # add dropout
                self.functors.append(
                    lambda x: F.dropout(x, p=dropout_rate) if dropout_rate else x
                )

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.process_cmd):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class PrepareEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_emb_dim,
        src_max_len,
        dropout_rate=0,
        bos_idx=0,
        word_emb_param_name=None,
        pos_enc_param_name=None,
    ):
        super().__init__()
        self.src_emb_dim = src_emb_dim
        self.src_max_len = src_max_len
        self.emb = torch.nn.Embedding(
            num_embeddings=self.src_max_len, embedding_dim=self.src_emb_dim
        )
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word_emb = src_word
        src_word_emb = src_word_emb.type(dtype=torch.float32)

        scale = self.src_emb_dim**0.5
        src_word_emb = torch.mul(src_word_emb, scale)

        src_pos = torch.squeeze(src_pos, dim=-1)
        src_pos_enc = self.emb(src_pos)
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(enc_input, p=self.dropout_rate)
        else:
            out = enc_input
        return out


class PrepareDecoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_emb_dim,
        src_max_len,
        dropout_rate=0,
        bos_idx=0,
        word_emb_param_name=None,
        pos_enc_param_name=None,
    ):
        super().__init__()
        self.src_emb_dim = src_emb_dim
        """
        self.emb0 = Embedding(num_embeddings=src_vocab_size,
                              embedding_dim=src_emb_dim)
        """
        self.emb0 = torch.nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=self.src_emb_dim,
            padding_idx=bos_idx,
            # initializer=nn.init.Normal(0.0, src_emb_dim ** -0.5),
        )
        nn.init.normal_(self.emb0.weight, 0.0, src_emb_dim**-0.5)

        self.emb1 = torch.nn.Embedding(
            num_embeddings=src_max_len,
            embedding_dim=self.src_emb_dim,
            # weight_attr=torch.ParamAttr(name=pos_enc_param_name),
        )
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word = src_word.type(dtype=torch.int64)
        src_word = torch.squeeze(src_word, dim=-1)
        src_word_emb = self.emb0(src_word)
        scale = self.src_emb_dim**0.5
        src_word_emb = torch.mul(src_word_emb, scale)
        src_pos = torch.squeeze(src_pos, dim=-1)
        src_pos_enc = self.emb1(src_pos)
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(enc_input, p=self.dropout_rate)
        else:
            out = enc_input
        return out


class FFN(nn.Module):
    """
    Feed-Forward Network
    """

    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = torch.nn.Linear(in_features=d_model, out_features=d_inner_hid)
        self.fc2 = torch.nn.Linear(in_features=d_inner_hid, out_features=d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        if self.dropout_rate:
            hidden = F.dropout(hidden, p=self.dropout_rate)
        out = self.fc2(hidden)
        return out
