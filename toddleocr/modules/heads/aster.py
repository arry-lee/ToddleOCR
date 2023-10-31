"""
This code is refer from:
https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/attention_recognition_head.py
"""
__all__ = ["AsterHead"]

import torch
from torch import nn
from torch.nn import functional as F


class AsterHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sDim,
        attDim,
        max_len_labels,
        time_step=25,
        beam_width=5,
        **kwargs
    ):
        super().__init__()
        self.num_classes = out_channels
        self.in_planes = in_channels
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels
        self.decoder = AttentionRecognitionHead(
            in_channels, out_channels, sDim, attDim, max_len_labels
        )
        self.time_step = time_step
        self.embeder = Embedding(self.time_step, in_channels)
        self.beam_width = beam_width
        self.eos = self.num_classes - 3

    def forward(self, x, targets=None, embed=None):
        return_dict = {}
        embedding_vectors = self.embeder(x)

        if self.training:
            rec_targets, rec_lengths, _ = targets
            rec_pred = self.decoder([x, rec_targets, rec_lengths], embedding_vectors)
            return_dict["rec_pred"] = rec_pred
            return_dict["embedding_vectors"] = embedding_vectors
        else:
            rec_pred, rec_pred_scores = self.decoder.beam_search(
                x, self.beam_width, self.eos, embedding_vectors
            )
            return_dict["rec_pred"] = rec_pred
            return_dict["rec_pred_scores"] = rec_pred_scores
            return_dict["embedding_vectors"] = embedding_vectors

        return return_dict


class Embedding(nn.Module):
    def __init__(self, in_timestep, in_planes, mid_dim=4096, embed_dim=300):
        super().__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.eEmbed = nn.Linear(
            in_timestep * in_planes, self.embed_dim
        )  # Embed encoder output to a word-embedding like

    def forward(self, x):
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.eEmbed(x)
        return x


class AttentionRecognitionHead(nn.Module):
    """
    input: [b x 16 x 64 x in_planes]
    output: probability sequence: [b x T x num_classes]
    """

    def __init__(self, in_channels, out_channels, sDim, attDim, max_len_labels):
        super().__init__()
        self.num_classes = (
            out_channels  # this is the output classes. So it includes the <EOS>.
        )
        self.in_planes = in_channels
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels

        self.decoder = DecoderUnit(
            sDim=sDim, xDim=in_channels, yDim=self.num_classes, attDim=attDim
        )

    def forward(self, x, embed):
        x, targets, lengths = x
        batch_size = x.shape[0]
        # Decoder
        state = self.decoder.get_initial_state(embed)
        outputs = []
        for i in range(max(lengths)):
            if i == 0:
                y_prev = torch.full([batch_size], self.num_classes)
            else:
                y_prev = targets[:, i - 1]
            output, state = self.decoder(x, state, y_prev)
            outputs.append(output)
        outputs = torch.concat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    # inference stage.
    def sample(self, x):
        x, _, _ = x
        batch_size = x.size(0)
        # Decoder
        state = torch.zeros([1, batch_size, self.sDim])

        predicted_ids, predicted_scores = [], []
        for i in range(self.max_len_labels):
            if i == 0:
                y_prev = torch.full([batch_size], self.num_classes)
            else:
                y_prev = predicted

            output, state = self.decoder(x, state, y_prev)
            output = F.softmax(output, dim=1)
            score, predicted = output.max(1)
            predicted_ids.append(predicted.unsqueeze(1))
            predicted_scores.append(score.unsqueeze(1))
        predicted_ids = torch.concat([predicted_ids, 1])
        predicted_scores = torch.concat([predicted_scores, 1])
        # return predicted_ids.squeeze(), predicted_scores.squeeze()
        return predicted_ids, predicted_scores

    def beam_search(self, x, beam_width, eos, embed):
        def _inflate(tensor, times, dim):
            repeat_dims = [1] * tensor.dim()
            repeat_dims[dim] = times
            output = torch.tile(tensor, repeat_dims)
            return output

        # https://github.com/IBM/pytorch-seq2seq/blob/fede87655ddce6c94b38886089e05321dc9802af/seq2seq/models/TopKDecoder.py
        batch_size, l, d = x.shape
        x = torch.tile(x.unsqueeze(1).permute(1, 0, 2, 3), [beam_width, 1, 1, 1])
        inflated_encoder_feats = torch.reshape(x.permute(1, 0, 2, 3), [-1, l, d])

        # Initialize the decoder
        state = self.decoder.get_initial_state(embed, tile_times=beam_width)

        pos_index = torch.reshape(torch.arange(batch_size) * beam_width, shape=[-1, 1])

        # Initialize the scores
        sequence_scores = torch.full([batch_size * beam_width, 1], -float("Inf"))
        index = [i * beam_width for i in range(0, batch_size)]
        sequence_scores[index] = 0.0

        # Initialize the input vector
        y_prev = torch.full([batch_size * beam_width], self.num_classes)

        # Store decisions for backtracking
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()

        for i in range(self.max_len_labels):
            output, state = self.decoder(inflated_encoder_feats, state, y_prev)
            state = torch.unsqueeze(state, dim=0)
            log_softmax_output = torch.nn.functional.log_softmax(output, dim=1)

            sequence_scores = _inflate(sequence_scores, self.num_classes, 1)
            sequence_scores += log_softmax_output
            scores, candidates = torch.topk(
                torch.reshape(sequence_scores, [batch_size, -1]), beam_width, dim=1
            )

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            y_prev = torch.reshape(
                candidates % self.num_classes, shape=[batch_size * beam_width]
            )
            sequence_scores = torch.reshape(scores, shape=[batch_size * beam_width, 1])

            # Update fields for next timestep
            pos_index = torch.expand_as(pos_index, candidates)
            predecessors = candidates / self.num_classes + pos_index.type(
                dtype=torch.int64
            )
            predecessors = torch.reshape(
                predecessors, shape=[batch_size * beam_width, 1]
            )
            state = torch.index_select(state, index=predecessors.squeeze(), dim=1)

            # Update sequence socres and erase scores for <eos> symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            y_prev = torch.reshape(y_prev, shape=[-1, 1])
            eos_prev = torch.full_like(y_prev, fill_value=eos)
            mask = eos_prev == y_prev
            mask = torch.nonzero(mask)
            if mask.dim() > 0:
                sequence_scores = sequence_scores.numpy()
                mask = mask.numpy()
                sequence_scores[mask] = -float("inf")
                sequence_scores = torch.Tensor(sequence_scores)

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            y_prev = torch.squeeze(y_prev)
            stored_emitted_symbols.append(y_prev)

        # Do backtracking to return the optimal values
        # ====== backtrak ======#
        # Initialize return variables given different types
        p = list()
        l = [
            [self.max_len_labels] * beam_width for _ in range(batch_size)
        ]  # Placeholder for lengths of top-k sequences

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = torch.topk(
            torch.reshape(stored_scores[-1], shape=[batch_size, beam_width]), beam_width
        )

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * batch_size  # the number of EOS found
        # in the backward loop below for each batch
        t = self.max_len_labels - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = torch.reshape(
            sorted_idx + pos_index.expand_as(sorted_idx),
            shape=[batch_size * beam_width],
        )
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = torch.index_select(
                stored_emitted_symbols[t], index=t_predecessors, dim=0
            )
            t_predecessors = torch.index_select(
                stored_predecessors[t].squeeze(), index=t_predecessors, dim=0
            )
            eos_indices = stored_emitted_symbols[t] == eos
            eos_indices = torch.nonzero(eos_indices)

            if eos_indices.dim() > 0:
                for i in range(eos_indices.shape[0] - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / beam_width)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_width - (batch_eos_found[b_idx] % beam_width) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_width + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
                    current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = stored_scores[t][idx[0], 0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)
            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(beam_width)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = torch.reshape(
            re_sorted_idx + pos_index.expand_as(re_sorted_idx),
            [batch_size * beam_width],
        )

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        p = [
            torch.reshape(
                torch.index_select(step, re_sorted_idx, 0),
                shape=[batch_size, beam_width, -1],
            )
            for step in reversed(p)
        ]
        p = torch.concat(p, -1)[:, 0, :]
        return p, torch.ones_like(p)


class AttentionUnit(nn.Module):
    def __init__(self, sDim, xDim, attDim):
        super().__init__()

        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim

        self.sEmbed = nn.Linear(sDim, attDim)
        self.xEmbed = nn.Linear(xDim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

    def forward(self, x, sPrev):
        batch_size, T, _ = x.shape  # [b x T x xDim]
        x = torch.reshape(x, [-1, self.xDim])  # [(b x T) x xDim]
        xProj = self.xEmbed(x)  # [(b x T) x attDim]
        xProj = torch.reshape(xProj, [batch_size, T, -1])  # [b x T x attDim]

        sPrev = sPrev.squeeze(0)
        sProj = self.sEmbed(sPrev)  # [b x attDim]
        sProj = torch.unsqueeze(sProj, 1)  # [b x 1 x attDim]
        sProj = torch.unsqueeze(sProj, 0).repeat(
            batch_size, T, self.attDim
        )  # [b x T x attDim]

        sumTanh = torch.tanh(sProj + xProj)
        sumTanh = torch.reshape(sumTanh, [-1, self.attDim])

        vProj = self.wEmbed(sumTanh)  # [(b x T) x 1]
        vProj = torch.reshape(vProj, [batch_size, T])
        alpha = F.softmax(
            vProj, dim=1
        )  # attention weights for each sample in the minibatch
        return alpha


class DecoderUnit(nn.Module):
    def __init__(self, sDim, xDim, yDim, attDim):
        super().__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.yDim = yDim
        self.attDim = attDim
        self.emdDim = attDim

        self.attention_unit = AttentionUnit(sDim, xDim, attDim)
        self.tgt_embedding = nn.Embedding(
            yDim + 1, self.emdDim
        )  # the last is used for <BOS>
        nn.init.normal_(self.tgt_embedding.weight, std=0.01)
        self.gru = nn.GRUCell(input_size=xDim + self.emdDim, hidden_size=sDim)
        self.fc = nn.Linear(sDim, yDim)
        nn.init.zeros_(self.fc.bias)
        self.embed_fc = nn.Linear(300, self.sDim)

    def get_initial_state(self, embed, tile_times=1):
        assert embed.shape[1] == 300
        state = self.embed_fc(embed)  # N * sDim
        if tile_times != 1:
            state = state.unsqueeze(1)
            trans_state = state.permute(1, 0, 2)
            state = torch.tile(trans_state, [tile_times, 1, 1])
            trans_state = state.permute(1, 0, 2)
            state = torch.reshape(trans_state, shape=[-1, self.sDim])
        state = state.unsqueeze(0)  # 1 * N * sDim
        return state

    def forward(self, x, sPrev, yPrev):
        # x: feature sequence from the image decoder.
        batch_size, T, _ = x.shape
        alpha = self.attention_unit(x, sPrev)
        context = torch.squeeze(torch.matmul(alpha.unsqueeze(1), x), dim=1)
        yPrev = yPrev.type(dtype=torch.int64)
        yProj = self.tgt_embedding(yPrev)

        concat_context = torch.concat([yProj, context], 1)
        concat_context = torch.squeeze(concat_context, 1)
        sPrev = torch.squeeze(sPrev, 0)
        output, state = self.gru(concat_context, sPrev)
        output = torch.squeeze(output, dim=1)
        output = self.fc(output)
        return output, state
