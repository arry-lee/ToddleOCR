import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class REDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size, max_seq_len = entities.shape[:2]
        new_relations = torch.full(
            [batch_size, max_seq_len * max_seq_len, 3], fill_value=-1, dtype=relations.dtype
        )
        for b in range(batch_size):
            if entities[b, 0, 0] <= 2:
                entitie_new = torch.full([512, 3], fill_value=-1, dtype=entities.dtype)
                entitie_new[0, :] = 2
                entitie_new[1:3, 0] = 0  # start
                entitie_new[1:3, 1] = 1  # end
                entitie_new[1:3, 2] = 0  # label
                entities[b] = entitie_new
            entitie_label = entities[b, 1 : entities[b, 0, 2] + 1, 2]
            all_possible_relations1 = torch.arange(0, entities[b, 0, 2], dtype=entities.dtype)
            all_possible_relations1 = all_possible_relations1[entitie_label == 1]
            all_possible_relations2 = torch.arange(0, entities[b, 0, 2], dtype=entities.dtype)
            all_possible_relations2 = all_possible_relations2[entitie_label == 2]

            all_possible_relations = torch.stack(
                torch.meshgrid(all_possible_relations1, all_possible_relations2), dim=2
            ).reshape([-1, 2])
            if len(all_possible_relations) == 0:
                all_possible_relations = torch.full_like(all_possible_relations, fill_value=-1, dtype=entities.dtype)
                all_possible_relations[0, 0] = 0
                all_possible_relations[0, 1] = 1

            relation_head = relations[b, 1 : relations[b, 0, 0] + 1, 0]
            relation_tail = relations[b, 1 : relations[b, 0, 1] + 1, 1]
            positive_relations = torch.stack([relation_head, relation_tail], dim=1)

            all_possible_relations_repeat = all_possible_relations.unsqueeze(dim=1).tile(
                [1, len(positive_relations), 1]
            )
            positive_relations_repeat = positive_relations.unsqueeze(dim=0).tile([len(all_possible_relations), 1, 1])
            mask = torch.all(all_possible_relations_repeat == positive_relations_repeat, dim=2)
            negative_mask = torch.any(mask, dim=1)
            negative_relations = all_possible_relations[negative_mask]

            positive_mask = torch.any(mask, dim=0)
            positive_relations = positive_relations[positive_mask]
            if negative_mask.sum() > 0:
                reordered_relations = torch.concat([positive_relations, negative_relations])
            else:
                reordered_relations = positive_relations

            relation_per_doc_label = torch.zeros([len(reordered_relations), 1], dtype=reordered_relations.dtype)
            relation_per_doc_label[: len(positive_relations)] = 1
            relation_per_doc = torch.concat([reordered_relations, relation_per_doc_label], dim=1)
            assert len(relation_per_doc[:, 0]) != 0
            new_relations[b, 0] = relation_per_doc.shape[0]#.astype(new_relations.dtype)
            new_relations[b, 1 : len(relation_per_doc) + 1] = relation_per_doc
            # new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = torch.full([7, 2], fill_value=-1, dtype=relations.dtype)
            rel[0, 0] = relations[:, 0][i]
            rel[1, 0] = entities[:, 0][relations[:, 0][i] + 1]
            rel[1, 1] = entities[:, 1][relations[:, 0][i] + 1]
            rel[2, 0] = entities[:, 2][relations[:, 0][i] + 1]
            rel[3, 0] = relations[:, 1][i]
            rel[4, 0] = entities[:, 0][relations[:, 1][i] + 1]
            rel[4, 1] = entities[:, 1][relations[:, 1][i] + 1]
            rel[5, 0] = entities[:, 2][relations[:, 1][i] + 1]
            rel[6, 0] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_length, _ = entities.shape
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = torch.full(
            [batch_size, max_length * max_length, 7, 2], fill_value=-1, dtype=entities.dtype
        )
        for b in range(batch_size):
            relation = relations[b, 1 : relations[b, 0, 0] + 1]
            head_entities = relation[:, 0]
            tail_entities = relation[:, 1]
            relation_labels = relation[:, 2]
            entities_start_index = torch.tensor(entities[b, 1 : entities[b, 0, 0] + 1, 0])
            entities_labels = torch.tensor(entities[b, 1 : entities[b, 0, 2] + 1, 2])
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            tmp_hidden_states = hidden_states[b][head_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = torch.unsqueeze(tmp_hidden_states, dim=0)
            head_repr = torch.concat((tmp_hidden_states, head_label_repr), dim=-1)

            tmp_hidden_states = hidden_states[b][tail_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = torch.unsqueeze(tmp_hidden_states, dim=0)
            tail_repr = torch.concat((tmp_hidden_states, tail_label_repr), dim=-1)

            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relation, entities[b])
            if len(pred_relations) > 0:
                pred_relations = torch.stack(pred_relations)
                all_pred_relations[b, 0, :, :] = pred_relations.shape[0]#.astype(all_pred_relations.dtype)
                all_pred_relations[b, 1 : len(pred_relations) + 1, :, :] = pred_relations
        return loss, all_pred_relations
