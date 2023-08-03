# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionGRUCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = torch.shape(inputs)[0]
        num_steps = batch_max_length

        hidden = torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens.append(torch.unsqueeze(outputs, dim=1))
            output = torch.concat(output_hiddens, axis=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros([batch_size], dtype="int32")
            probs = None
            char_onehots = None
            outputs = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(outputs)
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.concat([probs, torch.unsqueeze(probs_step, dim=1)], axis=1)
                next_input = probs_step.argmax(axis=1)
                targets = next_input
        if not self.training:
            probs = torch.nn.functional.softmax(probs, axis=2)
        return probs


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)

        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = torch.transpose(alpha, [0, 2, 1])
        context = torch.squeeze(torch.mm(alpha, batch_H), axis=1)
        concat_context = torch.concat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


class AttentionLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionLSTMCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length

        hidden = (torch.zeros((batch_size, self.hidden_size)), torch.zeros((batch_size, self.hidden_size)))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                # one-hot vectors for a i-th char
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)

                hidden = (hidden[1][0], hidden[1][1])
                output_hiddens.append(torch.unsqueeze(hidden[0], dim=1))
            output = torch.concat(output_hiddens, axis=1)
            probs = self.generator(output)

        else:
            targets = torch.zeros([batch_size], dtype="int32")
            probs = None
            char_onehots = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = (hidden[1][0], hidden[1][1])
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.concat([probs, torch.unsqueeze(probs_step, dim=1)], axis=1)

                next_input = probs_step.argmax(axis=1)

                targets = next_input
        if not self.training:
            probs = torch.nn.functional.softmax(probs, axis=2)
        return probs


class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[0]), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = torch.transpose(alpha, [0, 2, 1])
        context = torch.squeeze(torch.mm(alpha, batch_H), axis=1)
        concat_context = torch.concat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha
