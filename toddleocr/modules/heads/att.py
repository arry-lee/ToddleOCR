import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AttentionHead"]


class AttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super().__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionGRUCell(
            in_channels, hidden_size, out_channels, use_gru=False
        )
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length

        hidden = torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes
                )
                (outputs, hidden), alpha = self.attention_cell(
                    hidden, inputs, char_onehots
                )
                output_hiddens.append(torch.unsqueeze(outputs, dim=1))
            output = torch.concat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros([batch_size], dtype=torch.int32)
            probs = None
            char_onehots = None
            outputs = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes
                )
                (outputs, hidden), alpha = self.attention_cell(
                    hidden, inputs, char_onehots
                )
                probs_step = self.generator(outputs)
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.concat(
                        [probs, torch.unsqueeze(probs_step, dim=1)], dim=1
                    )
                next_input = probs_step.argmax(axis=1)
                targets = next_input
        if not self.training:
            probs = torch.nn.functional.softmax(probs, dim=2)
        return probs


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size
        )

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):

        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden).unsqueeze(1)

        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)  # 得分
        # alpha = torch.squeeze(alpha)
        alpha = alpha.transpose(1, 2)  # .permute(0, 2, 1)1,1,256
        context = torch.squeeze(torch.bmm(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttentionLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super().__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionLSTMCell(
            in_channels, hidden_size, out_channels, use_gru=False
        )
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length

        hidden = (
            torch.zeros((batch_size, self.hidden_size)),
            torch.zeros((batch_size, self.hidden_size)),
        )
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                # one-hot vectors for a i-th char
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes
                )
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)

                hidden = (hidden[1][0], hidden[1][1])
                output_hiddens.append(torch.unsqueeze(hidden[0], dim=1))
            output = torch.concat(output_hiddens, dim=1)
            probs = self.generator(output)

        else:
            targets = torch.zeros([batch_size], dtype=torch.int32)
            probs = None
            char_onehots = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes
                )
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = (hidden[1][0], hidden[1][1])
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.concat(
                        [probs, torch.unsqueeze(probs_step, dim=1)], dim=1
                    )

                next_input = probs_step.argmax(axis=1)

                targets = next_input
        if not self.training:
            probs = torch.nn.functional.softmax(probs, dim=2)
        return probs


class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size
            )
        else:
            self.rnn = nn.GRUCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size
            )

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[0]), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.mm(alpha, batch_H), dim=1)
        concat_context = torch.concat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


import math

# from paddle.nn import GRUCell
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))

        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)
        nn.init.uniform_(self.bias_ih, -std, std)
        nn.init.uniform_(self.bias_hh, -std, std)

        self._gate_activation = F.sigmoid
        self._activation = F.tanh

    def forward(self, inputs, states=None):

        # if states is None:
        #     states = self.get_initial_states(inputs, self.state_shape)

        pre_hidden = states
        x_gates = torch.matmul(inputs, self.weight_ih.t())
        # print(x_gates)
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = torch.matmul(pre_hidden, self.weight_hh.t())
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh

        x_r, x_z, x_c = torch.chunk(x_gates, 3, dim=1)
        h_r, h_z, h_c = torch.chunk(h_gates, 3, dim=1)

        r = self._gate_activation(x_r + h_r)
        z = self._gate_activation(x_z + h_z)
        c = self._activation(x_c + r * h_c)  # apply reset gate after mm
        h = (pre_hidden - c) * z + c
        # assert torch.all(z)
        # print(z)
        # print(states==h)
        return h, h

    @property
    def state_shape(self):
        return (self.hidden_size,)

    def extra_repr(self):
        return "{input_size}, {hidden_size}".format(**self.__dict__)
