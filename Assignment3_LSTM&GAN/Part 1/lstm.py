from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input modulation gate g: W_gx, W_gh, b_g
        self.W_gx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_gh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Input gate i: W_ix, W_ih, b_i
        self.W_ix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_ih = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Forget gate f: W_fx, W_fh, b_f
        self.W_fx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_fh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Output gate o: W_ox, W_oh, b_o
        self.W_ox = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_oh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Output layer: W_ph, b_p
        self.W_ph = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        device = x.device

        # Initialize h^(0) and c^(0) to zeros
        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        for t in range(self.seq_length):
            x_t = x[:, t, :]  # (batch_size, input_dim)

            # Equation (1): g^(t) = tanh(W_gx * x^(t) + W_gh * h^(t-1) + b_g)
            g_t = torch.tanh(self.W_gx(x_t) + self.W_gh(h_t))

            # Equation (2): i^(t) = sigmoid(W_ix * x^(t) + W_ih * h^(t-1) + b_i)
            i_t = torch.sigmoid(self.W_ix(x_t) + self.W_ih(h_t))

            # Equation (3): f^(t) = sigmoid(W_fx * x^(t) + W_fh * h^(t-1) + b_f)
            f_t = torch.sigmoid(self.W_fx(x_t) + self.W_fh(h_t))

            # Equation (4): o^(t) = sigmoid(W_ox * x^(t) + W_oh * h^(t-1) + b_o)
            o_t = torch.sigmoid(self.W_ox(x_t) + self.W_oh(h_t))

            # Equation (5): c^(t) = g^(t) * i^(t) + c^(t-1) * f^(t)
            c_t = g_t * i_t + c_t * f_t

            # Equation (6): h^(t) = tanh(c^(t)) * o^(t)
            h_t = torch.tanh(c_t) * o_t

        # Equation (7): p^(t) = W_ph * h^(t) + b_p
        p = self.W_ph(h_t)

        # Equation (8): y_hat = softmax(p) - softmax is applied in CrossEntropyLoss
        return p