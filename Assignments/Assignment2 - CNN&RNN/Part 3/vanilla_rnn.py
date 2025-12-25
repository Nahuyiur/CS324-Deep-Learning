from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        # Weight matrices
        self.Wxh = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Whh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.Tensor(hidden_dim))

        self.Why = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.by = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)
        nn.init.zeros_(self.bh)
        nn.init.xavier_uniform_(self.Why)
        nn.init.zeros_(self.by)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, seq_len, _ = x.shape
        device = x.device
        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.Wxh + h_t @ self.Whh + self.bh)

        logits = h_t @ self.Why + self.by
        return logits
        
    # add more methods here if needed
