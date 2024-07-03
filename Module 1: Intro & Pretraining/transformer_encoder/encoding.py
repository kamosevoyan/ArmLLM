import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", positional_encoding.unsqueeze(0))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:, : x.size(1)]
        x = self.dropout(x + pe)
        return x