import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "The parameter `d_model` must be divisble by `num_heads`"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ):
        # shapes: [batch_size, num_heads, seq_len, d_k]
        attn_s = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)
        attn_probs = torch.softmax(attn_s, dim=-1)
        output = torch.matmul(attn_probs, V)

        return output
        # Shape: [batch_size, num_heads, seq_len, d_k]

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output = self.scaled_dot_product_attention(Q, K, V)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )

        return self.W_o(output)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO: Implement the feed-forward network
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, seq_len, d_model]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, seq_len, d_model]
        attn_output = self.self_attn(x, x, x)
        attn_output = self.dropout1(attn_output)
        # Shape: [batch_size, seq_len, d_model]
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)
        return x
