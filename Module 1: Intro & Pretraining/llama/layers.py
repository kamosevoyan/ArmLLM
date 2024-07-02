import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import ModelArgs
from utils import apply_rotary_emb, find_multiple, repeat_kv


# RMSNorm (Root Mean Square Layer Normalization)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):

        denominator = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * denominator * self.gamma

        return x


# Attention mechanism
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Initialize weight matrices for Q, K, V, and output
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor]):
        bsz, seqlen, _ = x.shape

        # Compute Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape Q, K, V
        xq = xq.view(bsz, seqlen, self.n_heads_q, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary positional embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat K and V for multi-query attention
        xk = repeat_kv(xk, self.n_rep)  # (bs, n_heads_q, seqlen, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, n_heads_q, seqlen, head_dim)
                
        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (bs, n_heads_q, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        # Compute attention scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads_q, seqlen, seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Compute attention output
        output = torch.matmul(scores, xv)  # (bs, n_heads_q, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


# Feedforward network with SwiGLU
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = find_multiple(hidden_dim, multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.intermediate_size,
            multiple_of=args.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        # Apply attention
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        # Apply feedforward
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
