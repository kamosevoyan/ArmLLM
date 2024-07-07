import torch
import torch.nn as nn
from config import ModelArgs
from layers import RMSNorm, TransformerBlock
from utils import precompute_freqs_cis


# Main Transformer model
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Create causal mask
        mask = None
        if seqlen > 1:
            raw_mask = torch.full((1, 1, seqlen, seqlen), float("-inf"))
            mask = torch.triu(raw_mask, diagonal=start_pos + 1).to(
                device=tokens.device, dtype=h.dtype
            )

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)  # Return logits for all positions
        return output
