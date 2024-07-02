from dataclasses import dataclass
from typing import Optional

from utils import find_multiple


@dataclass
class ModelArgs:
    dim: int = 4096  # Dimension of the model
    n_layers: int = 32  # Number of transformer layers
    n_heads: int = 32  # Number of attention heads
    n_kv_heads: Optional[int] = (
        None  # Number of key/value heads (if different from n_heads)
    )
    vocab_size: int = 32000  # Size of the vocabulary
    multiple_of: int = 256  # Ensures certain dimensions are multiples of this value
    ffn_dim_multiplier: Optional[float] = (
        None  # Multiplier for FFN intermediate dimension
    )
    norm_eps: float = 1e-5  # Epsilon for normalization
    max_batch_size: int = 32  # Maximum batch size
    max_seq_len: int = 2048  # Maximum sequence length

    def __post_init__(self):
        # Set default values and calculate intermediate size
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.ffn_dim_multiplier is None:
            self.ffn_dim_multiplier = 4 / 3
        self.intermediate_size = int(2 * self.ffn_dim_multiplier * self.dim)
        self.intermediate_size = find_multiple(self.intermediate_size, self.multiple_of)
