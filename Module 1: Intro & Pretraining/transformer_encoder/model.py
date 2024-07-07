import torch
import torch.nn as nn
from encoding import PositionalEncoding
from layers import EncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_classes: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.positional_embedding = PositionalEncoding(d_model, self.num_patches)
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def __patchify(self, images: torch.Tensor):
        batch_size = images.shape[0]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)
        return patches

    def forward(self, x: torch.Tensor):
        x = self.__patchify(x)
        x = self.patch_embedding(x)
        x = self.positional_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)
