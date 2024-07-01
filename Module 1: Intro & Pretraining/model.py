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
        # images shape: [batch_size, channels, height, width]
        batch_size = images.shape[0]
        # patches shape: [batch_size, num_patches, patch_dim]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)
        return patches  # Shape: [batch_size, num_patches, patch_dim]

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, channels, height, width]
        x = self.__patchify(x)  # Shape: [batch_size, num_patches, patch_dim]
        x = self.patch_embedding(x)  # Shape: [batch_size, num_patches, d_model]
        # TODO: positional embedding, layers, norm,
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Take the mean across patches
        return self.fc(x)  # Shape: [batch_size, num_classes]
