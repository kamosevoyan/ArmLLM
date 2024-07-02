import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# Helper function to repeat key/value heads
def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    
    # return torch.repeat_interleave(x, dim=1, repeats=n_rep)
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# Function to find the nearest multiple of k
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# Precompute frequency tensor for rotary positional embedding
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# Apply rotary positional embedding
def apply_rotary_emb(
    xq: Tensor, xk: Tensor, freqs_cis: Tensor
) -> tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def create_dataloader(batch_size: int, max_seq_len: int) -> DataLoader:
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_overflowing_tokens=True,
            return_length=True,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    def collate_fn(examples):
        return torch.tensor(
            [example["input_ids"] for example in examples], dtype=torch.long
        )

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=16,
    )
