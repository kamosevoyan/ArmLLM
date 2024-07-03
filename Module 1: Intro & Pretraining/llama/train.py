import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ModelArgs
from model import Transformer
from utils import create_dataloader


# Training function
def train(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: device,
    epoch: int,
    pad_token_id: int,
    writer: SummaryWriter,
) -> tuple[int, int]:

    model.train()
    total_loss = 0
    total_tokens = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        optimizer.zero_grad()

        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        logits = model(input_ids, start_pos=0)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="sum",
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_tokens += (target_ids != pad_token_id).sum().item()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Lr", current_lr, epoch * len(dataloader) + batch_i)

        progress_bar.set_postfix(
            {"loss": f"{avg_loss:.4f}", "ppl": f"{perplexity:.2f}"}
        )

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def main():

    model_args = ModelArgs(
        dim=1024,
        n_layers=8,
        n_heads=16,
        n_kv_heads=4,
        multiple_of=32,
        max_seq_len=128,
        max_batch_size=256,
        vocab_size=50257,
    )

    batch_size = model_args.max_batch_size
    num_epochs = 500
    learning_rate = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(model_args).to(device)

    dataloader = create_dataloader(batch_size, model_args.max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader), eta_min=learning_rate / 10
    )

    writer = SummaryWriter("../../runs/llama/exp0", flush_secs=5)

    try:
        for epoch in range(num_epochs):
            perplexity, loss = train(
                model,
                dataloader,
                optimizer,
                scheduler,
                device,
                epoch,
                pad_token_id,
                writer,
            )
            writer.add_scalar("Loss", loss, epoch)
            writer.add_scalar("Perplexity", perplexity, epoch)

    except KeyboardInterrupt as kbe:
        pass
    finally:
        hparam_dict = asdict(model_args)
        hparam_dict.update(
            {
                "learning_rate": learning_rate,
            }
        )
        writer.add_hparams(
            hparam_dict=hparam_dict,
            metric_dict={"perplexity": perplexity},
        )
        torch.save(model.state_dict(), "llama_wikitext_trained.pth")


if __name__ == "__main__":
    main()
