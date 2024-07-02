import math

import torch
import torch.nn.functional as F
from config import ModelArgs
from model import Transformer
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import create_dataloader


# Training function
def train(model, dataloader, optimizer, scheduler, device, num_epochs, pad_token_id):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_tokens = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
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

            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "ppl": f"{perplexity:.2f}"}
            )

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        print(
            f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}"
        )


def main():

    model_args = ModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        multiple_of=32,
        max_seq_len=256,
        max_batch_size=256,
        vocab_size=50257,
    )

    batch_size = model_args.max_batch_size
    num_epochs = 100
    learning_rate = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(model_args).to(device)

    dataloader = create_dataloader(batch_size, model_args.max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader)
    )

    try:
        train(model, dataloader, optimizer, scheduler, device, num_epochs, pad_token_id)
    except KeyboardInterrupt as kbe:
        pass
    finally:
        torch.save(model.state_dict(), "llama_wikitext_trained.pth")


if __name__ == "__main__":
    main()
