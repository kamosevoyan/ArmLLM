import torch
from torch import device
from torch.nn import Module
from transformers import AutoTokenizer

from config import ModelArgs
from model import Transformer


def generate_text_greedy(
    model: Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: device,
    max_length: int = 50,
    temperature: float = 1.0,
):
    model.eval()
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokenized_prompt, start_pos=0)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokenized_prompt = torch.cat([tokenized_prompt, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(tokenized_prompt[0], skip_special_tokens=True)

    return generated_text


def generate_text_sampling(
    model: Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: device,
    max_length: int = 50,
    temperature: float = 1.0,
):
    model.eval()
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokenized_prompt, start_pos=0)[:, -1, :]
            logits /= temperature
            proba = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(proba, 1)
            tokenized_prompt = torch.cat([tokenized_prompt, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(tokenized_prompt[0], skip_special_tokens=True)

    return generated_text


def generate_text_topk(
    model: Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: device,
    max_length: int = 50,
    temperature: float = 1.0,
    topk: int = 50,
):
    model.eval()
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokenized_prompt, start_pos=0)[:, -1, :]
            logits /= temperature

            topk_values, topk_indices = torch.topk(logits, topk, dim=-1)
            proba = -float("inf") * logits.new_ones(logits.shape)
            proba.scatter_(index=topk_indices, src=topk_values, dim=-1)
            proba = torch.softmax(proba, dim=-1)
            next_token = torch.multinomial(proba, 1)

            tokenized_prompt = torch.cat([tokenized_prompt, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(tokenized_prompt[0], skip_special_tokens=True)

    return generated_text


def generate_text_topp(
    model: Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: device,
    max_length: int = 50,
    temperature: float = 1.0,
    topp: float = 0.9,
):
    model.eval()
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokenized_prompt, start_pos=0)[:, -1, :]
            logits /= temperature
            proba = torch.softmax(logits, dim=-1)

            sorted_indices = proba.argsort(descending=True, dim=-1)
            cumsum = torch.cumsum(proba.gather(index=sorted_indices, dim=-1), dim=-1)
            topp_elements = cumsum > topp

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(index=sorted_indices, src=topp_elements, dim=-1)

            logits[indices_to_remove] = -float("inf")
            proba = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(proba, 1)

            tokenized_prompt = torch.cat([tokenized_prompt, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(tokenized_prompt[0], skip_special_tokens=True)

    return generated_text


def main():

    model_args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        multiple_of=32,
        max_seq_len=128,
        max_batch_size=256,
        vocab_size=50257,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(model_args).to(device)

    # path = "/home/kamo/resources/ArmLLM/Module 1: Intro & Pretraining/llama/llama_wikitext_trained.pth"
    # weights = torch.load(path, map_location=device)
    # model.load_state_dict(weights)

    prompt = "1 2 3 4 5 6 7"#input("Input:\t")
    generated_text = generate_text_sampling(
        model, tokenizer, prompt, device, max_length=200, temperature=1
    )

    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()
