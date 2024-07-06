import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import bitsandbytes as bnb

# Load the pre-trained model and tokenizer
model_name = "google/gemma-2-27b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("xsum")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples['document']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
)

# Initialize the QLoRA configuration
lora_config = bnb.optim.LoRAConfig(
    lora_dim=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    enable_lora=True,
)

# Apply QLoRA to the model
model = bnb.optim.prepare_model(model, lora_config)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine-tuned-gemma-2-27b")

# Save the tokenizer
tokenizer.save_pretrained("./fine-tuned-gemma-2-27b")
