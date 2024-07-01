import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


def load_and_preprocess_data() -> tuple[DatasetDict]:

    dataset = load_dataset("chriamue/bird-species-dataset", split="train[:5%]")
    train_dataset = dataset.train_test_split(test_size=0.1)

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def preprocess_image(example):
        inputs = image_processor(example["image"], return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "label": example["label"],
        }

    train_dataset["train"] = train_dataset["train"].map(
        preprocess_image, remove_columns=["image"]
    )
    train_dataset["train"].set_format(type="torch", columns=["pixel_values", "label"])

    train_dataset["test"] = train_dataset["test"].map(
        preprocess_image, remove_columns=["image"]
    )
    train_dataset["test"].set_format(type="torch", columns=["pixel_values", "label"])

    return train_dataset["train"], train_dataset["test"]


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[int]:

    model.eval()
    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)

    return accuracy, average_loss


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int]:

    model.train()
    total = 0
    correct = 0
    total_loss = 0

    for batch in dataloader:
        inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)

    return accuracy, average_loss