import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import TransformerEncoder
from utils import load_and_preprocess_data, train, validate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes: int = 525
    img_size: int = 224
    patch_size: int = 16

    num_layers: int = 8
    num_heads: int = 4
    d_model: int = 512
    d_ff: int = 1024

    num_epochs: int = 100
    batch_size: int = 256

    learning_rate: float = 1e-4
    weight_decay: float = 5e-5

    train_data, validation_data = load_and_preprocess_data()

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        persistent_workers=True,
    )

    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )

    model = TransformerEncoder(
        img_size, patch_size, d_model, num_heads, num_layers, d_ff, num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, cooldown=5, min_lr=1e-7
    )

    writer = SummaryWriter("../../runs/exp0", flush_secs=5)

    for epoch in range(num_epochs):
        train_acc, train_loss = train(epoch, model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = validate(model, validation_loader, criterion, device)
        scheduler.step(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Lr", current_lr, epoch)

    writer.add_hparams(
        hparam_dict={
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        metric_dict={"val_acc": val_acc},
    )


if __name__ == "__main__":
    main()
