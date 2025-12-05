# train.py
import os
from datetime import datetime
import argparse

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from dataset import get_loaders
import config


def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if scaler:
            # Mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()
    return val_loss / len(loader)


def save_checkpoint(model, optimizer, epoch, prefix="checkpoint"):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"{prefix}_epoch{epoch+1}_{date_str}.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_name)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "checkpoint_latest.pth")
    print(f"Saved checkpoint: {ckpt_name}")


def main(batch_size=4, num_epochs=5, use_amp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(config.MODEL_PATH):
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        val_loss = validate_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--use_amp", action="store_true")  # <-- True if flag present
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(batch_size=args.batch_size, num_epochs=args.num_epochs, use_amp=args.use_amp)
