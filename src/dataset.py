import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import config
from utils import ClinicalDataset  # your custom dataset

SPLIT_PATH = "data_splits.pth"

def get_loaders(batch_size=config.BATCH_SIZE, train_frac=0.8, val_frac=0.1, force_resplit=False):
    """
    Returns DataLoaders for train, validation, and test sets.
    train_frac + val_frac < 1.0, remainder is test fraction.
    """
    # Load data
    df = pd.read_csv(config.DATA_PATH)
    df = df.dropna(subset=["findings", "impression"])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    dataset = ClinicalDataset(df, tokenizer)

    # Check if split exists
    if os.path.exists(SPLIT_PATH) and not force_resplit:
        print(f"[INFO] Loading existing split from {SPLIT_PATH}")
        split = torch.load(SPLIT_PATH)
        train_indices = split["train"]
        val_indices = split["val"]
        test_indices = split["test"]
    else:
        print("[INFO] Creating new train/val/test split")
        num_samples = len(dataset)
        indices = torch.randperm(num_samples).tolist()

        train_size = int(train_frac * num_samples)
        val_size = int(val_frac * num_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        torch.save({
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        }, SPLIT_PATH)

    # Subset datasets
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
