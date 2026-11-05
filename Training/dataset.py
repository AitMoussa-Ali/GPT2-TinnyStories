
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenBlockDataset(Dataset):
    def __init__(self, file_path: str, block_size: int):
        self.file_path = Path(file_path)
        self.block_size = block_size

        # Open the binary token stream without loading everything into RAM
        self.data = np.memmap(
            self.file_path,
            dtype=np.uint16,
            mode="r"
        )

        if len(self.data) <= block_size:
            raise ValueError(
                "Token stream must contain more tokens than block_size."
            )

        # Number of complete blocks containing block_size + 1 tokens
        self.num_blocks = (len(self.data) - 1) // block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        if index < 0 or index >= self.num_blocks:
            raise IndexError("Dataset index out of range.")

        start = index * self.block_size
        end = start + self.block_size + 1

        # Read one block and convert it to int64
        chunk = np.array(
            self.data[start:end],
            dtype=np.int64,
            copy=True
        )

        # Shift inputs and targets by one position
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y


def create_dataloaders(
    train_path: str,
    validation_path: str,
    block_size: int,
    batch_size: int,
    num_workers: int = 0
):
    train_dataset = TokenBlockDataset(
        file_path=train_path,
        block_size=block_size
    )

    validation_dataset = TokenBlockDataset(
        file_path=validation_path,
        block_size=block_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, validation_loader
