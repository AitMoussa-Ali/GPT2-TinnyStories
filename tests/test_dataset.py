
from Training.dataset import create_dataloaders
from config import GPTConfig
import torch

config = GPTConfig()

train_loader, validation_loader = create_dataloaders(
    train_path="Data/processed/tinystories/tokenized/train.bin",
    validation_path="Data/processed/tinystories/tokenized/validation.bin",
    block_size=config.block_size,
    batch_size=16
)

# Get one batch
x, y = next(iter(train_loader))

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("\nFirst input sequence:")
print(x[0].tolist())

print("\nFirst target sequence:")
print(y[0].tolist())

# Verify dimensions
assert x.shape == (16, config.block_size)
assert y.shape == (16, config.block_size)

# Verify the next-token relationship
assert torch.equal(x[:, 1:], y[:, :-1])

print("\nDataset tests passed!")
