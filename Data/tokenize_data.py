
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from Tokenizer.tokenizer_gpt2 import GPT2Tokenizer


# Paths
DATA_DIR = Path("Data/processed/tinystories")
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"

OUTPUT_DIR = DATA_DIR / "tokenized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def tokenize_and_save(dataset, tokenizer, output_path):
    """
    Tokenize stories individually, append EOS, and write
    all token IDs into one continuous binary stream.
    """
    total_tokens = 0

    with open(output_path, "wb") as file:
        for story in tqdm(dataset, desc=f"Tokenizing {output_path.name}"):
            text = story["text"]

            # Tokenize the story and append EOS
            token_ids = tokenizer.encode_story(text)

            # GPT-2 vocabulary IDs fit into unsigned 16-bit integers
            token_array = np.asarray(token_ids, dtype=np.uint16)

            # Write directly to disk instead of accumulating Python lists
            token_array.tofile(file)

            total_tokens += len(token_ids)

    return total_tokens


def main():
    # Initialize GPT-2 tokenizer
    tokenizer = GPT2Tokenizer()

    print("Vocabulary size:", tokenizer.vocab_size)
    print("EOS token ID:", tokenizer.eos_token_id)

    # Load previously saved datasets
    train_dataset = load_from_disk(str(TRAIN_DIR))
    validation_dataset = load_from_disk(str(VALIDATION_DIR))

    print("\nTraining stories:", len(train_dataset))
    print("Validation stories:", len(validation_dataset))

    # Tokenize and save each split independently
    train_tokens = tokenize_and_save(
        train_dataset,
        tokenizer,
        OUTPUT_DIR / "train.bin"
    )

    validation_tokens = tokenize_and_save(
        validation_dataset,
        tokenizer,
        OUTPUT_DIR / "validation.bin"
    )

    # Save metadata describing the binary files
    metadata = {
        "tokenizer": "gpt2",
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nTokenization completed!")
    print("Training tokens:", train_tokens)
    print("Validation tokens:", validation_tokens)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
