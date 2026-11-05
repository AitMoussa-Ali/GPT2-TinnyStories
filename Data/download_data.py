from datasets import load_dataset

DATASET_NAME = "roneneldan/TinyStories"
DATASET_SIZE = 100_000
VALIDATION_SIZE = 0.1
RANDOM_SEED = 42

OUTPUT_DIR = "data/processed/tinystories"


def load_stories():
    dataset = load_dataset(
        DATASET_NAME,
        split=f"train[:{DATASET_SIZE}]"
    )

    return dataset


def split_dataset(dataset):
    splits = dataset.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=RANDOM_SEED,
        shuffle=True
    )

    return splits["train"], splits["test"]


def save_dataset(train_dataset, validation_dataset):
    train_dataset.save_to_disk(f"{OUTPUT_DIR}/train")
    validation_dataset.save_to_disk(f"{OUTPUT_DIR}/validation")


def main():
    dataset = load_stories()

    print("Original dataset:")
    print(dataset)
    print("\nFirst story:")
    print(dataset[0]["text"])

    train_dataset, validation_dataset = split_dataset(dataset)

    print("\nTraining stories:", len(train_dataset))
    print("Validation stories:", len(validation_dataset))

    save_dataset(train_dataset, validation_dataset)

    print("\nDatasets saved successfully!")


if __name__ == "__main__":
    main()
