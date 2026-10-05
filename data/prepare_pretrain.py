from pathlib import Path
import re
import random


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "raw"
CLEANED_DIR = BASE_DIR / "cleaned"
TRAIN_DIR = BASE_DIR / "train"
VALIDATION_DIR = BASE_DIR / "validation"

RAW_FILE = RAW_DIR / "german.txt"
CLEANED_FILE = CLEANED_DIR / "german_cleaned.txt"
TRAIN_FILE = TRAIN_DIR / "train.txt"
VALIDATION_FILE = VALIDATION_DIR / "validation.txt"

VALIDATION_RATIO = 0.02
SEED = 42


def clean_line(line: str) -> str:
    """Basic text normalization."""
    line = line.strip()

    # Collapse multiple spaces
    line = re.sub(r"\s+", " ", line)

    # Remove control characters while keeping German characters
    line = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", line)

    return line


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{RAW_FILE}\n\n"
            "Put your German raw text in data/raw/german.txt"
        )

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading raw German dataset...")

    lines = []

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_line(line)

            # Ignore very short lines
            if len(line) < 10:
                continue

            lines.append(line)

    print(f"Raw usable lines: {len(lines):,}")

    # Deduplicate
    lines = list(dict.fromkeys(lines))

    print(f"After deduplication: {len(lines):,}")

    if len(lines) < 100:
        raise ValueError(
            "Dataset is too small. Add more German text to data/raw/german.txt"
        )

    # Shuffle deterministically
    random.seed(SEED)
    random.shuffle(lines)

    validation_size = max(1, int(len(lines) * VALIDATION_RATIO))

    validation_lines = lines[:validation_size]
    train_lines = lines[validation_size:]

    # Save cleaned dataset
    with open(CLEANED_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Save train dataset
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    # Save validation dataset
    with open(VALIDATION_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(validation_lines) + "\n")

    print("\nDataset preparation complete.")
    print(f"Train lines:      {len(train_lines):,}")
    print(f"Validation lines: {len(validation_lines):,}")
    print(f"\nCreated:")
    print(f"  {CLEANED_FILE}")
    print(f"  {TRAIN_FILE}")
    print(f"  {VALIDATION_FILE}")


if __name__ == "__main__":
    main()