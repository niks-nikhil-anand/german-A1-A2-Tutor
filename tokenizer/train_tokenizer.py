from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "train" / "train.txt"
OUTPUT_DIR = BASE_DIR / "tokenizer"

TOKENIZER_FILE = OUTPUT_DIR / "tokenizer.json"
VOCAB_FILE = OUTPUT_DIR / "vocab.json"
MERGES_FILE = OUTPUT_DIR / "merges.txt"


VOCAB_SIZE = 8192


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_FILE}"
        )

    print("Training German BPE tokenizer...")
    print(f"Dataset: {TRAIN_FILE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
        ],
    )

    tokenizer.train(
        files=[str(TRAIN_FILE)],
        trainer=trainer,
    )

    tokenizer.save(str(TOKENIZER_FILE))

    # Export vocabulary and merges separately.
    model = tokenizer.model

    if hasattr(model, "save"):
        model.save(str(OUTPUT_DIR))

    print("\nTokenizer training complete.")
    print(f"Tokenizer: {TOKENIZER_FILE}")
    print(f"Vocabulary: {VOCAB_FILE}")
    print(f"Merges: {MERGES_FILE}")

    print("\nActual vocabulary size:")
    print(tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()