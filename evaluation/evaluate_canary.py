import torch
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import TransformerLM
from training.dataset import GermanTextDataset


# ---------------------------------------------
# Configuration
# ---------------------------------------------

CHECKPOINT_PATH = "checkpoints/checkpoint_step_5500.pt"

TRAIN_DATA = "data/train/train.txt"
VALIDATION_DATA = "data/validation/validation.txt"

TOKENIZER_PATH = "tokenizer/tokenizer.json"

BATCH_SIZE = 16


# ---------------------------------------------
# Device
# ---------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------
# Load checkpoint
# ---------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)


# ---------------------------------------------
# Create model
# ---------------------------------------------

config = ModelConfig()

model = TransformerLM(config)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print("Checkpoint step:", checkpoint["step"])
print("Training loss at checkpoint:", checkpoint["loss"])


# ---------------------------------------------
# Create datasets
# ---------------------------------------------

train_dataset = GermanTextDataset(
    TRAIN_DATA,
    TOKENIZER_PATH,
    config.context_length,
)

validation_dataset = GermanTextDataset(
    VALIDATION_DATA,
    TOKENIZER_PATH,
    config.context_length,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)


# ---------------------------------------------
# Evaluation function
# ---------------------------------------------

def evaluate(loader):

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(
                input_ids,
                labels,
            )

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


# ---------------------------------------------
# Evaluate
# ---------------------------------------------

train_loss = evaluate(train_loader)
validation_loss = evaluate(validation_loader)


print()
print("=" * 50)
print("CANARY EVALUATION")
print("=" * 50)

print(f"Train loss:      {train_loss:.4f}")
print(f"Validation loss: {validation_loss:.4f}")

print("=" * 50)