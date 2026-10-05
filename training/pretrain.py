from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import TransformerLM, count_parameters
from training.dataset import GermanTextDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/train/train.txt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
CHECKPOINT_DIR = Path("checkpoints")

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1

NUM_EPOCHS = 5

LOG_EVERY = 50
SAVE_EVERY = 500


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Model
# --------------------------------------------------

config = ModelConfig()

model = TransformerLM(config)

model = model.to(device)

print("Model parameters:", count_parameters(model))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = GermanTextDataset(
    text_path=DATA_PATH,
    tokenizer_path=TOKENIZER_PATH,
    context_length=config.context_length,
)

print("Training samples:", len(dataset))
print("Total tokens:", len(dataset.tokens))


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

print("Batches per epoch:", len(loader))


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=WEIGHT_DECAY,
)


# --------------------------------------------------
# Checkpoint directory
# --------------------------------------------------

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Training
# --------------------------------------------------

model.train()

global_step = 0

for epoch in range(NUM_EPOCHS):

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 60)

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # ------------------------------------------
        # 1. Forward pass
        # ------------------------------------------

        logits, loss = model(
            input_ids,
            labels,
        )

        # ------------------------------------------
        # 2. Clear previous gradients
        # ------------------------------------------

        optimizer.zero_grad()

        # ------------------------------------------
        # 3. Backpropagation
        # ------------------------------------------

        loss.backward()

        # ------------------------------------------
        # 4. Update model parameters
        # ------------------------------------------

        optimizer.step()

        global_step += 1

        # ------------------------------------------
        # Logging
        # ------------------------------------------

        if global_step % LOG_EVERY == 0:

            print(
                f"Step {global_step:5d} | "
                f"Loss: {loss.item():.4f}"
            )

        # ------------------------------------------
        # Save checkpoint
        # ------------------------------------------

        if global_step % SAVE_EVERY == 0:

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"checkpoint_step_{global_step}.pt"
            )

            torch.save(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "config": config.__dict__,
                },
                checkpoint_path,
            )

            print(
                f"Checkpoint saved: {checkpoint_path}"
            )


print()
print("Training complete.")
print("Final step:", global_step)