from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class GermanTextDataset(Dataset):
    """
    Converts German text into fixed-length
    input/target sequences for next-token prediction.
    """

    def __init__(
        self,
        text_path: str,
        tokenizer_path: str,
        context_length: int = 128,
    ):
        self.text_path = Path(text_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.context_length = context_length

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(
            str(self.tokenizer_path)
        )

        # Load text
        text = self.text_path.read_text(
            encoding="utf-8"
        )

        # Find EOS token
        self.eos_token_id = self.tokenizer.token_to_id("[EOS]")

        # Tokenize the complete dataset
        token_ids = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            encoded = self.tokenizer.encode(line)

            token_ids.extend(encoded.ids)

            # Separate sentences/documents
            if self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)

        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        # We need context_length + 1 because
        # target is shifted by one position.
        self.num_samples = (
            len(self.tokens) - self.context_length
        )

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, index):

        start = index
        end = start + self.context_length + 1

        chunk = self.tokens[start:end]

        # Input:
        # [1, 2, 3, 4]
        #
        # Target:
        # [2, 3, 4, 5]

        input_ids = chunk[:-1]
        target_ids = chunk[1:]

        return {
            "input_ids": input_ids,
            "labels": target_ids,
        }