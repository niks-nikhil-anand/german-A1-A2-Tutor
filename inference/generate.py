import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.config import ModelConfig
from model.transformer import TransformerLM


# ============================================
# Configuration
# ============================================

CHECKPOINT_PATH = "checkpoints/checkpoint_step_5500.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

MAX_NEW_TOKENS = 30

TEMPERATURE = 0.8

TOP_K = 20


# ============================================
# Device
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================
# Load tokenizer
# ============================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print(
    "Vocabulary:",
    tokenizer.get_vocab_size()
)


# ============================================
# Load model
# ============================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

config = ModelConfig()

model = TransformerLM(config)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    "Checkpoint step:",
    checkpoint["step"]
)

print(
    "Training loss:",
    checkpoint["loss"]
)


# ============================================
# Generation function
# ============================================

def generate(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
):

    # Convert prompt to token IDs
    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Don't exceed model context length
            input_for_model = input_ids[
                :, -config.context_length:
            ]

            # Model prediction
            logits, _ = model(
                input_for_model
            )

            # Take prediction for final token
            next_token_logits = logits[
                :, -1, :
            ]

            # Temperature
            next_token_logits = (
                next_token_logits / temperature
            )

            # Top-K sampling
            if top_k is not None:

                values, indices = torch.topk(
                    next_token_logits,
                    min(top_k, next_token_logits.size(-1)),
                )

                filtered_logits = torch.full_like(
                    next_token_logits,
                    float("-inf"),
                )

                filtered_logits.scatter_(
                    1,
                    indices,
                    values,
                )

                next_token_logits = filtered_logits

            # Convert logits to probabilities
            probabilities = F.softmax(
                next_token_logits,
                dim=-1,
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            # Add token to sequence
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

    # Convert IDs back to text
    generated_text = tokenizer.decode(
        input_ids[0].tolist()
    )

    return generated_text


# ============================================
# Test prompts
# ============================================

prompts = [
    "Ich bin",
    "Das ist",
    "Ich lerne",
    "Hallo",
]


for prompt in prompts:

    print()
    print("=" * 60)
    print("Prompt:", prompt)

    result = generate(prompt)

    print("Generated:", result)