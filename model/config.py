from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Tokenizer
    vocab_size: int = 577

    # Sequence
    context_length: int = 128

    # Transformer
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384

    # Regularization
    dropout: float = 0.0

    # RoPE
    rope_theta: float = 10000.0

    # RMSNorm
    rms_norm_eps: float = 1e-5