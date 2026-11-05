
from dataclasses import dataclass
from Tokenizer.tokenizer_gpt2 import GPT2Tokenizer

tokenizer = GPT2Tokenizer()

@dataclass
class GPTConfig:
    # Tokenizer
    vocab_size: int = tokenizer.vocab_size 

    # Transformer architecture
    block_size: int = 256 # sequence length
    n_embd: int = 256 # Embedding dimension
    n_layer: int = 4 # number of tranformer blocks
    n_head: int = 4 # number of head attentions

    # Regularization
    dropout: float = 0.1

    # Linear layers and LayerNorm
    bias: bool = True
