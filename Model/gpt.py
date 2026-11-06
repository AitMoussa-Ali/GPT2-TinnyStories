import torch
from torch import nn
from Model.embedding import GPTEmbedding
from Model.transformer_block import GPTTransformerBlock
class GPTModel(nn.Module):

    def __init__(self, vocab_size : int, n_embed : int, block_size :int, n_head: int, n_layer: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_head = n_head
        self.n_layer = n_layer

        self.embedding = GPTEmbedding(vocab_size=vocab_size, n_embd=n_embed, block_size=block_size)
        self.transformer_block = nn.ModuleList([GPTTransformerBlock(n_embed=n_embed, block_size=block_size, n_head=n_head) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(normalized_shape=n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_block:
            x = block(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits