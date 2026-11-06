from torch import nn
import torch
from Model.attention import GPTAttention

class GPTFeedForward(nn.Module):

    def __init__(self, n_embed: int):

        super().__init__()
        self.linear1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear2 = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()

    def forward(self, x):

        x = self.activation(self.linear1(x))
        x = self.linear2(x)

        return self.dropout(x)

class GPTTransformerBlock(nn.Module):

    def __init__(self, n_embed: int, block_size: int, n_head: int):

        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_head = n_head

        self.layer_norm1 = nn.LayerNorm(normalized_shape=n_embed)
        self.head_attention = GPTAttention(n_embed=n_embed, n_head=n_head, block_size=block_size)
        self.dropout1 = nn.Dropout(0.1)
        self.feed_forward = GPTFeedForward(n_embed=n_embed)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=n_embed)

    def forward(self, x):

        # multihead attention
        x = x + self.dropout1(self.head_attention(self.layer_norm1(x)))

        # feed forward
        x = x + self.feed_forward(self.layer_norm2(x))

        return x