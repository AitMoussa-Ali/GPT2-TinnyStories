from torch import nn
import torch
import math
class GPTAttention(nn.Module):

    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.size_attention = n_embed // n_head
        self.block_size = block_size

        self.key = nn.Linear(n_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.wo = nn.Linear(n_embed, n_embed)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):

        k = self.key(x) #(batch_size, block_size, n_embed)
        q = self.query(x)
        v = self.value(x)

        k = k.reshape(x.shape[0], x.shape[1], self.n_head, self.size_attention).transpose(1, 2)
        q = q.reshape(x.shape[0], x.shape[1], self.n_head, self.size_attention).transpose(1, 2)
        v = v.reshape(x.shape[0], x.shape[1], self.n_head, self.size_attention).transpose(1, 2)
        
        T = x.shape[1]

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.size_attention)

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(scores, dim=-1)
        
        qkv = torch.matmul(attention_weights, v) #(batch_size, n_head, block_size, size_attention)

        qkv = qkv.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], self.n_head * self.size_attention)

        return self.wo(qkv)

