from torch import nn
import torch
class GPTEmbedding(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        print(x.shape)
        batch_size, block_size = x.shape
        positions = torch.arange(block_size, device=x.device).unsqueeze(0).expand(batch_size, -1)

        #apply the embeddings
        tokens_embeddings = self.token_embedding(x) 
        position_embeddings = self.position_embedding(positions)

        return self.dropout(tokens_embeddings + position_embeddings)