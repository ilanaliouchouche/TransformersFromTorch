import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

'''
Embeddings Class:
- This class is responsible for creating token and position embeddings.
- The embeddings are normalized and dropout is applied for regularization.
'''
class Embeddings(nn.Module):
    def __init__(self, vocab, embedding_dim, position_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab, embedding_dim)
        self.positions = nn.Embedding(position_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-05)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.positions(position_ids)
        all_embeddings = token_embeddings + position_embeddings
        all_embeddings = self.norm(all_embeddings)
        all_embeddings = self.dropout(all_embeddings)
        return all_embeddings  # tokens * embedding_dim

'''
Scaled Dot Product Attention:
- This function calculates the attention weights and applies them to the value matrix.
- A mask is applied to allow the model to attend only to past tokens.
'''
def scaled_dot_product(q, k, v, masked : bool = False):
    dim_k = k.size(-1)
    seq_len = k.size(1)
    scores = torch.bmm(q, k.transpose(1, 2)) / sqrt(dim_k)
    if masked:
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    softed = F.softmax(scores, dim=-1)
    attn_head = torch.bmm(softed, v)
    return attn_head

'''
AttentionHead Class:
- Represents a single attention head.
- Applies linear transformations to the input, and calculates attention using scaled dot product.
'''
class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim, masked : bool = False):
        super().__init__()
        self.q = nn.Linear(embedding_dim, head_dim)
        self.k = nn.Linear(embedding_dim, head_dim)
        self.v = nn.Linear(embedding_dim, head_dim)
        self.masked = masked

    def forward(self, x):
        return scaled_dot_product(self.q(x), self.k(x), self.v(x), self.masked)

'''
MultiHeadAttention Class:
- Contains multiple attention heads.
- The outputs of the heads are concatenated and linearly transformed.
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, masked : bool = False):
        super().__init__()
        head_dim = embedding_dim // num_attention_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, head_dim, masked) for _ in range(num_attention_heads)]
        )
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.fc(out)
        return out

'''
FeedForward Class:
- Implements the feed-forward neural network present in each transformer block.
- Consists of two linear layers with a GELU activation in between.
'''
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, intermediate_dim):
        super().__init__()
        self.l1 = nn.Linear(embedding_dim, intermediate_dim)
        self.l2 = nn.Linear(intermediate_dim, embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x

'''
Decoder Class:
- Represents a single transformer block.
- Consists of multi-head attention and feed-forward neural network with layer normalization.
'''
class Decoder(nn.Module):
    def __init__(self, embedding_dim, intermediate_dim, num_attention_heads):
        super().__init__()
        self.l1 = nn.LayerNorm(embedding_dim)
        self.l2 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_attention_heads, True)
        self.feed_forward = FeedForward(embedding_dim, intermediate_dim)

    def forward(self, x):
        x = self.l1(x + self.attention(x))
        x = self.l2(x + self.feed_forward(x))
        return x

'''
Encoder Class:
- This class represents the encoder layer of the transformer model.
- It applies layer normalization, multi-head attention, and feed-forward neural network sequentially.
'''
class Encoder(nn.Module):
    def __init__(self, embedding_dim, intermediate_dim, num_attention_heads):
        super().__init__()
        self.l1 = nn.LayerNorm(embedding_dim)
        self.l2 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_attention_heads)
        self.feed_forward = FeedForward(embedding_dim, intermediate_dim)

    def forward(self, x):
        out = self.l1(x)
        x = x + self.attention(out)
        x = x + self.feed_forward(self.l2(x))
        return x

