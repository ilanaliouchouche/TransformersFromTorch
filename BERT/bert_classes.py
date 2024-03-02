import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Embeddings(nn.Module):
    '''
    Embeddings Class:
    - This class is responsible for creating token embeddings and position embeddings.
    - The embeddings are then normalized and dropout is applied.
    '''

    def __init__(self, 
                 vocab: int, 
                 embedding_dim: int, 
                 position_size: int) -> None:
        '''
        Constructor for the Embeddings class.

        Parameters:
            - vocab (int): The size of the vocabulary.
            - embedding_dim (int): The dimension of the token embeddings.
            - position_size (int): The maximum position size.
        '''
        
        super().__init__()
        self.embeddings = nn.Embedding(vocab, embedding_dim)
        self.positions = nn.Embedding(position_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, 
                input_ids: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the Embeddings class.

        Parameters:
            - input_ids (torch.Tensor): The input token IDs.

        Returns:
            - torch.Tensor: The token embeddings with position embeddings added.
        '''

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.positions(position_ids)
        all_embeddings = token_embeddings + position_embeddings
        all_embeddings = self.norm(all_embeddings)
        all_embeddings = self.dropout(all_embeddings)
        return all_embeddings


def scaled_dot_product(q: torch.Tensor, 
                       k: torch.Tensor, 
                       v: torch.Tensor) -> torch.Tensor:
    '''
    Scaled Dot Product Function:
    - This function calculates the attention scores and returns the attention head.

    Parameters:
        - q (torch.Tensor): The query tensor.
        - k (torch.Tensor): The key tensor.
        - v (torch.Tensor): The value tensor.
    
    Returns:
        - torch.Tensor: The attention head.
    '''

    dim_k = k.size(-1)
    scores = torch.bmm(q, k.transpose(1, 2)) / sqrt(dim_k)
    softed = F.softmax(scores, dim=-1)
    attn_head = torch.bmm(softed, v)
    return attn_head


class AttentionHead(nn.Module):
    '''
    AttentionHead Class:
    - This class applies linear transformations to the input (query, key, value).
    - Then it computes the scaled dot product attention.
    '''

    def __init__(self, 
                 embedding_dim : int,
                 head_dim : int) -> None:
        '''
        Constructor for the AttentionHead class.

        Parameters:
            - embedding_dim (int): The dimension of the input embeddings.
            - head_dim (int): The dimension of the attention head.
        '''

        super().__init__()
        self.q = nn.Linear(embedding_dim, head_dim)
        self.k = nn.Linear(embedding_dim, head_dim)
        self.v = nn.Linear(embedding_dim, head_dim)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the AttentionHead class.

        Parameters:
            - x (torch.Tensor): The input embeddings.

        Returns:    
            - torch.Tensor: The output tensor after applying scaled dot product attention.
        '''

        x = x.float()
        return scaled_dot_product(self.q(x), self.k(x), self.v(x))

class MultiHeadAttention(nn.Module):
    '''
    MultiHeadAttention Class:
    - This class contains multiple attention heads.
    - The outputs of these heads are concatenated and linearly transformed.
    '''

    def __init__(self, 
                 embedding_dim: int, 
                 num_attention_heads: int) -> None:
        '''
        Constructor for the MultiHeadAttention class.

        Parameters:
            - embedding_dim (int): The dimension of the embeddings.
            - num_attention_heads (int): The number of attention heads.
        '''

        super().__init__()
        head_dim = embedding_dim // num_attention_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, head_dim) for _ in range(num_attention_heads)]
        )
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the MultiHeadAttention class.

        Parameters:
            - x (torch.Tensor): The input embeddings.

        Returns:
            - torch.Tensor: The output tensor after applying multi-head attention.
        '''

        x = x.float()
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.fc(out)
        return out

class FeedForward(nn.Module):
    '''
    FeedForward Class:
    - This class represents the feed-forward neural network in the transformer model.
    - It applies two linear transformations with a GELU activation in between.
    '''

    def __init__(self, 
                 embedding_dim: int, 
                 intermediate_dim: int) -> None:
        '''
        Constructor for the FeedForward class.

        Parameters:
            - embedding_dim (int): The dimension of the embeddings.
            - intermediate_dim (int): The dimension of the intermediate layer.
        '''

        super().__init__()
        self.l1 = nn.Linear(embedding_dim, intermediate_dim)
        self.l2 = nn.Linear(intermediate_dim, embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForward class.

        Parameters:
            - x (torch.Tensor): The input embeddings.

        Returns:
            - torch.Tensor: The hidden state after applying the feed-forward neural network.
        '''

        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x
    
class Encoder(nn.Module):
    '''
    Encoder Class:
    - This class represents the encoder layer of the transformer model.
    - It applies layer normalization, multi-head attention, and feed-forward neural network sequentially.
    '''

    def __init__(self, 
                 embedding_dim: int, 
                 intermediate_dim: int, 
                 num_attention_heads: int) -> None:
        '''
        Constructor for the Encoder class.

        Parameters:
            - embedding_dim (int): The dimension of the embeddings.
            - intermediate_dim (int): The dimension of the intermediate layer.
            - num_attention_heads (int): The number of attention heads.
        '''

        super().__init__()
        self.l1 = nn.LayerNorm(embedding_dim)
        self.l2 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_attention_heads)
        self.feed_forward = FeedForward(embedding_dim, intermediate_dim)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the Encoder class.

        Parameters:
            - x (torch.Tensor): The input embeddings.

        Returns:
            - torch.Tensor: The hidden state after applying the encoder layer.
        '''

        out = self.l1(x)
        x = x + self.attention(out)
        x = x + self.feed_forward(self.l2(x))
        return x
