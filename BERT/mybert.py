from bert_classes import *

class MyBert(nn.Module):
    '''
    MyBert Class:
    - This class represents the complete BERT model architecture.
    - It initializes the embedding layer and a specified number of transformer encoder layers.
    - During the forward pass, it applies the embeddings and passes the output through each encoder layer.
    '''

    def __init__(self, 
                 vocab: int = 30522, 
                 embedding_dim: int = 768, 
                 position_size: int = 512, 
                 n_layers: int = 12, 
                 intermediate_dim: int = 3072, 
                 num_attention_heads: int = 12) -> None:
        '''
        Constructor for the MyBert class.

        Parameters:
            - vocab (int): The size of the vocabulary.
            - embedding_dim (int): The dimension of the embeddings.
            - position_size (int): The maximum position size for the positional embeddings.
            - n_layers (int): The number of transformer encoder layers.
            - intermediate_dim (int): The dimension of the intermediate layer in the feed-forward neural network.
            - num_attention_heads (int): The number of attention heads.
        '''

        super().__init__()
        
        self.embeddings = Embeddings(vocab, embedding_dim, position_size)
        
        self.layers = nn.ModuleList(
            [Encoder(embedding_dim, intermediate_dim, num_attention_heads) for _ in range(n_layers)]
        )

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the MyBert class.

        Parameters:
            - x (torch.Tensor): The input IDs.

        Returns:
            - torch.Tensor: The hidden state after applying the BERT model.
        '''

        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x


