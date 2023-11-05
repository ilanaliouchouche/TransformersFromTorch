# Import all classes and functions from bert_classes
from bert_classes import *

'''
MyBert Class:
- This class represents the complete BERT model architecture.
- It initializes the embedding layer and a specified number of transformer encoder layers.
- During the forward pass, it applies the embeddings and passes the output through each encoder layer.
'''
class MyBert(nn.Module):
    def __init__(self, vocab=30522, embedding_dim=768, position_size=512, 
                 n_layers=12, intermediate_dim=3072, num_attention_heads=12):
        super().__init__()
        
        # Initialize the embeddings layer
        self.embeddings = Embeddings(vocab, embedding_dim, position_size)
        
        # Initialize the specified number of encoder layers
        self.layers = nn.ModuleList(
            [Encoder(embedding_dim, intermediate_dim, num_attention_heads) for _ in range(n_layers)]
        )

    def forward(self, x):
        # Apply embeddings to the input tokens
        x = self.embeddings(x)
        
        # Pass the embeddings through each encoder layer
        for layer in self.layers:
            x = layer(x)
        
        # Return the output of the final layer
        return x


