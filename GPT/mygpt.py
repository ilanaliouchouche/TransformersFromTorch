from gpt_classes import *

class MyGPT(nn.Module):
  def __init__(self,  vocab=50257, embedding_dim=768, position_size=1024, n_layers=12,intermediate_dim=3072, num_attention_heads=12):
    super().__init__()
    self.embeddings = Embeddings(vocab, embedding_dim, position_size)
    self.layers = nn.ModuleList([Decoder(embedding_dim, intermediate_dim, num_attention_heads) for _ in range(n_layers)])

  def forward(self, x):
    x = self.embeddings(x)
    for layer in self.layers:
      x = layer(x)
    return x