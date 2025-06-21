# flake8: noqa

from gpt_classes import *

class GPT(nn.Module):

    def __init__(self,
                 config: GPTConfig) -> None:
        
        super().__init__()
        self.embedder = Embedder(config)
        self.decoders = nn.ModuleList(
            [Decoder(config) for _ in range(config.n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        )

        self.apply(self.init_weights)

        self.classifier[1].weight = self.embedder.embedding_layer.weight

    def init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:  # B, L
        
        x = self.embedder(x)  # B, L, D
        for layer in self.decoders:
            x = layer(x)  # B, L ,D
        x = self.classifier(x)  # B, L, V
        
        return x
