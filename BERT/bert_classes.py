# flake8: noqa

import torch
from torch import nn
from torch.nn import functional as F
import math

class BERTConfig(object):
    vocab_size: int
    n_layers: int
    embedding_dim: int
    max_ctx_len: int
    intermediate_dim: int
    dropout: float
    num_attn_heads: int

class Embedder(nn.Module):

    def __init__(self,
                 config: BERTConfig) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.embedding_dim
        )
        self.position_codes = nn.Embedding(
            config.max_ctx_len, config.embedding_dim
        )
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        _, L = x.size()
        indices = torch.arange(0, L, device=x.device)
        
        return self.embedding_layer(x) + self.position_codes(indices)

class Encoder(nn.Module):

    def __init__(self,
                 config: BERTConfig) -> None:

        super().__init__()

        self.H = config.num_attn_heads
        self.lnorm1 = nn.LayerNorm(config.embedding_dim)
        
        self.pre_mha = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
        self.att_dropout = nn.Dropout(config.dropout)
        self.post_mha = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.posta_dropout = nn.Dropout(config.dropout)

        self.lnorm2 = nn.LayerNorm(config.embedding_dim)

        self.ffw = nn.Sequential(
            nn.Linear(config.embedding_dim, config.intermediate_dim),
            nn.Linear(config.intermediate_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

    def forward(self,
                x: torch.Tensor,  # B, L, D
                attention_mask: torch.Tensor) -> torch.Tensor:  # B, L

        out = self.lnorm1(x)

        qkv: torch.Tensor = self.pre_mha(out)  # B, L, 3*D
        B, L, D3 = qkv.size()
        qkv = qkv.view(B, L, self.H, D3//self.H)  # B, L, H, 3*D//H
        qkv = qkv.permute(0, 2, 1, 3)  # B, H, L, 3*D//H
        q, k, v = torch.chunk(qkv, 3, -1)  # 3x (B, H, L, D//H)
        qkt = torch.einsum("bhld,bhmd->bhlm", q, k)  # B, H, L, L
        qkt = qkt / math.sqrt(q.size(-1))
        qkt: torch.Tensor = self.att_dropout(qkt)
        mask = attention_mask.unsqueeze(1).unsqueeze(1).to(x.device)
        masked_qkt = qkt.masked_fill(mask=~mask, value=-torch.inf)
        masked_qkt = F.softmax(masked_qkt, dim=-1)  # B, H, L, L
        out_attn = torch.einsum("bhlm,bhmd->bhld", masked_qkt, v)  # B, H, L, D//H
        out_attn = out_attn.permute(0, 2, 1, 3).contiguous()  # B, L, H, D//H
        out_attn = out_attn.view(B, L, -1)  # B, L, D
        out_attn = self.posta_dropout(out_attn)

        out_attn = out_attn + x  # B, L, D
        out = self.lnorm2(out_attn)

        out = self.ffw(out)  # B, L, D
        out = out_attn + out  

        return out  # B, L, D
