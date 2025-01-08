import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class Config:
    max_position_embeddings: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)  # q, k, v shape: (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # q shape: (B, n_head, T, C//n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = torch.nn.functional.scale_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_position_embeddings is not None
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.num_params = (
            sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()
        ) / 1e6  # million

    def forward(self, x, y=None):
        B, T = x.size()
        assert (
            T <= self.config.max_position_embeddings
        ), f"Input length {T} is greater than maximum position embedding {self.config.max_position_embeddings}"
        pos_ids = torch.arange(T, device=x.device, dtype=x.dtype).expand(B, T)
        x = self.tok_emb(x) + self.pos_emb(pos_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if y is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            )  # ??
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
