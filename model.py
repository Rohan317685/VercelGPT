import math 
import torch 
import torch.nn as nn
from torch.nn import functional as F

from data import block_size, vocab_size

n_embd = 128
n_head = 4 
n_layer = 4 
dropout = 0.1


class SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).contiguous()().view(B, T, C)
        return self.drop(self.proj(out))

class Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn = SelfAttention()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.GELU(),

            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x 

class GPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block() for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)

        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        
        self.apply(self._init)


        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):

                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

def forward(self, idx, targets=None):

    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

    for block in self.blocks:
        x = block(x)

    logits = self.head(self.ln_f(x))

    if targets is None:
        return logits, None 


        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.reshape(-1))

        return logits, loss

@torch.no_grad()

def generate(self, idx, max_new_tokens, temperature = 0.8, top_k=40):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = self (idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf") 

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx 