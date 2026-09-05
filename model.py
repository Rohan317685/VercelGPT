import math 
import torch 
import torch.nn as nn
from torch.nn import functional as F

from data import block_size, vocab_size

n_embd = 128
n_head = 4 
n_layer = 4 
dropout = 0.1


class SelfAttention(nn.module):

    def __init__(self):
        super().__init__()
        self.qkv = nn.linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, 1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).continguouse().view(B, T, C)
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
        super()__init__()

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block() for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)

        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        
        self.apply(self._init)
