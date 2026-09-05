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