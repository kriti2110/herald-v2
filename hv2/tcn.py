"""Causal multi-task temporal CNN: one shared encoder, two heads
(forecast = onset within 6 h; detect = SepsisLabel now).
Causality: left-only padding, per-timestep LayerNorm (never over time),
no bidirectional ops. Right-padding of batches cannot affect earlier steps."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, dil, drop):
        super().__init__()
        self.pad = 2 * dil
        self.conv = nn.Conv1d(d, d, 3, dilation=dil)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                      # x: (B,T,d)
        h = F.pad(x.transpose(1, 2), (self.pad, 0))
        h = self.conv(h).transpose(1, 2)
        return x + self.drop(F.gelu(self.norm(h)))


class CausalMTL(nn.Module):
    def __init__(self, n_in, d=96, dils=(1, 2, 4, 8, 16, 32), drop=0.15):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.blocks = nn.ModuleList([Block(d, k, drop) for k in dils])
        self.head = nn.Linear(d, 2)            # [:,:,0]=forecast  [:,:,1]=detect

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)
