import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    num_layers: int = 12
    num_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster



class Transformer(nn.Module):

    def __init__(self,config):
        self.super(Transformer,self).__init__()
        self.vocab_size = vocab_size

    def forward(self,x):
