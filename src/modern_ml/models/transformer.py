import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch
@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.lin_fc    = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.lin_proj  = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.lin_fc(x)
        x = self.gelu(x)
        x = self.lin_proj(x)
        x = self.dropout(x)
        return x
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class Transformer(nn.Module):

    def __init__(self,config):
        '''
            block_size: int = 1024
            vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            num_layers: int = 12
            num_heads: int = 12
            embed_dim: int = 768
            dropout: float = 0.0
            bias: bool = True 
        '''
        super(Transformer,self).__init__()
        self.config = config
        self.token_emb = nn.Embedding(self.config.vocab_size,self.config.embed_dim)
        self.pos_emb = nn.Embedding(self.config.block_size,self.config.embed_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.config.num_layers)])
        self.final_ln = nn.LayerNorm(self.config.embed_dim,bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout)
        self.lm_head = nn.Linear(self.config.embed_dim,self.config.vocab_size,bias=False)

    def forward(self,x):
        tok_embed = self.token_emb(x)
        pos_embed = self.token_emb(x)
        y = self.dropout(tok_embed+pos_embed)
        for block in self.blocks:
            y = block(y)
        y = self.final_ln(y)
        print('y before lm head:',y.shape)
        return self.lm_head(y[:,[-1],:])
