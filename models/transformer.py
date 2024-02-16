import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import sqrt
from torch.optim import Adam
from typing import Optional, Tuple

class Transformer(nn.Module):
    """
    Args:
        - n (int): number of Transformers Blocks (or EncoderDecoder);
        - dm (int): output dimension;
        - dff (int): feed-forward dimension;
        - heads (int): number of heads in Multi-head attention;
        - source_vocab (any):
        - target_vocab (any):
        - max_length (int):
        - dk (int):
        - dv (int):
        - dropout (float):
        - bias (bool): 
    """
    def __init__(self, n: int, dm: int, dff: int, heads: int,
                 source_vocab: any, target_vocab: any, max_len: int=128, 
                 dk: int=-1, dv: int=-1, dropout: float=0.1,
                 bias: bool=True, device: str='cuda'):
        super().__init__()

        self.n = n
        self.dm = dm
        self.dff = dff
        self.heads = heads
        self.max_len = max_len

        if dk < 0: dk=dm
        if dv < 0: dv=dm
        
        # --- building embeddings and positional encoding ---

        self.src_embedding_layer = TransformerEmbedding(source_vocab, dm)
        self.trg_embedding_layer = TransformerEmbedding(target_vocab, dm)
        
        self.positional_layer = PositionalEncoding(max_len, dm, device)

        self.output = nn.Linear(dm, len(target_vocab.get_itos()), bias=bias)
        
        self.encoders = nn.ModuleList([
            Encoder(heads, dff, dm, dk // heads, dv // heads, dropout, bias) for _ in range(n)
        ])

        self.decoders = nn.ModuleList([
            Decoder(heads, dff, dm, dk // heads, dv // heads, dropout, bias) for _ in range(n)
        ])

        plot_positional_encoding(self.positional_layer)
        
    def forward(self, x_src, x_trg):
        
        # --- padding mask ---

        src_mask = (x_src != 0).float()
        trg_mask = (x_trg != 0).float()
        
        # --- source embeddings + PE ---

        x_src = self.src_embedding_layer(x_src)
        x_src = self.positional_layer(x_src)

        # --- target embeddings + PE ---

        x_trg = self.trg_embedding_layer(x_trg)
        x_trg = self.positional_layer(x_trg)

        # --- Encoder-Decoder  ---

        for encoder in self.encoders:
           x_enc = encoder(x_src, src_mask)
           x_src = x_enc

        for decoder in self.decoders:
           x_dec = decoder(x_trg, x_src, x_src, trg_mask, src_mask)
           x_trg = x_dec
        
        # --- Output ---

        X = self.output(x_dec)

        return X

class SelfAttention(nn.Module):

    def __init__(self, dm: int, mask: bool):
        super().__init__()

        self.dm = dm
        self.mask = mask
    
    def _padding_mask(self, mask, scores):

        mask = torch.unsqueeze(torch.unsqueeze(mask, 1), 1)
        
        B, H, T, T = scores.size()

        padding_mask = mask.repeat(1, 1, H, T).view(B, -1, T, T)
        new_scores = scores * padding_mask

        return new_scores.masked_fill(new_scores == 0, float('-inf'))

    def forward(self, q, k, v, f, mask):
        
        scores = torch.div(torch.matmul(q, k), f)
        
        # --- padding mask ---

        if mask != None:
            scores = self._padding_mask(mask, scores)
        
        # --- upper triangular mask ---

        if self.mask:
            scores = torch.triu(torch.full(scores.shape, float('-inf')), diagonal=1)
        
        return torch.matmul(F.softmax(scores.to('cuda'), dim=-1), v)

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, dm: int, d_k: int, d_v: int, bias: bool, mask: bool):
        super().__init__()

        self.num_heads = num_heads
        self.dm = dm
        self.d_k = d_k
        self.d_v = d_v
        
        # --- parameters ---

        self.w_q = nn.Parameter(torch.FloatTensor(num_heads, dm, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(num_heads, dm, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(num_heads, dm, d_v))
        
        # --- inicialization --- 

        std = 1.0 / math.sqrt(d_k)
        self.w_q.data.normal_(mean=0, std=std)
        self.w_k.data.normal_(mean=0, std=std)

        std = 1.0 / math.sqrt(d_v)
        self.w_v.data.normal_(mean=0, std=std)

        self.linear = nn.Linear(d_v * num_heads, dm, bias=bias)
        
        self.attention_mechanism = SelfAttention(dm, mask)

    def forward(self, xq, xk, xv, mask):
        
        batch_size, sequence_length, _ = xq.size()
        
        xq = xq.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        xk = xk.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        xv = xv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # --- Scaled Dot-Product Attention ---
        
        # query, key, value and scale factor
        query = torch.matmul(xq, self.w_q)
        key = torch.matmul(xk, self.w_k).transpose(2, 3)
        value = torch.matmul(xv, self.w_v)
        scale_factor = torch.tensor(self.d_k, dtype=torch.float)

        out = self.attention_mechanism(query, key, value, scale_factor, mask)
        
        concat = out.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, -1)

        return self.linear(concat)

class Encoder(nn.Module):

    def __init__(self, heads: int, d_ff: int, dm: int, d_k: int, d_v: int, dropout: float, bias=bool):
        super().__init__()
        self.mha = MultiHeadAttention(heads, dm, d_k, d_v, bias, mask=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(dm, eps=1e-6)
        self.ffn = PositionWiseFFS(dm, d_ff, bias)

    def forward(self, X, src_mask):
        
        residual = X

        # --- multi-head attention without mask ---
        
        X = self.dropout(self.mha(X, X, X, src_mask))
        X = self.layer_norm(residual + X)
        
        # --- feed forward ---
        
        X = self.ffn(X)
        
        return X

class Decoder(nn.Module):

    def __init__(self, heads: int, d_ff: int, dm: int, d_k: int, d_v: int, dropout: float, bias=bool):
        super().__init__()
        self.mha = MultiHeadAttention(heads, dm, d_k, d_v, bias, mask=True)
        self.cross_attention = MultiHeadAttention(heads, dm, d_k, d_v, bias, mask=None)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm_mask_attention = nn.LayerNorm(dm, eps=1e-6)
        self.layer_norm_cross_attention = nn.LayerNorm(dm, eps=1e-6)
        self.ffn = PositionWiseFFS(dm, d_ff, bias)

    def forward(self, X, k, v, trg_mask, src_mask):
        
        residual = X

        # --- masked multi-head attention with mask ---

        X = self.dropout(self.mha(X, X, X, trg_mask))
        X = self.layer_norm_mask_attention(residual + X)

        # --- multi-head attention with cross-attention---
        
        residual = X

        X = self.dropout(self.cross_attention(X, k, v, src_mask))
        X = self.layer_norm_cross_attention(residual + X)
        
        # --- feed forward ---
        
        X = self.ffn(X)

        return X

class PositionWiseFFS(nn.Module):

    def __init__(self, dm: int, d_ff: int, bias: bool):
        super().__init__()
        self.dm = dm
        self.d_ff = d_ff
        self.layer_norm = nn.LayerNorm(dm, eps=1e-6)
        self.layer1 = nn.Linear(dm, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, dm, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, X):
         
        residual = X

        X = self.relu(self.layer1(X))
        X = self.relu(self.layer2(X))
        X = self.layer_norm(residual + X)

        return X

class TransformerEmbedding(nn.Module):

    def __init__(self, vocab, dm):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab.get_itos()), 
            embedding_dim=dm,
            padding_idx=vocab.get_stoi()['<pad>']
        )
         
    def forward(self, X):

        return self.embedding(X) * math.sqrt(self.embedding.embedding_dim)

#TODO
class AdamWithWarmupAndLLRD(Adam):

    def __init__(self, *args, dim=512, warmup_steps=4000, **kwargs):
        super(AdamWithWarmup, self).__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.dim = dim
        self.lr_history = []

    def step(self, closure=None):
        
        if self.current_step < self.warmup_steps:
           lr = (self.dim**(-.5)) * min(self.current_step**(-.5), \
                self.current_step * self.warmup_steps**(-1.5))

           for group in self.param_groups:
                group['lr'] = lr * group['lr']

           self.lr_history.append(lr)

        super(AdamWithWarmup, self).step(closure)

        self.current_step += 1

class PositionalEncoding(nn.Module):
    
    def __init__(self, max_len, dm, device):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        wk = torch.exp(torch.arange(0, dm, 2).float() * -(torch.log(torch.tensor(10000.0)) / dm))
        
        pos_enc = torch.zeros((max_len, dm))
        
        pos_enc[:, 0::2] = torch.sin(pos * wk).to(device)
        pos_enc[:, 1::2] = torch.cos(pos * wk).to(device)
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))
        
    def forward(self, X):

        return X + self.pos_enc[:X.size()[1], :]

# --- Plots ---

def plot_positional_encoding(positional_layer):

    plt.imshow(positional_layer.pos_enc.detach().numpy().T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Positional Encoding Matrix")
    plt.xlabel("Position")
    plt.ylabel("Dimension")

    plt.savefig('positional_encoding_matrix.png')
