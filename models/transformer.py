import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import sqrt
from torch.optim import Adam
from typing import Optional, Tuple
import torch.nn.init as init

class Transformer(nn.Module):
    '''
    Arguments (__init__):
        - n (int): Number of Transformer blocks;
        - d_model (int): Default model dimension for blocks (Encoder-Decoder);
        - d_ff (int): Feed-forward dimension;
        - heads (int): Number of heads in MultiHeadAttention;
        - src_vocab (int): Structured vocabulary from the source language for mapping tokens into indices;
        - tgt_vocab (int): Structured vocabulary from the target language for mapping tokens into indices;
        - max_len (int): Maximum sequence length supported for the source vocabulary;
        - dropout (float): Value between [0, 1] representing the dropout percentage;
        - bias (bool): Flag for bias;
        - pad_idx (int): Padding index.

    Return (forward): Tensor with dimensions [batch_size, sequence_len, vocab_size]
    '''
    def __init__(self, src_vocab: int, tgt_vocab: int, n: int=2, d_model: int=512, heads: int=8,
                       max_len: int=128, d_ff: int=2048,
                       dropout: float=0.1, bias: bool=True, pad_idx: int=0):
        super().__init__()

        self.n = n
        self.d_model = d_model
        self.heads = heads
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.dropout = dropout
        self.bias = bias
        self.pad_idx = pad_idx
        self.d_ff = d_ff
        self.act_func = act_func

        assert d_model % heads == 0, f'Dimension model ({d_model}) is not divisible by number of heads ({heads})'

        self._build_architecture()

    def _build_architecture(self):

        # --- building embeddings and positional encoding ---
        
        # Embeddings and Positional Encoding [batch_size, sequence_length]
        self.src_embedding_layer = TransformerEmbedding(self.src_vocab, self.d_model, self.pad_idx)
        self.tgt_embedding_layer = TransformerEmbedding(self.tgt_vocab, self.d_model, self.pad_idx)
        self.positional_layer = PositionalEncoding(self.max_len, self.d_model, self.dropout)
        
        # Encoder-Decoder [batch_size, sequence_length, d_model]
        dqkv = self.d_model // self.heads

        self.encoders = nn.ModuleList([
            Encoder(self.heads, self.d_ff, self.d_model, dqkv, dqkv, self.dropout, self.bias) for _ in range(self.n)])

        self.decoders = nn.ModuleList([
            Decoder(self.heads, self.d_ff, self.d_model, dqkv, dqkv, self.dropout, self.bias) for _ in range(self.n)])

        # Final layer [batch_size, sequence_length, vocab_size]

        self.linear = nn.Linear(self.d_model, self.tgt_vocab, bias=self.bias)

        init.xavier_uniform_(self.linear.weight.data)
        init.constant_(self.linear.bias.data, 0)
    
    def encode(self, src_input, src_mask):
        
        x_enc = self.src_embedding_layer(src_input)
        x_enc = self.positional_layer(x_enc)

        for encoder in self.encoders:
            x_enc = encoder(x_enc, src_mask)

        return x_enc

    def decode(self, x_enc, tgt_input, src_mask, tgt_mask):
        
        x_dec = self.tgt_embedding_layer(tgt_input)
        x_dec = self.positional_layer(x_dec)
        
        for decoder in self.decoders:
            x_dec = decoder(x_dec, x_enc, src_mask, tgt_mask)

        return x_dec

    def forward(self, src_input, tgt_input):
        
        src_mask = (src_input != 0).float()
        tgt_mask = (tgt_input != 0).float()

        x_enc = self.encode(src_input, src_mask)
        x_dec = self.decode(x_enc, tgt_input, src_mask, tgt_mask)

        X = self.linear(x_dec)

        return X

class SelfAttention(nn.Module):
    '''Implement a single-head attention layer in a Transformer network.

    Arguments (__init__):
        - dm (int): dimensionality of model;
        - mask (bool): flag indicating whether to apply masking during attention.

    Return (forward): weighted sum of values based on attention scores.
    '''
    def __init__(self, dm: int, mask: bool):
        super().__init__()

        self.dm = dm
        self.mask = mask
    
    def _padding_mask(self, vmask, hmask, scores):

        vmask = torch.unsqueeze(torch.unsqueeze(vmask, 1), 1)
        hmask = torch.unsqueeze(torch.unsqueeze(hmask, 1), 1)

        B, H, T, T = scores.size()

        padding_vmask = vmask.repeat(1, 1, H, T).view(B, -1, T, T)
        padding_hmask = hmask.repeat(1, 1, H, T).view(B, -1, T, T).transpose(3, 2)

        new_scores = scores * (padding_vmask * padding_hmask)

        return new_scores.masked_fill(new_scores == 0, 1e-10)

    def forward(self, q, k, v, f, vmask, hmask):
        
        scores = torch.div(torch.matmul(q, k), f)
        
        # --- padding mask ---
        
        scores = self._padding_mask(vmask, hmask, scores)
        
        # --- upper triangular mask ---

        if self.mask:
            scores = scores + torch.triu(torch.full(scores.shape, float('-inf')), diagonal=1).to('cuda')

        return torch.matmul(F.softmax(scores.to('cuda'), dim=-1), v)

class MultiHeadAttention(nn.Module):
    '''Implement a multi-head attention layer in a Transformer network.

    Arguments (__init__):
        - num_heads (int): number of attention heads;
        - dm (int): model dimension;
        - d_k (int): dimensionality of keys;
        - d_v (int): dimensionality of values;
        - bias (bool): flag indicating whether to include bias in linear layers;
        - mask (bool): flag indicating whether to apply masking during attention.

    Return (forward): output after applying multi-head attention mechanism.
    '''
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

    def forward(self, xq, xk, xv, vmask, hmask):
        
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

        out = self.attention_mechanism(query, key, value, scale_factor, vmask, hmask)
        
        concat = out.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, -1)

        return self.linear(concat)

class Encoder(nn.Module):
    '''Implement an encoder layer in a Transformer network.

    Arguments (__init__):
        - heads (int): number of attention heads in multi-head attention;
        - d_ff (int): dimensionality of feedforward layer;
        - d_model (int): model dimension;
        - d_k (int): dimensionality of keys in multi-head attention;
        - d_v (int): dimensionality of values in multi-head attention;
        - dropout (float): dropout rate for normalization layers;
        - bias (bool): flag indicating whether to include bias in linear layers.

    Return (forward): output after applying encoder layer.
    '''
    def __init__(self, heads: int, d_ff: int, d_model: int, d_k: int, d_v: int, dropout: float, bias=bool):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(heads, d_model, d_k, d_v, bias, mask=False)
        self.ffnn = PositionWiseFFN(d_model, d_ff, bias)
        self.AddNorm = nn.ModuleList([AddNorm(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        
        x = self.AddNorm[0](x, self.self_attention(x, x, x, src_mask, src_mask)) 
        
        x = self.AddNorm[1](x, self.ffnn(x))

        return x

class Decoder(nn.Module):
    '''Implement a decoder layer in a Transformer network.

    Arguments (__init__):
        - heads (int): number of attention heads in multi-head attention;
        - d_ff (int): dimensionality of feedforward layer;
        - d_model (int): model dimension;
        - d_k (int): dimensionality of keys in multi-head attention;
        - d_v (int): dimensionality of values in multi-head attention;
        - dropout (float): dropout rate for normalization layers;
        - bias (bool): flag indicating whether to include bias in linear layers.

    Return (forward): output after applying decoder layer.
    '''
    def __init__(self, heads: int, d_ff: int, d_model: int, d_k: int, d_v: int, dropout: float, bias=bool):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(heads, d_model, d_k, d_v, bias, mask=True)
        self.cross_attention = MultiHeadAttention(heads, d_model, d_k, d_v, bias, mask=False)
        self.ffnn = PositionWiseFFN(d_model, d_ff, bias, dropout)
        self.AddNorm = nn.ModuleList([AddNorm(d_model, dropout) for _ in range(3)])

    def forward(self, x, x_enc, src_mask, tgt_mask):
        
        x = self.AddNorm[0](x, self.self_attention(x, x, x, tgt_mask, tgt_mask))

        x = self.AddNorm[1](x, self.cross_attention(x, x_enc, x_enc, src_mask, tgt_mask))
        x = self.AddNorm[2](x, self.ffnn(x))

        return x

class AddNorm(nn.Module):
    '''Implement the additive normalization layer in a Transformer network.

    Arguments (__init__):
        - d_model (int): model dimension;
        - dropout (float): dropout rate for normalization.

    Return (forward): output after normalization and residual addition.
    '''
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual, x):

        return self.layer_norm(residual + self.dropout(x))
    
class TransformerEmbedding(nn.Module):
    '''Transform token indices into word embeddings.

    Arguments (__init__):
        - vocab_size (int): total number of tokens in vocab;
        - d_model (int): model dimension;
        - padding_idx (int): indice of padding in vocab dict.

    Return (forward): word embeddings multiplied by weight scale factor.
    '''
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int) -> None:
        super().__init__()

        self.d_model = d_model

        # bulding embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )

    def forward(self, x):

        # embeddings * weight scaling
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionWiseFFN(nn.Module):
    '''Position-wise Feed-Forward Neural Network after Multi-head attention.

    Arguments (__init__):
        - d_model (int): model dimension;
        - d_ff (int): feed-forward dimension;
        - bias (bool): flag for bias;
        - dropout (float): value between [0, 1] represent percentage of dropout.

    Return (forward): a tensor with dimension [batch, sequence_length, d_model]
    '''
    def __init__(self, d_model: int, d_ff: int, bias: bool, dropout: float=0.1):
        super().__init__()

        self.first_linear = nn.Linear(d_model, d_ff, bias=bias)
        self.second_linear = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):

        residual = x

        # [batch_size, sequence_length, d_model] to [batch_size, sequence_length, d_ff]
        x = self.activation(self.first_linear(x))
        # [batch_size, sequence_length, d_ff] to [batch_size, sequence_length, d_model]
        x = self.second_linear(x)

        return x

class PositionalEncoding(nn.Module):
    '''Fixed positional encoding for addition in word embeddings.

    Arguments (__init__):
        - max_len (int): maximum sequence length of context size;
        - d_model (int): model dimension;
        - dropout (float): float between [0, 1] represent percentage of dropout;

    Return (forward): sum between word embeddings and positional encoding.
    '''

    def __init__(self, max_len: int, d_model: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # defining positional encoding matrix [sequence_length, d_model]
        pe = torch.zeros((max_len, d_model))

        # defining positions (pos) and frequencies (freq) [sequence_length, 1]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        # filling pe matrix with sines for even positions and cosines for odd positions
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)

        # register buffer (automatically transition CPU/GPU and grad=False)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        #x = x + (self.pe[:, :x.size()[1], :])
        x = x + (self.pe[:x.size()[1], :])

        return self.dropout(x)
