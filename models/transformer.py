import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import math

from typing import Optional, Tuple

class Transformer(nn.Module):
    
    def __init__(self, n: int, d_model: int, d_ff: int, heads: int,
                 source_vocab: any, target_vocab: any, max_len: int):
        super().__init__()

        self.n = n
        self.d_model = d_model
        self.d_ff = d_ff
        self.heads = heads
        self.max_len = max_len
        print(len(source_vocab.get_itos()))
        print(source_vocab.get_stoi()['<pad>'])

        self.src_embedding_layer = TransformerEmbedding(source_vocab, d_model)
        #self.trg_embedding_layer = TransformerEmbedding(target_vocab, d_model)
    
        self.src_positional_layer = PositionalEncoding(max_len, d_model)
        #self.trg_positional_layer = PositionalEncoding(batch_sizemax_len, d_model)

        '''
        self.src_embeddings = nn.Embedding(num_embeddings=len(source_vocab.get_itos()), 
                                            embedding_dim=d_model,
                                            padding_idx=source_vocab.lookup_indices(['<pad>'][0]))
        self.trg_embeddings = nn.Embedding(num_embeddings=len(target_vocab.get_itos()), 
                                             embedding_dim=d_model,
                                            padding_idx=target_vocab.lookup_indices(['<pad>'][0]))
        ''' 
        
        self.attention_test = SelfAttention(self.d_model, self.d_model, self.d_model)

        plot_positional_encoding(self.src_positional_layer)

    def _build_architecture(self):

        pass        

    def forward(self, X):
        
        # apply embeddings and positional encoding
        X = self.src_embedding_layer(X)
        print('Embeddings values:', X)
        print('Embeddings output shape:', X.shape)
        X = self.src_positional_layer(X)
        print('Embeddings + PE values:', X) 
        print('Positional enconding output:', X.shape)

        X = self.attention_test(X)
        print('Attention:', X.shape)

        return X

class EncoderDecoder(nn.Module):
    
    def __init__(self):
        pass

    def forward(self, X):

        return X
#TODO
class SelfAttention(nn.Module):

    def __init__(self, d_model: int, d_qk: int, d_v: int):
        super().__init__()
        self.d_qk = d_qk
        self.d_v = d_v
        self.d_model = d_model
        self.w_q = nn.Parameter(torch.FloatTensor(d_qk, d_qk))
        self.w_k = nn.Parameter(torch.FloatTensor(d_qk, d_qk))
        self.w_v = nn.Parameter(torch.FloatTensor(d_v, d_v))

    def forward(self, X):

        query = torch.bmm(X, self.w_q.unsqueeze(0).expand(X.size(0), -1, -1))
        print('Query size:', query.size())
        key = torch.bmm(X, self.w_k.unsqueeze(0).expand(X.size(0), -1, -1)).transpose(1, 2)
        print('Key size:', key.size())
        d_qk = torch.tensor(self.w_q.size(-1), dtype=torch.float)
        qk = torch.div(torch.bmm(query, key), torch.sqrt(d_qk))
        print('Query x Key Normalized size:', qk.size())
        s_qk = F.softmax(qk, dim=1)
        print('Softmax QK size:', s_qk.size())
        value = torch.bmm(X, self.w_v.unsqueeze(0).expand(X.size(0), -1, -1))
        print('Value size:', value.size())
        output = torch.bmm(s_qk, value)
        print('Output size:', output.size())

        return output
#TODO
class MHA(nn.Module):

    def __init__(self, h: int):

        pass

    def forward(self, X):

        pass
#TODO
class Encoder(nn.Module):

    def __init__(self):
        pass

    def forward(self, X):
        pass
#TODO
class Decoder(nn.Module):

    def __init__(self):
        pass
   
    def forward(self, X):
        pass

class TransformerEmbedding(nn.Module):

    def __init__(self, vocab, d_model):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab.get_itos()), 
            embedding_dim=d_model,
            padding_idx=vocab.get_stoi()['<pad>']
        )
         
    def forward(self, X):

        return self.embedding(X) * math.sqrt(self.embedding.embedding_dim)

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
    
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        wk = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        self.pos_enc = torch.zeros((max_len, d_model))
        
        self.pos_enc[:, 0::2] = torch.sin(pos * wk)
        self.pos_enc[:, 1::2] = torch.cos(pos * wk)

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
