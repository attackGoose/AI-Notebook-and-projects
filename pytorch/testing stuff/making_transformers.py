import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads #returns the integer value of the quotient (floor division: dumps the decimal values)

        assert (self.head_dim * heads == embed_size) # raises an error if embed size is not able to be divided by heads

        self.values = nn.Linear(in_features=self.head_dim,
                                out_features=self.head_dim,
                                bias=False)
        self.keys = nn.Linear(in_features=self.head_dim,
                                out_features=self.head_dim,
                                bias=False)
        self.query = nn.Linear(in_features=self.head_dim,
                                out_features=self.head_dim,
                                bias=False)
        
        self.fully_connected_out = nn.Linear(in_features=embed_size, #this should be the number of heads times the head dimension, which should be equal to the embed_size
                                             out_features=embed_size)
        
    def forward(self, 
                values: torch.Tensor, 
                keys: torch.Tensor, 
                query: torch.Tensor, 
                mask: bool) -> torch.Tensor:
        
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] #this value will correspond to the input/source sentence length and the target sentence length

        #split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        query = query.reshape(N, key_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)

        