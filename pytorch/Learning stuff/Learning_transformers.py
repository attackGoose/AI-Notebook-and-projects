import numpy as np
import math

##For visualizing self-attention in transformers:

#each word will contain these 3 vectors, 
length, dim_key, dim_value = 4, 8, 8
query_vector = np.random.randn(L, dim_key)
key_vector = np.random.randn(L, dim_key)
value_vector = np.random.randn(L, dim_value)

#initial self-attention matrix
scaled = np.matmul(query_vector, key_vector.T) / math.sqrt(dim_key)

##masking: for the decoding so we don't look at a future word for generating the current context of the current word since you don't know what words will be generated next
mask = np.tril(np.ones( (length, length)))

