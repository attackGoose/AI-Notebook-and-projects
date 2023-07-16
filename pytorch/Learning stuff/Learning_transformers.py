#resources to use: https://youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4
#btw the stuff isn't in order, they're mostly coding out the theory


import numpy as np
import math

##For visualizing self-attention in transformers:

#each word will contain these 3 vectors, 
length, dim_key, dim_value = 4, 8, 8
query_vector = np.random.randn(length, dim_key)
key_vector = np.random.randn(length, dim_key)
value_vector = np.random.randn(length, dim_value)

#initial self-attention matrix
scaled = np.matmul(query_vector, key_vector.T) / math.sqrt(dim_key)

##masking: for the decoding so we don't look at a future word for generating the current context of the current word since you don't know what words will be generated next

mask = np.tril(np.ones( (length, length))) #sets every value that's ahead of the current word at 0, next step has 1 word, next has 2, so we progress one word at a time
print(mask) 
 #creates a step matrix (idk the name for it) 


mask[mask == 0] = -np.infty
mask[mask == 1] = 0

mask_scaled = scaled + mask


# softmax: (stabalizes some values and adjusts how much attention should be paid to each vector/word)
def softmax(x): #there's a pytorch function for this
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

#this way, the attention vector won't incorporate any word that comes after it
attention = softmax(mask_scaled)

new_vector = np.matmul(attention, value_vector) #these new matrcies should better encapsulate the meaning of a word


#putting it all into a function:
def scaled_dot_product_attention(q, k, v, mask=None): #q = query vector, k = key vector, v = value vector
    dim_key = q.shape[-1]
    scaled = np.matmul(q, k.T) / math.sqrt(k)
    if mask is not None:
        scaled = scaled + mask
    attention_head = softmax(scaled)
    out = np.matmul(attention_head, v)
    return out, attention_head

#we can have multi-headed attentions in each cell

#multi-headed attention:







"""Transformer encoder:"""

#NOTE:
#usually we will be padding the tensor that we create, and will usually be encoded into a one hot dimensional vector

##meaning encoders are the produt of the embedding transformer encoding
##Positional Encodings are predetermined and is a fixed value matrix

#add these together and you'll have a matrix that contains both the meaning and position of the given input matrix/tensor, and this marks the start of the encoding layer
#which can be mapped into the key vector, value vector, and query vector, all of which can be learned by the machine

#query vector: what you're looking to find
#key vector: what can you offer
#value offer: what can you actually offer

##each of these vectors are a max_sequence_length that you set/can be passed into the transformer x 512

