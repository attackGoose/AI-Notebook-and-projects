#this file is for learning pytorch, and choosing if i like tensorflow or pytorch more
#https://pytorch.org/tutorials/
#https://youtu.be/V_xro1bcAuA


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

### creating a tensor
scalar = torch.Tensor(7) #has no dimensions, one number

print(scalar.shape) #returns the number we gave it

vector = torch.Tensor([7, 7]) #has 1 dim since its a 1 dim array
print(vector.shape)

MAXTRIX = torch.Tensor([[7, 8],
                        [9, 10]])

TENSOR = torch.Tensor([[[1,2,3],
                        [3,6,9],
                        [2,4,5]]])
print(TENSOR.shape)


### random tensors: (creates more accurate training data)

rand_tensor = torch.rand(3, 4) #creates a tensor of size 3, 4 (matrix) 