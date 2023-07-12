### Introduction to Pytorch Workflow timestamp to where it starts (link: https://www.youtube.com/watch?v=V_xro1bcAuA): 4:22:01

import torch
from torch import nn #nn contains all of pytorch's building blocks for neuro networks, pytorch documentation has a lot of building blocks for all sorts of layers

#you can combine layers in all the ways imaginable to make a beuro network model to do what you want it to do

import matplotlib.pyplot as plt


## preparing and loading data (data can be almost anything in machine learning, like images, csv, videos, audio, text, or even dna)

# machine learning is a game of 2 major parts: (that can be further subdivided into many other parts)
# 1. get data into a numerical representation (tensors)
# 2. build a model to learn patterns in that numerical representation

# making data using a linear regression formula:

#creating known parameters:
weight = 0.7
bias = 0.3

#create:
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) #x is usually used as a tensor 
y = weight * X + bias
print(X)