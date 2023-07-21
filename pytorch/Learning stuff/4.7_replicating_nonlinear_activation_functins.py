import torch
from torch import nn
import matplotlib.pyplot as plt



#replicating Non-Linear functions:

"""
remember that rather than us telling the model what to learn, we give it the tools to discover patterns in the data and it tries to figure out the best patterns on its own

these tools being linear and/or non-linear functions, or a combination of both
"""

A = torch.arange(-10, 10, 1, dtype=torch.float32)

#since we gave it iunt values, the default will be torch.int64, so we can manually change it to the other default, which is torch.float32


#replicating a relu function:
plt.plot(torch.relu(A))
plt.show()

#making our own relu function:
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x) #it returns all the numbers within the range and all numbers below the range is set to 0, google for specifics

print(relu(A))
plt.plot(relu(A))
plt.show()

#making a sigmoid function:
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

plt.plot(torch.sigmoid(A))
plt.show()

plt.plot(sigmoid(A))
plt.show()