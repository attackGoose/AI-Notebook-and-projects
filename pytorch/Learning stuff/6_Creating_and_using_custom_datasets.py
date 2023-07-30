## timestamp: 19:44:05

#bookmark timestamp: 19:50:21 picture is helpful, the domain libearies are used for specialized types of data like text, image,s audio, or etc and have functions specialized 
# for these types of data

#another bookmark timestamp: 19:53:38 (the last one is data that's not in the training nor testing dataset, so predicting on real-world data)

#domain libraries: (since we're using this with a vision problem, this is going to use torchvision)

#pytorch and its domain libraries
import torch
from torch import nn

import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"

#current timestamp: 20:00:13