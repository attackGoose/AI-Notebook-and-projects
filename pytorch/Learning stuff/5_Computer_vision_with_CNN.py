##timestamp: 14:13:33

"""
computer vision could be used for classification (both binary and multi-class), image recognition, object detection, spliting different sections of an image or segmenting an image
and many more, this model is going ot be using the convolutional neuro network model but the transformer model have also been shown to have good performance on this as well

About data:
the data is going to be of shape:
Shape = [batch_size, width, height, color_channels] color channel = RGB or etc, height and width of the image, and batch size is the the number of models processed before model 
parameters are updated
ex. Shape = [32, 224, 224, 3] #this is the most common sizes
or: Size = [None, 224, 224, 3] #also note that these parameters can vary

outputshape = [amount_of_possible_results]

About Convolutional Neuro Network (CNNs):

common architecture: (there are many more)
 - input image(s)
 - input layer
 - convolutional layer - extract/learns the most important features from the target imgaes, can be created with torch.nn.ConvXd() 
    (where X is a number/can be multiple values) like torch.nn.Conv2d(), which works with a bias vector and a weight matrix, which opperate over the input
 - hidden activation/non-linear activation - adds non-linearity to the learned features, usually torch.ReLU(), although can be many more
 - pooling layer - reduces dimensionality of the learned image features, uses torch.nn.MaxPool2d for max and torch.nn.AvgPool2d() for an average
 - output layer/linear layer - takes learned features and outputs them in the shapes of the target labels, uses torch.nn.Linear(out_features={number_of_classes}) 
    ex, it would be 3 for fitting into the classes of pizza, steak, or sushi since there are a total of 3 classes/categories
 - output activation - converts output logits into prediction values, usually torch.sigmoid() for binary classification or torch.softmax() for multi-class classification
"""

## other useful stuff that pytorch uses
"""
torchvision - main torch library for vision
torchaudio - audio
torchtext - text problems/stuff
TorchRec - recommendation systems
TorchData - data pipelines 
TorchServe - serving pytorch models

# for torchvision:
torchvision.datasets - getting datasets and data loading functions for computer vision
torchvision models - get pretrained computer vision models (already trained models)
torchvision.transforms - functions for manipulating for images/vision data to be suitable for use with an ML model
torch.utils.data.Datasets - base dataset class for pytorch, you can create your own custom datasets
torch.utils.data.DataLoader - creates a python iterable over a dataset
"""

#pytorch
import torch
from torch import nn

#torchvision stuff
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor #this converts a PIL image or a numpy.ndarray to a tensor

#for visualizing 
import matplotlib.pyplot as plt

print(f"Torch Version: {torch.__version__}\nTorchvision version: {torchvision.__version__}")


### Creating/getting a dataset:

#using Built-in datasets

#timestamp: 14:40:30