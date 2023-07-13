### Introduction to Pytorch Workflow timestamp to where it starts (link: https://www.youtube.com/watch?v=V_xro1bcAuA): 4:22:01

import torch
from torch import nn #nn contains all of pytorch's building blocks for neuro networks, pytorch documentation has a lot of building blocks for all sorts of layers

#you can combine layers in all the ways imaginable to make a beuro network model to do what you want it to do

import matplotlib.pyplot as plt

"""
preparing and loading data (data can be almost anything in machine learning, like images, csv, videos, audio, text, or even dna)

machine learning is a game of 2 major parts: (that can be further subdivided into many other parts)
1. get data into a numerical representation (tensors)
2. build a model to learn patterns in that numerical representation
"""
# making data using a linear regression formula:

#creating known parameters: (in an actual dataset scraped from the internet, these won't be given, these are what the model is going to figure out)
weight = 0.7
bias = 0.3

#create:
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) #x is usually used as a tensor, and we need the extra dimension for something later
y = weight * X + bias #the machine won't know this and will have to figure this out for itself, the y variable is the target

print(X[:10], y[:10], len(X), len(y))


## spliting data into training and test sets (one of the most important concepts in machine learning in general)

"""
visualizing the three datasets by comparing it to a school course:

training set: you can compare this to the course materials at a university that you would learn throughout the year, the model too would learn patterns from here
validation set: you can compare this to a practice exam, which would tune the model patterns/adjust the model's patterns (not always needed)
Test set: you can compare this to a final exam: which would see if the model is ready to be implimented/tests the model's performance on data it hasn't seen before

Goal: generalization (the ability for a machine learning model to perform well on data it hasn't seen before)

amount of data used for training set: ~60-80% (always needed)
amount of data used for validation set: 10-20% (not always needed)
amount of data used for test set: 10-20% (always needed)
"""

#create a train/test split/set (set and split mean the same thing in this case)
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split] #gets all the data that's previous to that index
X_test, y_test = X[train_split:], y[train_split:] #gets all the data that is past that index

print(len(X_train), len(y_train), len(X_test), len(y_test)) #prints the amount of training features, training lables, testing features, testing lables

#NOTE: you can also use the sklearn/scikit module to split the training data in a more random way


## building a function to visualize the data

def plot_prediction(train_data = X_train, 
                    train_lables = y_train, 
                    test_data = X_test, 
                    test_lables = y_test, 
                    predictions = None):
    """
    Plots training data, test data, and compares predictions
    """
    plt.figure(figsize=(10, 7))

    #plot training data in blue
    plt.scatter(train_data, train_lables, c="blue", s=4, label="Training Data")
    
    #plot testing data in green
    plt.scatter(test_data, test_lables, c="green", s=4, label="Testing Data")

    if predictions != None:
        #plot the predictions if they exist
        plt.scatter(test_data, predictions, c="red", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()

plot_prediction()


## building a model:

class LinearRegressionModel(nn.Module): # <- almost everything in pytorch inherits from nn, for more info: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    def __init__(self):
        super().__init__() #start with random parameters, then update them to fit the training data, by running it through the formula it'll adjust the data to fit the linear regression formula
        self.weight = nn.Parameter(torch.randn(1,
                                              requires_grad=True, #gradient descent = true
                                              dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float)) #we might also initialize a layer or a list of layers for our model to use
        
    # Forward method to define the computation in a model:
    def forward(self, x: torch.Tensor) -> torch.Tensor: #x is the input data (of torch.Tensor datatype), and this function is going to return a tensor datatype 
        return self.weight * x + self.bias #this is the linear regression formula, forward is what defines the opperation that a module does
    
    ### any subclass of nn.module needs to override the forward() method from model since it defines the computation of the model
    


"""
what the model does:
 - Starts with random values (weights and biases)
 - looks at training data and adjusts the random values to better represent/get closer to the ideal values (weight and bias values of our original formula)

How does it do it:
 1. Gradient Descent
 2. Back Propagation 
"""

#Current time stamp: 5:13:41