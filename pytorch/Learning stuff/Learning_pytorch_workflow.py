### Introduction to Pytorch Workflow timestamp to where it starts (link: https://www.youtube.com/watch?v=V_xro1bcAuA): 4:22:01

import torch
from torch import nn #nn contains all of pytorch's building blocks for neuro networks, pytorch documentation has a lot of building blocks for all sorts of layers

#you can combine layers in all the ways imaginable to make a beuro network model to do what you want it to do

import matplotlib.pyplot as plt
from pathlib import Path

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
        
    # Forward method to define the computation in a model:x is a parameter/input value, as you can see
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

 also check out the pytorch cheatsheet by googling pytorch cheatsheet
 Model building essentials:
  - torch.nn: contains all of the building materials for computational graphs (a neuro networks can be considered a computational graph)
  - torch.nn.Parameter(): what parameters our model should try and learn, often a pytorch layer from pytorch.nn will set these for us
  - torch.nn.Module: the base class for all neuro network modules, if you subclass it, you should override forward()
  - torch.optim: this references the optimizer in pytorch, they will help with gradient descent and contains various optimization algorithms
  - torch.data.Dataset: represents a map between the key (label) and sample (feature) pairs of your data, such as images and their associated labels
  - torch.data.DataLoader: creates a python iterable over a torch Dataset, allowing you to iterate over your data
  - torchvision.transforms: for pictures and vision into data into models
  - torchmetrics: 

  
  - def forward(): all nn.Module subclasses require you to override this, as previously stated, this method defines what happens in the forward computation

"""


## Checking the contents of our model:

#to check the parameters of our model, we can use .parameters():

#sets tha seed so the values won't vary and results will stay consistant, without this, the tensor values in the LinearRegressionModel would be random every time (which is what we want, but for educational purposes that's not needed here)
torch.manual_seed(42)

#initialize model
model = LinearRegressionModel()

print(list(model.parameters()))

#list named parameters: (a parameter is something that the model sets itself/is present in the "()" incase i'm dum and forgot)
print(model.state_dict()) #the name comes from the self.weight and self.bias i think


## making predictions using torch.inference_mode()

#context manager, its good to make this a habit since it turns off gradient tracking since when we're doing predictions, which makes it a lot faster in larger data sets
#there's also torch.no_grad() but inference_mode is the prefered
with torch.inference_mode():
    y_preds = model(X_test)

print(f"Predictions: {y_preds}\nTest Data: {y_test}")

plot_prediction(predictions=y_preds)


"""## Training the model (moving from unknown/random parameters closer to the actual accurate parameters, aka moving from a poor representation of the data to a better one)

The loss function tells us how wrong our model's predictions are
 - note that a loss function can also be refered to as a cost function or a criterion in different areas

Things we need to train: 
 - Loss function - a function that measures how wrong our model's predictions are compared to the idea outputs, the lower the better
 - Optimizer - takes into account the loss of a model and adjusts the model's parameters (e.g. weight and bias) to improve the loss function

For pytorch specifically
, we need:
 - a training loop
 - a testing look

you can check out all the loss functions in the pytorch documentation: https://pytorch.org/docs/stable/nn.html#loss-functions
"""

#choosing and implimenting a loss function and a optimizer:

#using L1Loss/Mean Absolute Error (taking the absolute difference between all the expected value/ideal value and the actual value and returns its average)
#measures how wrong our data is
loss_fn = nn.L1Loss()


#setup an optimizer (using a Stoch(SGD) algorithm)
#an optimizer adjusts the parameters according to the loss  function to reduce the loss
optimizer = torch.optim.SGD(model.parameters(), #the parameters that its going to take a look at/optimize
                            lr= 0.01) #learning rate: one of the most important hyperparameter (we set) you can set (regular parameters are set by the code)


#general idea of how optimizers work: it first increases the value in one direction, if the loss increases, then it increases in the other direction until the best value is achieved

"""
The learning rate (lr) is how mcuh it adjusts the parameters given to reduce the loss function/optimize the values, so the smaller the lr, the smaller the change in the parameter
the larget the learning rate, the larger the change int he parameter, if the lr is too bigthen it might skip over the optimal value, but if its too smal, then it'll take too
long to optimize

Q&A:
which loss function and optimizer should I use?
this depends on the context, with experience you'll get an idea of what works and what doesn't with your particular data set

ex. a regression problem would require something like a loss function of nn.L1Loss() and an optimizer like torch.optim.SGD()

but for classification problems like classifying whether or not a photo is of a dog or a cat, you'll likely want to use a loss function of nn.BCELoss() (binary cross entropy loss)
"""

## Building a training Loop (and a testing loop):

"""
steps: 
0. looping through the data
1. forward pass (moving our data through the forward() method), also called forward propagation, moving in the opposite direction of a back propagation
2. calculate the loss: compare the forward pass predictions to the ground truth labels
3. optimizer zero grad
4. Back propagation (loss backwards?): data moves backwards through the network to calculate the gradients of each of the parameters of the model with respect to loss
5. optimizer step: use the optimizer to adjust the model's parameters to try to improve the loss
"""

#an epoch is one loop through the data, a hyper parameter because we set it ourselves
epochs = 200

#track different values and tracks model progress, used to plot model progress later on, useful for comparing with future experiments
epoch_count = []
loss_values = []
test_loss_values = []

print(model.state_dict())

for epoch in range(epochs):

    #set the model to training mode, training mode sets all paramaters that requires gradients to require gradients, requires_grad=True
    model.train()

    #forward pass:
    y_pred = model(X_train)

    #loss function:
    loss = loss_fn(y_pred, y_train) #predictions first then target
    print(f"Loss: {loss}")

    #optimizer zero_grad()
    optimizer.zero_grad()

    #4. back propagation on loss with respect to the parameters of the model
    loss.backward()

    #Optimizer, we want to step towards a gradient with a slope of 0 (slope of the loss function) or as low as possible, this is gradient descent and pytorch is doing this for you
    #in torch autograd
    optimizer.step() #by default how the optimizer changes will accumulate through the loop, so we have to zero them above (shown in step 3) for the next iteration of the loop


    ### testing
    model.eval() #evaluation mode, turns off training, starts testing
    #this turns off different stuff in the model that's not used for testing (essentially its like dropout/batch norm layers, read docs for more info)

    with torch.inference_mode(): #turns off gradient tracking for inference and a couple of other stuff to make testing faster. torch.no_grad() does the same but slower
        #1. foward pass:
        test_pred = model(X_test)

        #2. loss calculation:
        test_loss = loss_fn(test_pred, y_test) #y_test is the test labels, calculates the testing loss value

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
            print(model.state_dict())

#matplotlib works with numpy, not working with the gpu because i don't have one so i can skip the "".cpu()"".numpy() part and just go right to .numpy
plt.plot(torch.tensor(epoch_count).numpy(), torch.tensor(loss_values).numpy(), label="Train loss") 
plt.plot(torch.tensor(epoch_count).numpy(), torch.tensor(test_loss_values).numpy(), label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

#there is also learning rate scheduling, which is basically starting with big steps in the learning rate, then slowly lowering it,like reacing for the coin at the backofthe couch
#the lowest point is the convergence, its the point where the loss function is at its minimum

#the steps in the loop can be turned into a function, do later, first build intuition for it

with torch.inference_mode():
    y_preds_new = model(X_test)

plot_prediction(predictions=y_preds_new)


## Saving models: 

"""
there are 3 main methods you should know about when it comes to saving and loading: (https://pytorch.org/tutorials/beginner/saving_loading_models.html)

1. torch.save(): saves a serialized object to disk, uses the python pickle library's utility for serialization. Models, tensors, and dictionaries are all kinds of objects that
can be saved using this function, its recommended to save the state_dict, but you can also save the entire model

2. torch.load(): uses the pickle module to unpickle facilities to deserialize object files to memory, in the process also facilitates the device that the data is being loaded
into

3. torch.nn.Module.load_state_dict(): Loads a model's parameter dictionary using a deserialized state_dict, for more info, check out the website linked above
"""

#create model directory:
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#create a model save path
MODEL_NAME = "01_pytorch_workflow_tutorial.pth" #the .pth is for saving a pytorch model

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#saving only the model's state_dict(): (the model's weights and biases and etc)
print(f"saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), 
           f=MODEL_SAVE_PATH)


## Loading a model into a new instance of the model:

new_model = LinearRegressionModel()

#loading the state dict/loading the pre-trained values to replace the random values
new_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) #loads all the state dictionaries like the weights and biases and etc

#making predictions using the loaded model:
new_model.eval()
with torch.inference_mode():
    new_model_pred = new_model(X_test)

    y_preds = model(X_test) #incase the y_preds value was changed

    ##compare the predictions/forward() calculations of both models, they should be the same since the values would be the same
    print(new_model_pred == y_preds)


##continued in Workflow_ractice.py

##more info on loading and saving models on the pytorch docs: https://pytorch.org/tutorials/beginner/saving_loading_models.html