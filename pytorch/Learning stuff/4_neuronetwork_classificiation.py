#this is for classification and an introduction to non-linear models that use non-linear activation functions like ReLU and discuss things like softmax() and sigmoid() for others

#link the tutorial: https://www.youtube.com/watch?v=V_xro1bcAuA
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
#for machine learning datasets
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
#this is one of the main problems with machine learning
#timestamp: 8:34:33

"""
what this covers (broadly):
1. architecture of a neuronetwork classification model
2. input and ourputshapes of a classification model (features and labels)
3. custom data to view, fit and predict on
4. steps in modeling
 4.5. creating the model, setting a loss func and an optimizer, creating a training and testing loop, evailuating the model
5. saving and loading the model
6. using non-linearity
7. different classification evaluation methods
"""

#there's a bunch of theory from 8:34:33 - 8:57:30, 8:53:10 has a few important points i should be aware of


#input and outputs:
input_tensor = torch.randn(size=(224, 224, 3)) #a 224 x 224 picture that uses RGB as its color complex (the 3 at the end is the R, G, B)

n_samples = 1000

#x is matrix, y is labels
X, y = make_circles(n_samples,
                    noise=0.3,
                    random_state=143)

print(len(X), len(y))

#from this, you can see that x has 2 features for each label of y
#also since y only has 0s and 1s, it uses binary classification, if its 0, 1, 2, etc, then it would be multi-class classification
print(f"first 5 samples of X: {X[:5]}\nfirst 5 samples of y: {y[:5]}") 

#making a dataframe: 
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

circles.head(10)


plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y)

#plt.show()

#the data we're working with is refered to a toy dataset, a dataset that's small enough to experiment but sizeable enough to practice the fundimentals


## changing the data into tensors to work with
print(X.shape, y.shape)

X_sample, y_sample = X[0], y[0]

print(f"values for one sample of x: {X_sample} | value for one sample of y: {y_sample}")
print(f"shape for one sample of x: {X_sample.shape} | shape for one sample of y: {y_sample.shape}")

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#reminder that x is usually the input tensor/feature that we have it evaluate, and y is the lebel/output sorta
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, #20% will be test, 80% will be train
                                                    random_state=143)

#timestamp: 9:20:50

#to confirm the length of the tensors we just split:
print(f"length of: x_train: {len(X_train)}, x_test: {len(X_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}")


#making the model using subclassing, which can extend to build a lot more complicated neuro networks, and the forward computation is something that you can control
#unlike the nn.sequential, hence its usually good to use subclassing for making models for more complex models, unlike the simple model below
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()

        #create 2 nn.Linear layers that are capable of handling the shapes of the data,
        self.first_layer = nn.Linear(in_features=2, out_features=8) #in this case, i have 2 input feature neurons that pass to 6 hidden neurons
        self.second_layer = nn.Linear(in_features=8, out_features=1) 
        
        #the 6 hidden neurons pass to 2 output/label neurons, this allows the model to learn 6 patterns from 2 numbers, 
        #potentially leading to better outputs, but it doesn't always work, the optimal number will vary
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_layer(self.first_layer(x))
    

#in this case, we could use nn.sequential() for making this model since it goes in sequence, while making the model on your own it doesn't have to be in sequence
#like in LTSM's or transformers' self attention, which don't usually go in sequence. 

device = "cuda" if torch.cuda.is_available() else "cpu"

circ_model = CircleModel().to(device=device)
print(circ_model.state_dict())

print(X_test.shape)

with torch.inference_mode():
    untrianed_prediction = circ_model(X_test)
    print(f"\n\n\nlength of prediction: {len(untrianed_prediction)}, shape: {untrianed_prediction.shape}")
    print(f"Length of test samples: {len(y_test)}, shape: {y_test.shape}")
    print(f"\nfirst 10 predictions: \n{untrianed_prediction[:10]}\nfirst 10 test labels:\n{y_test[:10]}\n\n\n")

#L1Loss is for predicting a number, in this case we're doing classification, so we want bineary cross entropy, or categorical cross entropy/cross entropy


#NOTE: timestamp is currently at 10:07:22


loss_func = nn.BCEWithLogitsLoss() #has built in sigmoid activation func

#logits in deep learning: https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean

optimizer = torch.optim.SGD(params=circ_model.parameters(),
                            lr=0.01,)

#calculates accuracy out of 100 examples, what percentage does our model get right
def accuracy_func(y_true, y_pred):
    #correct checks how many items in y_true are equal to y_pred and takes the sum of that, then takes the item of that to get the single value version of it
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


#training

"""
going from raw logits -> prediction probabilities -> prediction lables
we can convert the logits into prediction probabilities by passing them into some kind of activation function (e.g. sigmoid for binary cross entropy and softmax for multiclass
classification)

then we can convert the prediction probabilities into prediction labels by either rounding them or taking the argmax in the case of the softmax activation function
"""
with torch.inference_mode():
    #raw output of our model are going to be refered to logits:
    y_logits = circ_model(X_test.to(device=device))[:5]
    print(y_logits, "\nthese are the raw outputs of the model without the activation function, which determines the magnitude of effect of an neuron")

    #we need to pass the logits through an activation function to turn it into prediction probabilities
    y_pred_probs = torch.sigmoid(y_logits)

    #then we round those probabilities and turn it into prediction labels, anything equal to or above 0.5 is est to 1, anything below that is set to 0
    y_preds = torch.round(y_pred_probs)

    #piecing together everything above: (prediction labels <- prediction probabilities via sigmoid <- logits)
    y_pred_labels = torch.round(torch.sigmoid(circ_model(X_test.to(device=device))[:5])) #the [:5] is to make sure that I don't overflow the terminal and have to scroll up a bunch

    #check for equality: torch.eq checks for equality in the tensors, also gets rid of extra dimension
    print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

    print(y_preds.squeeze())


""" for logits
in summary, they are the raw outputs of our models that can be turned into probabilities for our labels of the data via the sigmoid function (becaues this is a binary cross entropy
problem, it would be softmax() if it was a multi-class classification problem), and finally by rounding that we can get our predicted labels
"""

##building the actual training and testing loop

torch.manual_seed(42)

epoches = 300

for epoch in range(epoches):

    circ_model.train()

    #forward pass
    y_logit = circ_model(X_train.to(device=device)).squeeze()
    training_pred_label = torch.round(torch.sigmoid(y_logit))

    """
    if this was BCELoss: 
    loss = loss_func(torch.sigmoid(y_logits), y_train) BCE expects prediction probabilities as inputs
    """

    # ****this is a different loss function, this loss expects raw logits as inputs****
    loss = loss_func(y_logit, y_train)

    acc = accuracy_func(y_true=y_train,
                        y_pred=training_pred_label)

    optimizer.zero_grad()

    loss.backward() #back propagation

    optimizer.step() #gradient descent

    ##testing 
    circ_model.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = circ_model(X_test.to(device=device)).squeeze()
        testing_pred_label = torch.round(torch.sigmoid(test_logits))

        #test loss/acc
        test_loss = loss_func(test_logits, y_test)
        test_acc = accuracy_func(y_true=y_test,
                                 y_pred=testing_pred_label)

    if epoch % 10 == 0:
        #the :5f and :.2f represent the amount of numbers past the decimal/floating point
        print(f"Epoch: {epoch} | Training Loss: {loss:.5f} | Training accuracy: {acc:.2f} | Testing Loss: {test_loss:.5f} | Testing accuracy: {test_acc:.2f}")



## Making predictions and evaluating the model since from the metrics, the model isn't learning anything


#NOTE: 10:49:44 there's a good resource

#Visualizing:
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

#creating the graph
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(circ_model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(circ_model, X_test, y_test)
plt.show()



## step 5: Improving the model:

"""
what we could do (these options are all from our model's perspective since they all deal with the data):
add more layers, to give the model more chances to learn about patterns in the data
add more hidden units
fit for more epochs
changing the activation function or putting activation functions within your model (yes you can do that)
changing the learning rate
changing the loss function

### becaues these are all things that we can change, these are called hyperparameters


we could also change the data but that comes later
"""

# adding more hidden units and adding more layers and increasing epochs to 1000,
# generally you would only like to change 1 value at a time and track the results since you won't know what actually increased the accuracy
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=16)
        self.layer_2 = nn.Linear(in_features=16, out_features=16)
        self.layer_3 = nn.Linear(in_features=16, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #also, this way of writing operations is the fastest since you're doing them all at once
        return self.layer_3(self.layer_2(self.layer_1(x)))


modelV2 = CircleModelV2()

#loss function:
loss_funcV2 = nn.BCEWithLogitsLoss()

#optimizer:
optimizerV2 = torch.optim.Adam(modelV2.parameters(),
                               lr=0.01)

#training loop:
epochs = 1000

X_train, y_train = X_train.to(device=device), y_train.to(device=device)

for epoch in range(epochs):
    #training mode on
    modelV2.train()

    #forward pass
    y_logit = modelV2(X_train).squeeze()
    y_train_pred = torch.round(torch.sigmoid(y_logit))

    #calculate the loss
    lossV2 = loss_funcV2(y_logit, y_train)
    #the accuracy is just for us to see
    trainV2_acc = accuracy_func(y_true=y_train, y_pred=y_train_pred)

    optimizerV2.zero_grad()

    #back propagation
    lossV2.backward()

    #gradient descent
    optimizerV2.step()

    #testing:
    modelV2.eval()
    with torch.inference_mode():
        test_logit = modelV2(X_test.to(device=device)).squeeze()
        y_test_pred = torch.round(torch.sigmoid(test_logit))

        test_loss = loss_funcV2(test_logit, y_test)
        testV2_acc = accuracy_func(y_true=y_test, y_pred=y_test_pred)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Train loss: {lossV2} | Train accuracy: {trainV2_acc} | Test loss: {test_loss} | Test Accuracy: {testV2_acc}")

#creating the graph
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(modelV2, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(modelV2, X_test, y_test)
plt.show()

# still a coin toss


#changing it to learn a straight line to see if its learning anything at all:

weight = 7
bias = 3
start = 0
end = 1
step = 0.01

#creating data:
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

train_split = int(0.8*len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]


plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)
plt.show()

modelV3 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10, bias=True),
    nn.Linear(in_features=10, out_features=10, bias=True),
    nn.Linear(in_features=10, out_features=1, bias=True)
)

loss_funcV3 = nn.L1Loss()

optimizerV3 = torch.optim.SGD(modelV3.parameters(),
                            lr=0.1)

epochs = 1000
for epoch in range(epochs):
    modelV3.train()
    y_predV3 = modelV3(X_train_regression)
    loss = loss_funcV3(y_predV3, y_train_regression)
    optimizerV3.zero_grad()
    loss.backward()
    optimizerV3.step()

    modelV3.eval()
    with torch.inference_mode():
        test_pred = modelV3(X_test_regression)
        test_loss = loss_funcV3(test_pred, y_test_regression)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss:5f} | Test Loss: {test_loss:5f}")


####NOTE: 6. Adding Non-Linearity into our model: (non-straight lines)

#re-creating non-Linear data
n_samples = 1000
X, y = make_circles(n_samples, 
                    noise = 0.3,
                    random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

#converting data to tensors:
from sklearn.model_selection import train_test_split

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#building a model with non-linearity:
class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=16)
        self.layer_2 = nn.Linear(in_features=16, out_features=16)
        self.layer_3 = nn.Linear(in_features=16, out_features=1)
        #the relu layer switches all negative numbers with 0 for the layer that's passed into it, introduces non-linearity
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    
ModelV4 = NonLinearModel().to(device=device)

loss_funcV4 = nn.BCEWithLogitsLoss() #binary classification problem

optimizerV4 = torch.optim.SGD(params=ModelV4.parameters(),
                            lr=0.05)

epochs = 1000

for epoch in range(epochs):
    #training
    ModelV4.train()

    #forward pass
    y_logit = ModelV4(X_train).squeeze()

    #calculate the loss
    loss = loss_funcV4(y_logit, y_train)
    #accuracy for us to see:

    train_acc = accuracy_func(y_true=y_train, y_pred=torch.round(torch.sigmoid(y_logit)))

    optimizerV4.zero_grad()

    #back propagation
    loss.backward()

    #gradient descent
    optimizerV4.step()

    #testing:
    ModelV4.eval()
    with torch.inference_mode():
        test_logit = ModelV4(X_test).squeeze()
        test_loss = loss_funcV4(test_logit, y_test)
        test_acc = accuracy_func(y_true=y_test, y_pred=torch.round(torch.sigmoid(test_logit)))
        
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Train accuracy: {train_acc} | Test loss: {test_loss} | Test Accuracy: {test_acc}")


#Evaluating the model:
with torch.inference_mode():
    y_pred = torch.round(torch.sigmoid(ModelV4(X_test))).squeeze()
    print(y_pred[:10], y_test[:10])

#creating the graph
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(ModelV4, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(ModelV4, X_test, y_test)
plt.show()


##the rest is continued in 4.7