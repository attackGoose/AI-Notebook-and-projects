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
        self.first_layer = nn.Linear(in_features=2, out_features=10) #in this case, i have 2 input feature neurons that pass to 6 hidden neurons
        self.second_layer = nn.Linear(in_features=10, out_features=1) 
        
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

#the shape is incompatable for the hidden layers, which is 5x1, so the [1] has to be equal to [0] which is 5
with torch.inference_mode():
    untrianed_prediction = circ_model(X_test)
    print(f"\n\n\nlength of prediction: {len(untrianed_prediction)}, shape: {untrianed_prediction.shape}")
    print(f"Length of test samples: {len(y_test)}, shape: {y_test.shape}")
    print(f"\nfirst 10 predictions: \n{untrianed_prediction[:10]}\nfirst 10 test labels:\n{y_test[:10]}\n\n\n")

#L1Loss is for predicting a number, in this case we're doing classification, so we want bineary cross entropy, or categorical cross entropy/cross entropy

loss_func = nn.BCEWithLogitsLoss() #has built in sigmoid activation func

#logits in deep learning: https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean


optimizer = torch.optim.Adam(params=circ_model.parameters(),
                            lr=0.02,)

epoches = 100

for epoch in range(epoches):
    circ_model.train()

    y_pred = circ_model(X_train)

    print(y_pred[:10], y_train[:10])

    loss = loss_func(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    circ_model.eval()
    with torch.inference_mode():
        test_pred = circ_model(X_test)

        test_loss = loss_func(test_pred, y_test)

        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}")
