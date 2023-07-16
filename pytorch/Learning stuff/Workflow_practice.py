import torch
from torch import nn

#check if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"

#creating data using cubic regression formula
weight = 4.0
bias = 7.0

#range values
start = 0
end = 1
step = 0.02

#x is usually a feature matrix and same iwth y
X = torch.arange(start, end, step)
y = weight * X + bias

train_split = int(0.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

##Building the model:

class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__() #this time, we're not initializing the parameters outselves, this time we're initializing the layer using the Linear to make things easier for outselves
        #this layer is also called the linear transform, probing layer, fully connected layer, dense layer, or etc
        #usually these layers will be created using precreated layers in pytorch rather than creating the layers yourself using nn.Parameter()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1,)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer

MANUAL_SEED = torch.manual_seed(42)

model = LinearRegressionModel()

print(model, list(model.state_dict()))
