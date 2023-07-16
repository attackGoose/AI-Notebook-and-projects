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

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

##Building the model:
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        #this time, we're not initializing the parameters outselves, this time we're initializing the layer using the Linear to make things easier for outselves
        #this layer is also called the linear transform, probing layer, fully connected layer, dense layer, or etc
        #usually these layers will be created using precreated layers in pytorch rather than creating the layers yourself using nn.Parameter()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

model = LinearRegressionModel()

print(model, list(model.state_dict()))

#setting the data and model to use the target device:
model.to(device=device)

print(f"x_train size: {X_train.shape} | x_test size: {X_test.shape} | y_train size: {y_train.shape} | y_test size: {y_test.shape}")

##Training:
"""
we need a loss function
optimizer
training loop
testing loop
"""

loss_func = nn.L1Loss()

optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.05)

epoches = 200

#training loop:

for epoch in range(epoches):
    model.train()

    y_prediction = model(X_train)

    #creates a loss value based on difference in predicted value and the expected value
    loss = loss_func(y_prediction, y_train)

    #reset gradients
    optimizer.zero_grad()

    #back prop
    loss.backward()


    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)

        test_loss = loss_func(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"epoch = {epoch} | loss = {loss} | test loss = {test_loss}")

