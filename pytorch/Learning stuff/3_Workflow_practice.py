import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

#check if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"

#creating data using cubic regression formula
weight = 3
bias = 7.0

#range values
start = 0
end = 1
step = 0.01

#x is usually a feature matrix and same iwth y
X = torch.arange(start, end, step)
print(X.shape)
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

torch.manual_seed(143)

model = LinearRegressionModel()

print(model, list(model.state_dict()))

#setting the data and model to use the target device:
model.to(device=device)

X_test = X_test.to(device=device)
X_train = X_train.to(device=device)
y_test = y_test.to(device=device)
y_train = y_train.to(device=device)

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

    y_prediction = model(X_train.reshape(shape=(80, 1)))

    #creates a loss value based on difference in predicted value and the expected value
    loss = loss_func(y_prediction, y_train)

    #reset gradients
    optimizer.zero_grad()

    #back prop
    loss.backward()


    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test.reshape(shape=(20, 1)))

        test_loss = loss_func(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"epoch = {epoch} | loss = {loss} | test loss = {test_loss}")

print(model.state_dict())

model.eval()
with torch.inference_mode():
    y_pred = model(X_test.reshape(shape=(20, 1)))


#this should run into an error because its working with a tensor but idk why its not, so i'm not going to fix it since its not broken
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


plot_prediction(predictions=y_pred)

#save
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Linear_layer_model.pth"

MODEL_SAVE_PATH = MODEL_NAME / MODEL_PATH

print(f"saving to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)

loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

print(loaded_model.state_dict())
loaded_model.eval()
with torch.inference_mode:
    new_pred = loaded_model(X_test.reshape(shape=(20, 1)))
    print(new_pred == test_pred)


#continued in neuronetwork_classification.py