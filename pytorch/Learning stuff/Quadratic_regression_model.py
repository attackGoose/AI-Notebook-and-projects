import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

#creating known parameters
a_value = 1
b_value = 1
c_value = -1

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = (a_value*X**2) + (b_value*X) + c_value


#training dtaa and training split:
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


#for visualizing the data:
def plot_predictions(train_data = X_train,
                     train_label = y_train,
                     testing_data = X_test,
                     testing_label = y_test,
                     predictions = None):
    
    plt.figure(figsize=(10, 10))

    plt.scatter(train_data, train_label, c="blue", s=4, label="Training Data")

    plt.scatter(testing_data, testing_label, c="purple", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(testing_data, predictions, c="green", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()


class QuadraticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_value = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, 
                                                dtype=torch.float))
        self.b_value = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, 
                                                dtype=torch.float))
        self.c_value = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, 
                                                dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a_value*x**2) + (self.b_value*x) + self.c_value
    

#for saving the model later on if needed
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "Quadratic_Regression_Model.pth"

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


test_model = QuadraticRegressionModel()

#loss function:
loss_func = nn.L1Loss()

#optimizer:
optimizer = torch.optim.SGD(test_model.parameters(), lr=0.02)

#data collection for plotting
epoch_count = []
loss_values = []
test_loss_values = []
epoch = 1000

for epoches in range(epoch):
    test_model.train()

    y_pred = test_model(X_train)

    loss = loss_func(y_pred, y_train) 

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    ##Testing
    test_model.eval()
    with torch.inference_mode():
        test_pred = test_model(X_test)

        #loss calc
        test_loss = loss_func(test_pred, y_test)

        if epoches % 10 == 0:
            epoch_count.append(epoches)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoches} | Loss: {loss} | Test loss: {test_loss}")

            print(test_model.state_dict())

#saving the model:
print(f"saving to: {MODEL_SAVE_PATH}")
torch.save(obj=test_model.state_dict(),
           f=MODEL_SAVE_PATH)


#for plotting and seeing the results
plt.plot(torch.tensor(epoch_count).numpy(), torch.tensor(loss_values).numpy(), label="Train loss")
plt.plot(torch.tensor(epoch_count).numpy(), torch.tensor(test_loss_values).numpy(), label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

plot_predictions(predictions=test_pred)


##loading a model:

new_model = QuadraticRegressionModel()

new_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

