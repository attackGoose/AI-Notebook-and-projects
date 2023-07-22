import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import accuracy_fn, plot_decision_boundary

##Creating a mutli-class classification model:

"""to reiterate over the different types ofclassification models and what they do:
Binary class classification: one or the other (cat v dog, spam v not spam, fraud v not fraud, cow v pig, etc, one or the other) 
Multiclass classification: can be classified as more than one thing or another or multiple things
"""

## Creating the data:

NUM_FEATURES = 2
NUM_CLASSES = 4
RANDOM_SEED = 42

#setting hyperparameters:
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

#changing them into tensors
X_blob = torch.from_numpy(X_blob).type(dtype=torch.float)
y_blob = torch.from_numpy(y_blob).type(dtype=torch.long) #this one shoudl be long tensor because from cross entropy loss it needs to be of type long

#split into training and testing sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

#plotting/visualizing:
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob)
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

# creating the model:
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()

        """
        input_features = number of input features to the model
        output_features = number of output features of the output classes
        hidden_units = number of hidden units inbetween layers, default is 8
        """

        """self.layer_1 = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=output_features)
        self.relu = nn.ReLU()"""

        self.Linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Linear_layer_stack(x)
        #return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

Blobby = BlobModel(input_features=2,
                   output_features=4,
                   hidden_units=8).to(device=device)

#loss function: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html weight would be useful for an unbalanced training set/a dataset that has imbalanced
#samples/different number of red dots from blue or from blue to green or etc
loss_func = torch.nn.CrossEntropyLoss() 

optimizer = torch.optim.SGD(params=Blobby.parameters(),
                            lr=0.05)

### putting everything together:

#raw outputs: (logits)
#and since its outputing logits, we have to first convert it to prediction probabilities, then to prediction labels

#getting the data without training:
X_blob_test = X_blob_test.to(device=device)
X_blob_train = X_blob_train.to(device=device)
y_blob_test = y_blob_test.to(device=device)
y_blob_train = y_blob_train.to(device=device)

Blobby.eval()
with torch.inference_mode():
    y_logits = Blobby(X_blob_test)
    y_pred_probs = torch.softmax(y_logits, dim=1)

    #comparind raw logit ot actual value:
    print(y_logits[:10])
    print(y_pred_probs[:10])
    print(y_blob_test[:10])

    #converting prediction probs into prediction labels:
    y_preds = torch.argmax(y_pred_probs, dim=1)
    print(y_preds[:10])

    #remember that for softmax, it returns the probability of every value being the right one, which all together add up to 1.0, so you have to return the position that it is
    #showing as the prediction, and you can do that through the argmax() function which takes the index of the largest value


"""Steps for a multi-classification model:

logits (raw outputs of the model) -> prediction probs (using torch.softmax()) -> prediction labels (taking the argmax() of the prediction probabilities)
"""

##Training/Testing the model:
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 200

for epoch in range(epochs):
    Blobby.train()

    y_logits = Blobby(X_blob_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_func(y_logits, y_blob_train.type(torch.LongTensor))
    train_acc = accuracy_fn(y_true=y_blob_train,
                            y_pred=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    Blobby.eval()
    with torch.inference_mode():
        test_logits = Blobby(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        Test_loss = loss_func(test_logits, y_blob_test.type(torch.LongTensor))
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_preds)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss:4f} | Train Accuracy: {train_acc:2f} | Test Loss: {Test_loss:4f} | Test Accuracy: {test_acc:2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(Blobby, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(Blobby, X_blob_test, y_blob_test)
plt.show()

#timestamp: 13:38:24