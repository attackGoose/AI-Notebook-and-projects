#pytorch
import torch
from torch import nn

#torchvision stuff
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor #this converts a PIL image or a numpy.ndarray to a tensor

#for visualizing 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

print(f"Torch Version: {torch.__version__}\nTorchvision version: {torchvision.__version__}")

import requests
from pathlib import Path

##downloading helper functions if it doesn't exist
if Path("helper_functions.py").is_file():
   print("helper_function.py already exists, skipping download...")
else:
   print("downloading helper_functions.py")
   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
   with open("helper_function.py", "wb") as f:
      f.write(request.content)

from helper_functions import accuracy_fn

from tqdm.auto import tqdm

#creating a function to time the experiments/tracking model's efficiency
from timeit import default_timer as timer

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
   """prints the difference between the start and end time."""
   total_time = end - start
   print(f"Train time on {device}: {total_time:3f} seconds")
   return total_time

#copy and pasted from previous lesson
def eval_model(model: torch.nn.Module,
               data_loader: DataLoader, #torch.utils.data.DataLoader, this represents a set of data
               loss_func: torch.nn.Module,
               accuracy_func,
               device):
   
   """returns a dictionary containing the results of the model predicting on data_loader"""

   model.to(device=device)
   

   loss, acc = 0, 0

   model.eval()
   with torch.inference_mode():
      for X, y in tqdm(data_loader): #tqdm cus progress bar cool
         X, y = X.to(device), y.to(device)

         #making predictions:
         y_pred = model(X)

         # accumulate the loss and accuracy values per batch
         loss += loss_func(y_pred, y)
         acc += accuracy_func(y_true=y,
                              y_pred=y_pred.argmax(dim=1))
         
      #scale the loss and accuracy to find the average loss/acc per batch
      loss /= len(data_loader)
      acc /= len(data_loader)

   return {"model_name": model.__class__.__name__,
           "model_loss": loss.item(), #turning the loss into a single item after its been scaled
           "model_acc": acc}

## setting up device agnostic code:
device = "cuda" if torch.cuda.is_available() else "cpu"



##creating dataset again:
train_data = datasets.FashionMNIST(
   root="data",
   train=True,
   download=True,
   transform=torchvision.transforms.ToTensor(), #torchvision.transforms.ToTensor()
   target_transform=None
)

test_data = datasets.FashionMNIST(
   root="data",
   train=False,
   download=True,
   transform=torchvision.transforms.ToTensor(), #torchvision.transforms.ToTensor()
   target_transform=None
)

#other information to retrieve from the datasets
class_names = train_data.classes


BATCH_SIZE = 32

#batching together data
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

#creating a better model with non-linear activation functions to add non-linearity to check if it improves the performance
class FashionMNISTModelV1(nn.Module):
    def __init__(self, 
                input_shape: int,
                hidden_units: int,
                output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #flatten the layers to get a vector we could work with
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.layer_stack(x)
    
model_1 = FashionMNISTModelV1(
    input_shape=1*28*28, #this is the shape of the model, it has 1 color channel cus black and white image so the 1 represents the gray scale
    hidden_units=16,
    output_shape=len(class_names)
)

loss_func = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.05)

# functionizing the training step
def train_step(model: torch.nn.Module,
               data_loader: DataLoader, #torch.utils.data.DataLoader
               loss_func: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    
    """Trains the model"""

    train_loss, train_acc = 0, 0
    
    #train mode
    model.train()

    #add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        #calculate the loss
        loss = loss_func(y_pred, y)
        train_loss += loss #accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) #logits -> prediction labels
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
           print(f"looked at {batch * len(X)}/{len(data_loader.dataset)} samples")

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train Loss: {train_loss:5f} | Train Accuracy: {train_acc:2f}")

def test_step(model: torch.nn.Module,
              data_loader: DataLoader, #torch.utils.data.DataLoader
              loss_func: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
   
    test_loss, test_acc = 0, 0

    #testing mode
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):

            test_pred = model(X)
            
            loss = loss_func(test_pred, y)

            test_loss += loss
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))


        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:5f} | Test Accuracy: {test_acc:2f}")


epochs = 1

start_train_time = timer()

for epoch in tqdm(range(epochs)):

    print(f" Epoch: {epoch}\n-----------")

    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_func=loss_func,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_func=loss_func,
              accuracy_fn=accuracy_fn)


end_train_time = timer()


total_train_time = print_train_time(start=start_train_time, end=end_train_time)

print(total_train_time)

"""
According to the video, sometimes, depending on your data/hardware, your model might train faster on cpu than gpu

why is this?

1. it could be that your overhead for copying data/model to and from the gpu outweights the computational benefits offered by the gpu
2. the hardware you're using has a better compute capability than the gpu (rarer)

the compute time is heavily dependent on the hardware

for more information: https://horace.io/brrr_intro.html
"""

##getting model 1 results dictionary:

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_func=loss_func,
                             accuracy_func=accuracy_fn,
                             device=device)

print(model_1_results)


##continued in part 3