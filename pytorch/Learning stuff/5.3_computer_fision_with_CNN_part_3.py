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

image, label = train_data[0]

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



### Building a Convolutional Neuro Network Model

"""
also, to see what's happening inside a CNN, see: https://poloclub.github.io/cnn-explainer this is a super helpful website that i highly suggest messing around with
"""

class FashionMNISTModelV2(nn.Module):

   """Model architecture that replicates the TinyVGG CNN as shown on the website linked above"""

   def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
      super().__init__()
      self.conv_block_1 = nn.Sequential(
         #creat a convolutional layer
         nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units, 
                  #other hyperparameters in a convolutional block
                  kernel_size=3, #the the size of the window to take the max pool over/looks at a 3x3 pixel area of the photo of a time
                  stride=1, 
                  padding=1),
         nn.ReLU(),
         nn.Conv2d(in_channels=hidden_units, 
                   out_channels=hidden_units, 
                   kernel_size=3, #looks at a 3x3 area of the photo at a time
                   stride=1,
                   padding=1),
         nn.ReLU(),
         #takes the max of the input
         nn.MaxPool2d(kernel_size=2) #looks at a 2x2 area at a time
      )
      self.conv_block_2 = nn.Sequential(
         nn.Conv2d(in_channels=hidden_units, 
                   out_channels=hidden_units,
                   kernel_size=3,
                   stride=1,
                   padding=1),
         nn.ReLU(),
         nn.Conv2d(in_channels=hidden_units,
                   out_channels=hidden_units,
                   kernel_size=3, #this is the same as typing in (3, 3) since kernel_size takes in a x by x area where u give the x value
                   stride=1,
                   padding=1),
         nn.MaxPool2d(kernel_size=2)
      )
      #this will group and classify the results given by the previous 2 convolutional layers
      self.classifier = nn.Sequential(
         #since the image is in a matrix, the nn.flatten layer is needed to flatten everything onto one dimension/into a single vector
         nn.Flatten(), #this multiplies all dims of a matrix and turns it into a vector
         #we multiply this by 7 because when we flatten the result of conv_block_2, we need to get the result of conv_block_2, which we can get through the print(result.shape)
         #in the forward function as shown below
         nn.Linear(in_features=hidden_units*7*7, #there's a trick to calculating this: 
                   out_features=output_shape)
      )
   
   def forward(self, x:torch.Tensor) -> torch.Tensor:
      x = self.conv_block_1(x)
      print(f"output shape of conv block 1: {x.shape}")
      x = self.conv_block_2(x)
      print(f"output shape of conv block 2: {x.shape}") #the shape is 7
      x = self.classifier(x)
      print(x.shape)
      return x

model_2 = FashionMNISTModelV2(
   input_shape=1, #our channels are (color channel, width, height) and since we only have 1 color channel
   hidden_units=10,
   output_shape=len(class_names)
).to(device)

#print(model_2.state_dict())
#creating dummy data:
torch.manual_seed(42)

#creating a batch of images
images = torch.randn(size=(32, 3, 64, 64)) #batch size, color channel, width, height
test_image = images[0]

print(f"image batch shape: {images.shape}\nsingle image shape: {test_image.shape}\ntest image:\n {test_image[:5]}")


#testing out a single convolutional layer: (for more information, refer to the interactive convolutional layer tool for visualizing a gui at: https://poloclub.github.io/cnn-explainer)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       #the kernel_size is 3 because it represents how many pizels get seen at a time, usually a 3x3 square is most common
                       kernel_size=3, #this would be the same as typing in (3, 3) for the kernal (also known as a filter) as well
                       #the stride represents how many pixels it moves by each time, a stride of 1 represents it skipping no pixels and moving one at a time, a stride of 2 
                       #represents it moving 2 spaces in the image at a time
                       stride=1,
                       ##how many additional pizels to add around the edge (this is so that the kernel can opperate there, which words together with stride to ensure that the
                       # kernel size can scan every pixel you want it to)
                       padding=1) #by adding the extra padding we're able to make it so that the test input shape and test output shapes are the same (with the exception of 1 dim)


conv_output = conv_layer(test_image.unsqueeze(0))
print(conv_output.shape)

#putting it through MaxPool2d:

max_pool_layer = nn.MaxPool2d(kernel_size=2) # the higher this number is the more the tensor's dimensions will get compressed (ex: a [1, 10, 64, 64] -> [[1, 10, 32, 32]])
# (creates a 2x2 kernel) mess around with these values to rediscover patterns

test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
#seeing what the layer does to the data
print(f"shape after going through conv_layer: {test_image_through_conv[:5]}")

# passing image through max_pool_layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"shape after going through both conv and max pool layer: {test_image_through_conv_and_max_pool.shape}")


#timestamp: 18:03:05

## visualizing it using a smaller tensor:
torch.manual_seed(42)

random_tensor = torch.randn(size=(1, 1, 2, 2)) #[batch size, color channels, height, width]
print(f"random tensor: {random_tensor}\nshape of random tensor: {random_tensor.shape}")

max_pool_layer = nn.MaxPool2d(kernel_size=2)

max_pool_tensor = max_pool_layer(random_tensor)

print(f"Max pool tensor: {max_pool_tensor}\nMax pool tensor shape: {max_pool_tensor.shape}")

"""
in summary: a convolutional layer compresses the image and then it can be passed through a ReLU layer, but then the MaxPool layer really compresses it and we use that data
to draw conclusions from, as for how much it compresses and how it varies, it all depends on the kernel size, ourput_shape padding and stride

this summary is shortened
"""

#visualizing the image we're passing into the model:
plt.imshow(image.squeeze(), cmap="gray")
plt.show()

#testing the model to see if it works (seeing if all the shapes align)

#creating a random image tensor to pass through:
rand_image_tensor = torch.randn(size=(1, 28, 28))
print(rand_image_tensor.shape)

dummy_pred = model_2(rand_image_tensor.unsqueeze(dim=0).to(device=device))


## setup loss function/eval metrics/optimizer

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.05)


#training/testing:

epochs = 10

