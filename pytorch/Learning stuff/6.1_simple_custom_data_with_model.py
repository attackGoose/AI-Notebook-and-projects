#pytorch and its domain libraries
import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
#getting the custom dataset (starting on a small scale then increasing the scale when necessary (speeding up experiments))
# (subset of the food 101 dataset that only has 3 categories of food and 10 percent of the images, so around 75 training images per class and 25 testing images per class):

import requests
import zipfile 
from pathlib import Path

from typing import Tuple, Dict, List
#setup path to a data folder
data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"


#if the data doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} already exists, skipping download...")
else:
    print(f"{image_path} does not exist, creating one")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("downloading image data...")
        f.write(request.content)

    #unzip the data:
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("unzipping image data")
        #extracts data from the zipfile into the other file
        zip_ref.extractall(image_path)

# data preparation and exploration

import os

def walk_through_dir(dir_path):
    """walks through dir_path, returning its contents."""

    for dirpath, dirnames, filenames in os.walk(dir_path): #hover over the .walk() function to see what it does incase I forget
        print(f"There are {len(dirnames)} directories and {len(filenames)} imgaes in {dirpath}") #this shows how many test images and train images are in its corresponding path

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

import random
from PIL import Image

from torch.utils.data import Dataset

import pandas as pd

image_path_list = list(image_path.glob('*/*/*.jpg'))

#making the model to use the data on: (tinyVGG without augmentation first) link to the CNN explainer website: https://poloclub.github.io/cnn-explainer

#creating transforms and loading data for the model:
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)), #this shape is the same as the cnn explainer on the website
    transforms.ToTensor()
])

#loading and transforming data for the model
train_data_simple = datasets.ImageFolder(
    root=train_dir,
    transform=simple_transform
)

test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=simple_transform
)

class_names = train_data_simple.classes

#turning datasets into dataloaders:
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader_simple = DataLoader(
    dataset=train_data_simple,
    batch_size=BATCH_SIZE,
    #num_workers=NUM_WORKERS, #whenever I add this in it freezes everything and kills the script
    shuffle=True
)

test_dataloader_simple = DataLoader(
    dataset=test_data_simple,
    batch_size=BATCH_SIZE,
    #num_workers=NUM_WORKERS, #whenever I add this in it freezes everything and kills the script
    shuffle=False
)


#making a TinyVGG model
class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, 
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, #this has to be multiplied by soemthing later on cus flatten(), finding out by passing the data in and changing it to that
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.conv_block_1(x)
        #print(x.shape)
        #x = self.conv_block_2(x)
        #print(x.shape) #this will tell us 
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
    

#opperator fusion: the most important optimization in deep learning compilers: https://horace.io/brrr_intro.html there's a good definition of it in here, do ctrl g and search
torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device=device)

# passing through dummy data to find out the final shape of the input in the classifier (aka using a single image batch as a test)

image_batch, label_batch = next(iter(train_dataloader_simple))

dummy_logit = model_0(image_batch.to(device))

#this tells us that the shape of the matrix after being flattened into a vector is a [32x2560] and 2560 is 10*16*16 (which is also the output shape of the last layer it goes through)
#the value will change every time you change the padding, stride, output and output shape
print(dummy_logit)

#the best way to find out the shape is simply to create some print statements, and pass through a single batch to test if it works or not

#using torchinfo to print out a summary of the model (aka an idea of the shapes of the model so we don't have to print them out manually in the model)
from torchinfo import summary

summary(model=model_0, input_size=(BATCH_SIZE, 3, 64, 64)) #input size is the the size of the input that you're putting into your model
#in its internals, its doing a forward pass of the input size through the model you give it

#side note: a parameter is a adjustable weight within the model, and this model, according to summary, has 8000 parameters, which is considered small compared to modern models
from Functionizing_training_code import train_step_multiclass, test_step_multiclass, epoch_loop_train
from helper_functions import accuracy_fn

#training and testing the model:
#creating a new model instance:
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device=device)

loss_func = torch.nn.CrossEntropyLoss()

print(f"Model parameters: {model_0.parameters()}")
optimizer = torch.optim.Adam(params=model_0.parameters(),
                            lr=0.001) #this is the default value

NUM_EPOCHS = 5

from timeit import default_timer as timer
start_time = timer()

#training model 0
model_0_results = epoch_loop_train(model=model_0,
                                train_dataloader=train_dataloader_simple,
                                test_dataloader=test_dataloader_simple,
                                loss_func=loss_func,
                                optimizer=optimizer,
                                device=device,
                                epochs=NUM_EPOCHS)

end_time = timer()

total_model_0_train_time = end_time - start_time

print(f"Model 0 results: {model_0_results}")

print(f"Total train time: {total_model_0_train_time:3f}")

#checking model result keys
print(model_0_results.keys())
#plotting the loss curve to trach the model's progress
def plot_loss_curves(results: Dict[str, List[float]]): #the model takes in a dictionary that contains a string that has a list of floats as its value
    """Plot training curve of a result dictionary"""
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    num_epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1) #1 row, 2 cols, and index 1
    plt.plot(num_epochs, loss, label="train_loss")
    plt.plot(num_epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(num_epochs, accuracy, label="train_acc")
    plt.plot(num_epochs, test_accuracy, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


plot_loss_curves(model_0_results)
plt.show()

# our model is underfitting, severely underfitting
# what an ideal loss curve should look like: https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic 
# a loss curve is one of the best ways to troubleshoot a model

#how to deal iwth underfitting and overfitting: timestamp: 1:00:05:01, course: https://www.youtube.com/watch?v=V_xro1bcAuA

"""
ways to deal with overfitting:

1. getting more data: gives the model more chances to learn patterns between the samples, can be done through getting more data, augmenting existing data or getting better data
    this increased diversity hopefully forces the model to learn more generalizable patterns
2. use transfer learning on another pre-built working model (refer to torchvision.models library for vision problems, and etc for their specific use cases)
3. simplify your model
4. use learning rate decay (how much the optimizer updates the model's weights every step) to prevent the model from yea (refer to torch.optim.lr_scheduler to adjust learning 
    rate overtime) 
5. there's also early stopping, where you save your model where it performed the best rather than saving the model at the latest step

ways to deal with underfitting:

1. adding more layers/units to your model (opposite of solving overfitting)
2. tweaking the learning rate (same as overfitting)
3. train for longer, 
4. use transfer learning on another working model
5. use less regularization (regulate the model's stuff less, regularization is used to prevent overfitting (includes all steps in overfitting), but when overdone it causes this)

machine learning is all about balancing between underfitting and underfitting and is a very prevalent area of machine learning research
"""

# changing the model data to prevent underfitting: TinyVGG w/ data augmentation (using trivial augment)
train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), #the 31 represents the magnitude of augmentation
    transforms.ToTensor()
])

test_transform_simple = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

#creating the dataset and dataloaders using the transforms
train_data_augmented = datasets.ImageFolder(root=train_dir,
                                            transform=train_transform_trivial)

test_data_simple = datasets.ImageFolder(root=test_dir, 
                                           transform=test_transform_simple)

BATCH_SIZE = 32

train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        #num_workers=os.cpu_count()
                                        )

test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    #num_workers=NUM_WORKERS
                                    )

#creating a new instance of the model
torch.manual_seed(42)
model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device=device)

#training the model
torch.manual_seed(42)

NUM_EPOCHS = 5

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model_1.parameters(),
                             lr=0.001)

start_time = timer()

model_1_results = epoch_loop_train(model=model_1,
                                   train_dataloader=train_dataloader_augmented,
                                   test_dataloader=test_dataloader_simple,
                                   loss_func=loss_func,
                                   optimizer=optimizer,
                                   device=device,
                                   epochs=NUM_EPOCHS)

end_time = timer()

print(f"Total train time for model_1: {end_time-start_time:.3f} seconds")

print(model_1_results)

#plot its loss curves
plot_loss_curves(results=model_1_results)
plt.show()
#model is prob both overfitting and underfitting, for some reason the test acc and loss is extremely weird


"""Its always tood to compare your model results to one another to see where the model might need more work, what works, and what doesn't work

ways to compare the modeling experiments to one another: 
1. hard coding functions to do it for us (using this for now since it involves raw pytorch) (method isn't prefered since when there are too many graphs it gets hard to read)
2. Pytorch tensorboard: https://pytorch.org/docs/stable/tensorboard.html
3. weights and biases: https://wandb.ai/site
4. MLFlow: https://mlflow.org/
"""

model_0_df = pd.DataFrame(data=model_0_results)
model_1_df = pd.DataFrame(data=model_1_results)

print(model_0_results)

#setting up a plot
plt.figure(figsize=(15, 10))

epochs = range(len(model_0_df))

#plotting train loss
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

#plotting train acc
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

#plotting test loss
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

#plotting test acc
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

plt.show()


#making a prediction on a single custom image:
custom_image_path = data_path/"04-pizza-dad.jpeg"

#if it doesn't exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        #downloading it from github using the raw image link
        request = requests.get(url="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"downloading {custom_image_path}")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download")


#loading in the imgae with pytorch: making sure the imgae is in the same format as the model it will be trained on: in tensorform, with dtype float32, shape: 63, 64, 3,on device
#reading the imgae into pytorch using: https://pytorch.org/vision/stable/generated/torchvision.io.read_image.html#torchvision.io.read_image 

#io stands for input output
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path)) #the type is uint8

#viewing the iage cus y not:
plt.imshow(custom_image_uint8.permute(1, 2, 0))
plt.axis(False)
plt.show()

#getting metadata of the image
print(f"Custom Image Tensor:\n{custom_image_uint8}\nCustom Image Shape: {custom_image_uint8.shape}\nCustom Image datatype: {custom_image_uint8.dtype}")

# we need to change the type to the same type that the model uses (float32) otherwise its going to return an error
