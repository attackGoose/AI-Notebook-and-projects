#pytorch and its domain libraries
import torch
from torch import nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape) #this will tell us 
        return self.classifier(x)
    

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

