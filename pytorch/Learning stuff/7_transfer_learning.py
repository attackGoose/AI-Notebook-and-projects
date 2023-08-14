
#transfer learning is essentially taking the weights of a already well performing model and adjusting those already trained weights to fit our own problem
#the main thing to worry about in this situation is finding the model to use it on

#it is recommend to use transfer learning wherever and whenever possible since it often achievest he greatest result with less custom data
"""
where to find pre-trained models for transfer learning:

1. pytorch domain libraries like torchvision.models, torchtext.models, torchaudio.models, or torchrec.models
2. HuggingFace Hub: https://huggingface.co/models, https://huggingface.co/datasets
3. timm (pytorch image models library that has all of the latest and greatest computer vision models and other helpful features): https://github.com/rwightman/pytorch-image-models
4. Paperswithcode (a collection of the latest state-of-the-art machine learning papers with code implimentations attatched and their menchmarks): https://paperswithcode.com/
"""

#setting up:
import torch
from torch import nn
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu" #note that the cuda just refers to the gpu

#getting a dataset to work with:
import os
import zipfile
from pathlib import Path

import requests

data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"

#if the image folder doesn't already exist, download and prepare it
if image_path.is_dir():
    print(f"Image path already exists, skipping download...")
else:
    print(f"did not find {image_path} directory, creating one")
    image_path.mkdir(parents=True, exist_ok=True)

    #download the images
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("downloading data")
        f.write(request)

    #unzip file:
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
       zip_ref.extractall(image_path)

    os.remove(data_path / "pizza_steak_sushi.zip")

train_dir = image_path / "train"
test_dir = image_path / "test"

from functionizing_creating_dataloaders import create_dataloaders


#creating a manualtransform for the pretrained model that matches its training data:
manual_pretrained_model_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)), #this is the size that this model uses
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
print(manual_pretrained_model_transforms)
#applying the manual transform and creating a dataset
BATCH_SIZE = 32

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transforms=manual_pretrained_model_transforms,
                                                                    batch_size=BATCH_SIZE)

print(f"{train_dataloader}\n{test_dataloader}\n{class_names}")


#creating a transform for torchvision.models (auto creation): link for more info on available options on other architectures: https://pytorch.org/vision/main/models.html

#selecting the default model weights (which is usually the most optimal/best):
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT #.DEFAULT is the best weights available from pretraining on ImageNet

#accessing the transform associated with the weights
auto_transforms = weights.transforms()
print(auto_transforms)

# the benefit of using an auto transform is that you can ensure you're using the same data transformation that the pretrained model used when it was trained, however it can not 
# be customized 

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transforms=auto_transforms,
                                                                    batch_size=BATCH_SIZE)


#getting a pre-trained model: (and customizing it to our specific use case)
