## timestamp: 19:44:05

#bookmark timestamp: 19:50:21 picture is helpful, the domain libearies are used for specialized types of data like text, image,s audio, or etc and have functions specialized 
# for these types of data

#another bookmark timestamp: 19:53:38 (the last one is data that's not in the training nor testing dataset, so predicting on real-world data)

#domain libraries: (since we're using this with a vision problem, this is going to use torchvision)

#pytorch and its domain libraries
import torch
from torch import nn

import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"


#getting the custom dataset (starting on a small scale then increasing the scale when necessary (speeding up experiments))
# (subset of the food 101 dataset that only has 3 categories of food and 10 percent of the images, so around 75 training images per class and 25 testing images per class):

import requests
import zipfile 
from pathlib import Path

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