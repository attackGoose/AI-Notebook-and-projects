## timestamp: 19:44:05

#bookmark timestamp: 19:50:21 picture is helpful, the domain libearies are used for specialized types of data like text, image,s audio, or etc and have functions specialized 
# for these types of data

#another bookmark timestamp: 19:53:38 (the last one is data that's not in the training nor testing dataset, so predicting on real-world data)

#domain libraries: (since we're using this with a vision problem, this is going to use torchvision)

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


#standard image classification data format:  (found on torchvision documentation)
"""
pizza_sushi_steak/ <- overall dataset folder
    train/ <- train dataset folder
        pizza/ <- class of the image
            image1.png
            image2.png
            ...
        sushi/ <- class of the image
            image1.png
            image2.png
            ...
        steak/ <- class of the image
            image1.png
            image2.png
            ...
        ...
    test/ <- test dataset folder
        pizza/  <- class of the image
            image1.png
            image2.png
            ...
        sushi/ <- class of the image
            image1.png
            image2.png
            ...
        steak/ <- class of the image
            image1.png
            image2.png
            ...
        ...
"""

"""
steps for what to do next:
1. get all image paths
2. pick a random image path using random.choice()
3. get image's class name using pathlib.Path.parent.stem
4. open the image with "pillow" for image processing/manupulation
5. show image and print metadata
"""

import random
from PIL import Image

#setting random seed:
random.seed(42)

#1. getting all the imgae paths:
image_path_list = list(image_path.glob('*/*/*.jpg')) #glob is basically globbing together all of the images that are in a certain path and fit the criteria
                                                    #the * means it could be anything

random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem

#opening the imgae)
img = Image.open(random_image_path)

# printing image's metadata
print(f"Random image path: {random_image_path}\nImage class: {image_class}\nImage height: {img.height}\nImage width: {img.width}\nImage: {img}")

#visualizing the image data

img_as_array = np.asarray(img) #turning image into array

#showing the image using matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color channels]") #this is the order of the image properties for matplotlib (diff for pytorch)
plt.axis(False)
plt.show()

print(img_as_array)

"""
steps to use the image with pytorch:
1. turn the target data into tensors (in this case its the numerical representation of images)
2. then turn it into a torch.utils.data.Dataset (like in the previous unit) then subsequently into a torch.utils.data.DataLoader to use (respectively called Dataset and DataLoader)
"""

#transforming data using torchvision.transforms
data_transform = transforms.Compose([
    #resizing the image so that we always know the size of what we're working with (to make stuff easier to use)
    transforms.Resize(size=(64, 64)),
    #flip the imgaes randomly on the horizontal 
    transforms.RandomHorizontalFlip(p=0.5),
    # turn the igae into a torch tensor
    transforms.ToTensor()
])

data_transform(img) #passing in a PIL image
print(data_transform(img).shape)

def plot_transformed_images(image_paths:list, transform, n=3, seed=None):
    """
    selects random images from a path of images and loads/transforms them then plots the original v transformed version
    """
    if seed:
        random.seed(seed)
    random_image_path = random.sample(image_paths, k=n)
    for image_path in random_image_path:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            #transform and plot the target image
            transformed_image = transform(f).permute(1, 2, 0) # note that we will need to change the shape for matplotlib
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nSize: {transformed_image.shape}")
            ax[1].axis(False)

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_paths=image_path_list,
                        transform=data_transform,
                        n=3, #number of imgaes to do this on
                        )

plt.show()


"""
options for loading data:

loading images using "ImageFolder" (torchvision.datasets.ImageFolder)
"""

train_data = datasets.ImageFolder(root=train_dir, #it will look at this for its label/target
                                  transform=data_transform, #a transform for the data
                                  target_transform=None) #a transform for the label/target

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform,
                                 target_transform=None)

print(train_data, test_data)

#the above method will produce a torch dataset

#timestamp: 21:05:40
#getting the classes of the training data
class_names = train_data.classes
print(class_names)

#its also possible to get the classes and their indexes in a dictonary:
class_dict = train_data.class_to_idx
print(class_dict)

#checking the length of our data:
print(len(train_data), len(test_data))

#viewing our labels:
print(train_data.targets)

#viewing a single sample 
print(train_data.samples[0])

# use indexing on the train_data Dataset to get a single image and its label:
img, label = train_data[0][0], train_data[0][1]

print(f"imge tensor:\n{img}")
print(f"image shape: {img.shape}")
print(f"image datatype: {img.dtype}")
print(f"image Label: {label}")
print(f"label datatype: {type(label)}")

#plotting the image using matplotlib

img_permuted = img.permute(1, 2, 0) #matplotlib works with [height, width, color channels] so we have to convert it to their format

print(f"original image: {img.shape} -> [color channels, height, width]")
print(f"Permuted image: {img_permuted.shape} -> [height, width, color channels]")

#plotting the image:
plt.figure(figsize=(10, 7))
plt.imshow(img_permuted)
plt.axis(False)
plt.title(class_names[label], fontsize=14)


#creating the dataloaders: (creating iterables of custom batches of data, which is good for our memory and overall training), this can be used for all types of data
BATCH_SIZE = 1

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=1, # os.cpu_count(), #this is used to find out how many cpu's we have to maximize the cpu usage 
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=1,
                             shuffle=False)

print(train_dataloader, test_dataloader)

img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape} -> [batch_size, color_channels, height, width]")