
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
from torchinfo import summary

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
        f.write(request.content)

    #unzip file:
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
       zip_ref.extractall(image_path)

    os.remove(data_path / "pizza_steak_sushi.zip")

train_dir = image_path / "train"
test_dir = image_path / "test"

from functionizing_creating_dataloaders import create_dataloaders
from Functionizing_training_code import epoch_loop_train
from timeit import default_timer as timer
from helper_functions import plot_loss_curves

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

"""
in this case, since I'm building a computer vision model for learning this, I'm going to be using torchvision.models (available at https://pytorch.org/vision/stable/models.html#classification)

in the models section, usually, the higher the number in the model name (e.g. efficientnet_b0() -> efficientnet_b1() -> efficientnet_b7()) the better the performance 
but the larger the model
while having good performance is beneficial, having a model that's too big means that it won't always fit onto the device we're working with
understanding the performance v speed v size trade off comes with time and practice

I'm using the efficientnet_bX (where X is a number) to practice but its important to **not get attatched to any one architecture** as they are always changing with new research
its best to experiment with different architectures to see which one works best for the problem you're working with
"""

#setup pretrained model with pretrained weights and sending it to target device: (the pretrained model that I'm using is going to be the efficientnet_b0 model)
model_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT #usually default = the best available weights
model = torchvision.models.efficientnet_b0(weights=weights).to(device=device)

#print(model) #super long

"""
the efficientnet model uses 3 main sections: 

features - a collection of convolution layers and other various activation layers to learn a base represent of the vision data (layer is often refered to as features or feature
            extractor)

avgpool = takes the output of the features layer(s) and turns it into a feature vector

classifier = turns the feature vector into a vector with the same dimensionality as the number of required output classes 

#most of these layers have been made in the previous tutorials in the previous CNN files
"""

#using torchinfo.summary() to summarize the model: 
summary(model=model,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]) #this shows all of the information on the model as well as if some of the parameters are trainable or not

#changing parts of the the base model and output layer to suit our own problem: (since our problem only has 3 output layers as compared to the model's 100)

#freezing all the layers/parameters in the "features" section of the model
for param in model.features.parameters():
    param.requires_grad = False #by turning off grad, it becomes untrainable, or "frozen"

#adjusting the output layer to fit our needs (since we only have 3 output classes as compared to the model's 1000 classes)
#the output is controlled by the classifier portion of the model, current classifier consists of Dropout(0.2, inplace=True), Lienar(in_features=1280, out_features-1000, bias=True)

output_shape = len(class_names)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=output_shape, bias=True)
).to(device=device)

#getting the summmary of the modified model: 
summary(model=model,
        input_size=(32, 3, 224, 224),
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)

start_time = timer()

#training the new model
results = epoch_loop_train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    device=device,
                    epochs=5)

end_time = timer()

print(f"Total training time: {(end_time-start_time):.3f} seconds") #idk why my test data is so weird, loss starts low and acc is also low so idk

#evaluating the model via loss curves
plot_loss_curves(results=results)
plt.show()

#i have no idea why my test accuracy is absolutely poop, I think I just need more test data (since the test loss is also at a all time low which is prob why it no learn)

#making predictions on a custom image:

"""
things to keep in mind:

1. Shape errors: if our images are a different shape to what our model is trained on, we'll get a shape error
2. Datatype errors: if our images are a different datatype than what our model uses (i.e. float32, int8, or float64 just to make a few) we'll get an error
3. Device errors: if our images are on a different device than our model then we'll get a device error
4. Transformation mistakes: if our model is trained on images that have been transformed in a certain way, if we try to make predictions in another way, then the predictions might
    be off
"""
from typing import Tuple, Dict, List
from PIL import Image

def pred_and_plot_image(model: torch.nn.Module,
                        class_names: List[str],
                        image_path: str,
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
    img = Image.open(image_path)
    
    #create transform for image if it doesn't exist
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    #predicting on the image:
    model.to(device=device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0) #adding in extra dimension

        target_image_pred = model(transformed_image.to(device))
    
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_label = torch.argmax(target_image_pred_probs, dim=1)
    
    #plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_label]} | Prob: {target_image_pred_probs:.3f}")
    plt.axis(False)

import random

num_img_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_image_sample = random.sample(population=test_image_path_list,
                                  k=num_img_to_plot)

for image in test_image_sample:
    pred_and_plot_image(model=model,
                        class_names=class_names,
                        image_path=image_path,
                        transform=weights.transforms(),
                        image_size=(224, 224))
    
    plt.show()