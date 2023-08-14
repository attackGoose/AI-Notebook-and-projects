import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                      test_dir: str,
                      transforms: transforms.Compose,
                      batch_size: int
                      #num_workers: int = NUM_WORKERS #this freezes my terminal for some reason and i don't know how to solve it so I'm excluding this
                      ):
    """Creates train and test dataloaders
    
    parameters: train_dir = training directory
                test_dir = testing directory
                transforms = torchvision transform to perform on the training data
                batch_size = number of samples per batch in each dataloader

    returns:
        a tuple of (train_dataloader, test_dataloader, class_names) where class_names is a list of the target classes
    """

    #use ImageFolder to create datasets:
    train_data = datasets.ImageFolder(root=train_dir, transform=transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=transforms)

    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True)
    
    return (train_dataloader, test_dataloader, class_names)