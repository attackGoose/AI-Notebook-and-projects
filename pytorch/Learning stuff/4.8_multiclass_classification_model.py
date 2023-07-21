import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

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
y_blob = torch.from_numpy(y_blob).type(dtype=torch.float)

#split into training and testing sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

#plotting/visualizing:
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob)
plt.show()


# creating the model:
