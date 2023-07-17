#link the tutorial: https://www.youtube.com/watch?v=V_xro1bcAuA
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
#for machine learning datasets
from sklearn.datasets import make_circles
#this is one of the main problems with machine learning
#timestamp: 8:34:33

"""
what this covers (broadly):
1. architecture of a neuronetwork classification model
2. input and ourputshapes of a classification model (features and labels)
3. custom data to view, fit and predict on
4. steps in modeling
 4.5. creating the model, setting a loss func and an optimizer, creating a training and testing loop, evailuating the model
5. saving and loading the model
6. using non-linearity
7. different classification evaluation methods
"""

#there's a bunch of theory from 8:34:33 - 8:57:30, 8:53:10 has a few important points i should be aware of


#input and outputs:
input_tensor = torch.randn(size=(224, 224, 3)) #a 224 x 224 picture that uses RGB as its color complex (the 3 at the end is the R, G, B)

n_samples = 1000

#x is matrix, y is labels
X, y = make_circles(n_samples,
                    noise=0.3,
                    random_state=143)

print(len(X), len(y))

#from this, you can see that x has 2 features for each label of y
#also since y only has 0s and 1s, it uses binary classification, if its 0, 1, 2, etc, then it would be multi-class classification
print(f"first 5 samples of X: {X[:5]}\nfirst 5 samples of y: {y[:5]}") 

#making a dataframe: 
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

circles.head(10)


plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y)

plt.show()

 #the data we're working with is refered to a toy dataset, a dataset that's small enough to experiment but sizeable enough to practice the fundimentals