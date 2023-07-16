#this file is for learning pytorch by building a text generator
#https://pytorch.org/tutorials/
#https://youtu.be/V_xro1bcAuA


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

### creating a tensor
scalar = torch.Tensor(7) #has no dimensions, one number

print(scalar.shape) #returns the number we gave it

vector = torch.Tensor([7, 7]) #has 1 dim since its a 1 dim array
print(vector.shape)

MAXTRIX = torch.Tensor([[7, 8],
                        [9, 10]])

TENSOR = torch.Tensor([[[1,2,3],
                        [3,6,9],
                        [2,4,5]]])
print(TENSOR.shape)


### random tensors: (creates more accurate training data)

rand_tensor = torch.rand(3, 4) #creates a tensor of size 3, 4 (matrix since there's 2 dimensions) 
print(rand_tensor, rand_tensor.ndim)


#creating a random image tensor:
rand_image_size_tensor = torch.rand(size=(224, 224, 3)) #hgiht, width, color (RGB), also the size parameter isn't neededsince its the first parameter
print(rand_image_size_tensor.shape, rand_image_size_tensor.ndim)

zero_tensor = torch.zeros(size=(3, 4))

ones_tensor = torch.ones(size=(3, 4))

#the default data type of these tensors are float32, although u can customize that

#creating a tensor that has a range:
one_to_ten = torch.arange(start=1, end=10, step=2) #goes up by 2
print(one_to_ten)

ten_zeros = torch.zeros_like(one_to_ten) #creates a tensor of the same size as one_to_ten of all zero values


#NOTE: Tensor Datatypes:

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype = None, #type of the tensor: float_32, you can see a list of datatypes by googling "tensor datatypes pytorch"
                               device = None, #IMPORTANT: cuda uses gpu, operations on tensors need to be on the same device
                               requires_grad = False) #track gradiants with this tensor's operations?

float_16_tensor = float_32_tensor.type(torch.float16) #changing the type of a tensor
print(float_16_tensor)


print(float_16_tensor * float_32_tensor) #no error for some reason, dunno why, some operations will raise an error if they're not the same datatype

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int64)

print(int_32_tensor*float_32_tensor) #also works

### 3 big errors: 
# Tensor Wrong Datatype, use tensor.dtype to see the datatype
# Tensor not in the right shape, use tensor.shape to get the shape
# tensor not on the right device, use tensor.device to get the device

some_tensor = torch.rand(3, 4)
print(some_tensor, some_tensor.shape, some_tensor.dtype, some_tensor.device)


###Tensor Operations:

tensor = torch.tensor([1, 3, 6])
print(torch.add(tensor, 10)) #adds 100 to all the elements 

#for multiplication, try to always use torch's implimentation other than the basic implimentations like + and -, there's 2 types, element wise multiplication and dot multiplication/matrix multiplication
print(tensor*tensor) 

print(torch.matmul(tensor, tensor)) #prints out the products of the elements in the matrix after multiplication

# Rules for matrix multiplications: 
# 1. Inner dimentions must match: (2, 3) times (3, 2) will work since the inner dimensions both = 3, but (3, 2) times (3, 2) won't work since the inner dimensions aren't the same
print(torch.matmul(torch.rand([3,2]), torch.rand([2, 3])))

# 2. the resulting matrix will have the dimensions of the outer dimensions
print(torch.matmul(torch.rand([4,2]), torch.rand([2, 3])).shape) #creates a matrix with dimensions [4, 3]


# (Shape Error, a very common error in deep learning)
TensorA = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]])
TensorB = torch.tensor([[7, 8],
                        [9, 10],
                        [11, 12]])

#changing the shape of a tensor using transpose to switch the axis/dimensions of a tensor
TensorB = TensorB.T
print(TensorB, TensorB.shape)

print(torch.mm(TensorA, TensorB), torch.mm(TensorA, TensorB).shape) #mm = matmul, alias for writing less code, sideNote: @ is the sign fo matrix multiplication


##Tensor Aggrigation, finding the min, max, mean, sum, etc of a tensor

test_tensor = torch.arange(0, 100, 10)

#you could write this in 2 ways, both are shown below
print(test_tensor.min(), torch.min(test_tensor))
print(test_tensor.max(), torch.max(test_tensor))
print(test_tensor.type(torch.float32).mean(), torch.mean(test_tensor.type(torch.float32))) #gives us an error with datatype, so we have to change the datatype to what it wants
print(test_tensor.sum(), torch.sum(test_tensor))


#finding positional Datatypes, finds the position in the tensor that holds that value
print(f"index of the max position: {test_tensor.argmax()}\nindex of the min position: {test_tensor.argmin}")

#current time stamp: 2:59:28

"""
Solving Shape issues: reshaping, stacking, squeezing, and unsqueezing tensors:
reshaping - reshapes an input tensor to a defined shape
view - returns the view of an input tensor of a certain shape but keeps the same memory as the original tensor
stacking - combines multiple tensors on top of each other (vstack) or side by side (hstack)
squeeze - removes all '1' dimensions from a tensor
unsqueeze - adds a '1' dimension from a tensor
permute - returns the view of a tensor with the dimensions permuted (swapped) in a way

the purpose of these is to manipulate the dimensions of a tensor to resolve the shape issues
"""

another_tensor = torch.arange(1.0, 10.0)
print(another_tensor)
tensor_reshaped = another_tensor.reshape(3, 3) #changes the size/dimensions of a tensor into another one, the reshape has to be compatable with the original tensor
print(tensor_reshaped, another_tensor.reshape(9, 1))

another_tensorsView = another_tensor.view(1, 9)
print(another_tensorsView, another_tensorsView.shape)
#another_tensorView is a view of another_tensor, so by changing another_tensorsView, you will also change another_tensor

another_tensorsView[:, 0] = 5 #gets all rows of the tensor, and takes their value at the zeroth index and sets it equal to 5
print(another_tensor, another_tensorsView)


##Stacking Tensors
x = torch.arange(1.0, 10.0)

#there's also vstack and hstack, look into those later
x_stack = torch.stack([x, x, x, x], dim=0)
print(x_stack)

#squeezing tensors
x_squeeze = torch.squeeze(x)
print(f"original tensor: {x}\nsqueezed tensor: {x_squeeze}")

x_unsqueeze = x_squeeze.unsqueeze(dim=0) #it will add the extra dim into the dim given, which in this case is the 1st part of the dim, run the code and test to see more
print(f"previous target: {x_squeeze}\nprevious shape = {x_squeeze.shape}\nnew tensor: {x_unsqueeze}\nnew tensor shape: {x_unsqueeze.shape}")

#permuting tensors
x_original = torch.rand(size=(224, 224, 3)) #height, width, color_channels (RGB)
x_permuted = x_original.permute(2, 0, 1) #rearranges the color channel into the 0th index, and shifts the rest back by 1, shifts axis 0 -> 1, 1-> 2, 2-> 0
print(f"original shape: {x_original.shape}\npermuted shape: {x_permuted.shape}")


##Indexing / selecting data from tensors

new_tensor = torch.arange(1, 10).reshape(1, 3, 3) 
print(new_tensor)

print(new_tensor[0][0][0], new_tensor[0, 0, 0]) #these are both relatively the same, this shows 2 ways to refer to elements, one returns a list, the other returns a value

print(new_tensor[:, :, 1]) #the ":" would select all of a target dimension, this would get the last item of the 2nd dimensions 

#exercise: get all values of the 0th dimension but only 1 index value ofthe 1st and 2nd dimensions:
print(new_tensor[:, 1, 1])

#index of 0 of the 0th dimension and 1st dimension and all of the values of the 2nd dimension
print(new_tensor[0, 0, :])

#get the index to return 9, and index to return 3, 6, 9
print(new_tensor[:, :, 2], new_tensor[0, 2, 2])


##using Numpy with Pytorch

#converting data in numpy -> tensor
array = np.arange(1.0, 10.0)
numpy_to_tensor = torch.from_numpy(array).type(torch.float32)
print(array, numpy_to_tensor) #notice that the default datatype for the numpy array to tensor is a float64, so you have to manually specify it if needed

array += 1
print(array, numpy_to_tensor)
#also note that if you change the numpy array, it will not change the value of the tensor converted from it (they dont' share memory)


#converting data from tensor -> numpy
tensorN = torch.ones(7)
numpy_tensor = tensorN.numpy()
print(tensorN, numpy_tensor) # the default datatype of a tensor is float 32

tensorN += 1
print(tensorN, numpy_to_tensor)
#also note that if you change the numpy array, it will not change the value of the tensor converted from it


##reproducbility (trying to take the random out of random/guided values/basically like gradient descent?)

# 1st concept to reduce the randomness is a random seed, the random seed "flavors" the randomness: examples below

# how random tensors usually work: they usually just produce random numbers that have little to nothing to do wtih each other
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"{random_tensor_A}\n{random_tensor_B}")
print(random_tensor_A == random_tensor_B)

# setting the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(f"{random_tensor_C}\n{random_tensor_D}")
print(random_tensor_C == random_tensor_D)
#as you can see here, the torch.manual_seed() only works for one tensor at a time, so you have to set it again for another cell if you want it in another cell

#link for further information on reproducibility and random number generation: https://pytorch.org/docs/stable/notes/randomness.html https://en.wikipedia.org/wiki/Random_seed


# Running tensors on the GPU (and making faster computations since gpus are fast with numbers thanks to CUDA + NVIDIA hardware + pytorch working behind the scenes) 

# 1. Easiest way: use google colab for a free GPU (i'm not cus I'm dum), but you have to use a browser
# 2. Use your own gpu, which requires setup and purchasing 
# 3. Cloud computing: GCP, AWS, Azure, and other services allow you to rent computers on the cloud and access their gpu

# for 2 and 3, pytorch and CUDA drivers require a bit of setup if you get a gpu, so refer to pytorch documentation for the setup 

#check for gpu access with pytorch

print(torch.cuda.is_available())

#setup device agnostic code:
##since pytorch is capable of running on both the gpu and cpu, its best practice to setup device agnostic code: https://pytorch.org/docs/stable/notes/cuda.html#best-practices
device = "cuda" if torch.cuda.is_available() else "cpu"

#counts the number of gpu devices:
print(torch.cuda.device_count())


## Putting tensors (and models) on the GPU (because its faster with computations)

#create a tensor and change the device (if available)
GPU_tensor = torch.tensor([1, 2, 3])

tensor_on_gpu = GPU_tensor.to(device) #since i have no gpu, mine's will only be on my cpu, also the .to() method can also be used for moving models and tensors elsewhere
print(tensor, tensor.device)

#if tensor is on the gpu, you can't convert it to numpy, so you have to change it to the cpu first (device type errors, one ofthe most common type of errors)
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy() #the tensor_on_gpu should remain unchanged on the gpu
print(tensor_back_on_cpu)

#this only applies to one gpu, once you have more gpu's, refer to the pytorch documentation

### End of Pytorch Fundamentals, go to learnpytorch.io/00_pytorch_fundimentals/ you should find some exercises if I'm interested in practicing more on it



##continued in Learning_pytorch_workflow.py