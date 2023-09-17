# Introduciton to Pytorch


- [Overview](#overview)
  - [Creating a Virtual Environment](#creating-a-virtual-environment)
    - [Using a GPU?](#using-a-gpu)
    - [Recommended Code Structure](#recommended-code-structure)
    - [Running Experiments](#running-experiments)
        - [Training and Evaluation](#training-and-evaluation)
        - [Hyperparameter Search](#hyperparameter-search)
        - [Display the Results of Multiple Experiments](#display-the-results-of-multiple-experiments)
- [PyTorch Introduction](#pytorch-introduction)
    - [Goals of This Tutorial](#goals-of-this-tutorial)
    - [Resources](#resources)
    - [Code Layout](#code-layout)
- [Tensors and Variables](#tensors-and-variables)
    - [Changing Datatypes](#changing-datatypes)
    - [Automatic Differentiation](#automatic-differentiation)
    - [Disabling Automatic Differentiation](#disabling-automatic-differentiation)
        - [Using requires_grad=False](#using-requires_gradfalse)
        - [Using torch.no_grad()](#using-torchno_grad)
            - [Related: Using model.eval()](#related-using-modeleval)
- [Parameters](#parameters)
    - [nn.Parameter Internals](#nnparameter-internals)
    - [Difference Between Parameter vs. Tensor in PyTorch](#difference-between-parameter-vs-tensor-in-pytorch)
- [Core Training Step](#core-training-step)
- [Models in PyTorch](#models-in-pytorch)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Training vs. Evaluation](#training-vs-evaluation)
- [Computing Metrics](#computing-metrics)
- [Saving and Loading Models](#saving-and-loading-models)
- [Using the GPU](#using-the-gpu)
- [Painless Debugging](#painless-debugging)
- [Selected Methods](#selected-methods)
    - [Tensor Shape/size](#tensor-shapesize)
    - [Initialization](#initialization)
        - [Static](#static)
    - [Standard Normal](#standard-normal)
        - [Xavier/Glorot](#xavierglorot)
            - [Uniform](#uniform)
            - [Normal](#normal)
        - [Kaiming/He](#kaiminghe)
            - [Uniform](#uniform-1)
            - [Normal](#normal-1)
    - [Send Tensor to GPU](#send-tensor-to-gpu)
    - [Convert to NumPy](#convert-to-numpy)
    - [tensor.item(): Convert Single Value Tensor to Scalar](#tensoritem-convert-single-value-tensor-to-scalar)
    - [tensor.tolist(): Convert Multi Value Tensor to Scalar](#tensortolist-convert-multi-value-tensor-to-scalar)
    - [Len](#len)
    - [Arange](#arange)
    - [Linspace](#linspace)
    - [View](#view)
    - [Transpose](#transpose)
        - [Swapaxes](#swapaxes)
    - [Permute](#permute)
    - [Movedim](#movedim)
    - [Randperm](#randperm)
    - [Where](#where)
    - [Reshape](#reshape)
    - [Concatenate](#concatenate)
    - [Squeeze](#squeeze)
    - [Unsqueeze](#unsqueeze)
    - [Print Model Summary](#print-model-summary)
    - [Resources](#resources-2)
- [References](#references)
- [Citation](#citation)
  
[![Colab Notebook](https://aman.ai/primers/assets/colab-open.svg)](https://colab.research.google.com/github/amanchadha/aman-ai/blob/master/pytorch.ipynb)


## Overview

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible platform for building deep learning models and is known for its dynamic computational graph, which makes it particularly suitable for research. Originating as a research-centric tool, PyTorch has gained immense popularity among researchers and developers alike due to its ease of use, flexibility, and powerful capabilities.

-   This tutorial gives an overview of the Pytorch for deep learning model building, trainng and evaluation with practical examples as well as descrition of possible project environments

## Options for utilising GPU and environments

When diving into PyTorch, you have multiple options in terms of development environments and GPU resources. Two of the most popular choices are Visual Studio Code (VSCode) and Google Colab:

### Visual Studio Code (VSCode)
Description: VSCode is a free, open-source code editor developed by Microsoft. It supports a variety of programming languages and has a rich ecosystem of extensions, including support for Python and PyTorch.

### Google Colab:
 Offers cloud-based GPUs, which can be especially beneficial if your local machine doesn't have a powerful GPU or any GPU at all. However, there are usage limits to be aware of, as prolonged or heavy usage might lead to temporary restrictions, a longer training might stop unexpectedly due to lack of memory. It's essentially a Jupyter notebook environment that requires no setup. **_Unfortunatelly, there is only limited amount of computing power without credits and there is no free trial._**  

 **GPU Usage**: Google Colab provides free GPU access. To enable it, go to Runtime > Change runtime type > and select GPU under hardware accelerator.

### GPU Differences Between VSCode and Google Colab
VSCode: Utilizes your local machine's GPU. This means the performance is dependent on your hardware. Setting up GPU support might require additional configurations, especially for CUDA compatibility.

GPU Usage: If you have a local GPU, you can set up PyTorch to utilize it directly from VSCode. This requires proper drivers and CUDA installation.

Should you prefer using VSCode or other IDEs surhc as Pycharm, but you need GPU performance from Google Colab, there are ways to connect from VSCode to you account. For this please refer to other tutorials.
- [Connecting to Google Colab from VSCode](https://saturncloud.io/blog/is-it-possible-to-connect-vscode-on-a-local-machine-with-google-colab-the-free-service-runtime/#:~:text=In%20conclusion%2C%20it%20is%20possible,GPU%20runtime%20offered%20by%20Colab.)
### Free power of GPU? Use Azure student account! 👌👌👌
Microsoft Azure gves free 100$ for students which migh come helpful when the working with larger models. Making mutliple accounts for this purpose can cover most or all processing needs of the group in AI Track. It is again possibel to connect to the remote copute while using these credits and simply run on Azure GPU while still using project in VSCode: 
- [Tutorial on remotecomputes using VSCode and Microsoft Azure](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-launch-vs-code-remote?view=azureml-api-2&tabs=extension)

Alternatively avoid training large models from sratch, this will be adressed in the next tutorial.
## PyTorch Introduction 


### Goals of This Tutorial

-   Learn more about PyTorch.
-   Learn an example of how to correctly structure a deep learning project in PyTorch.
-   Understand the key aspects of the code well-enough to modify it to suit your needs.

### Resources

-   The main PyTorch [homepage](https://pytorch.org/).
-   The [official tutorials](https://pytorch.org/tutorials/) cover a wide variety of use cases- attention based sequence to sequence models, Deep Q-Networks, neural transfer and much more!
-   A quick [crash course](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) in PyTorch.
-   Justin Johnson’s [repository](https://github.com/jcjohnson/pytorch-examples) introduces fundamental PyTorch concepts through self-contained examples.
-   Tons of resources in this [list](https://github.com/ritchieng/the-incredible-pytorch).

## Tensors and Variables

-   Before going further, we strongly suggest going through [60 Minute Blitz with PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) to gain an understanding of PyTorch basics. This section offers a sneak peak into the same concepts.
    
-   PyTorch Tensors are similar in behavior to NumPy’s arrays.
    

```
import torch

a = torch.Tensor([[1, 2], [3, 4]])
print(a)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1., 2.],
               #         [3., 4.]])           
print(a.dtype) # Prints torch.float32  

print(a**2)    # Prints a torch.FloatTensor of size 2x2 
               # tensor([[ 1.,  4.],
               #         [ 9., 16.]])
```

-   Note that `torch.tensor()` infers the datatype `dtype` automatically, while `torch.Tensor()` always returns a `torch.FloatTensor`.

```
import torch

# Creating a IntTensor
a = torch.tensor([[1, 2], [3, 4]])
print(a)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1., 2.],
               #         [3., 4.]])
print(a.dtype) # Prints torch.int64

# Creating a FloatTensor
a = torch.tensor([[1., 2.], [3., 4.]])
print(a)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1., 2.],
               #         [3., 4.]])
print(a.dtype) # Prints torch.float32 
```

-   `torch.tensor()` supports the `dtype` argument, if you would like to change the type while defining the tensor. Put simply, a tensor of specific data type can be constructed by passing a `torch.dtype` to the constructor or tensor creation op:

```
torch.zeros([2, 4], dtype=torch.int32)                # Prints tensor([[ 0,  0,  0,  0],
                                                      #                [ 0,  0,  0,  0]], dtype=torch.int32)
```

-   Similarly, you can pass in a `torch.device` argument to the constructor:

```
cuda0 = torch.device('cuda:0')
torch.ones([2, 4], dtype=torch.float64, device=cuda0) # Prints tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
                                                      #                [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')
```

### Changing Datatypes

-   With PyTorch, the default float datatype is `float32` (single precision), while that with NumPy is `float64` (double precision). However, the default int datatype for both PyTorch and NumPy is `int64`. You may also change default floating point `dtype` to be `torch.float64` while defining the tensor:

```
import torch

torch.set_default_dtype(torch.float64)
a = torch.Tensor([[1, 2], [3, 4]])
print(a)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1., 2.],
               #         [3., 4.]])           
print(a.dtype) # Prints torch.float64

torch.set_default_dtype(torch.float64)
a = torch.tensor([[1., 2.], [3., 4.]])
print(a)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1., 2.],
               #         [3., 4.]])           
print(a.dtype) # Prints torch.float64
```

-   You may also change the tensor’s datatype after the tensor is defined:

```
import torch

a = torch.tensor([[1, 2], [3, 4]])
print(a)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1, 2],
               #         [3, 4]])           
print(a.dtype) # Prints torch.float32

b = a.double()
print(b)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1., 2.],
               #         [3., 4.]])           
print(b.dtype) # Prints torch.float64

# Same as "b"
c = a.type('torch.DoubleTensor')
print(c)       # Prints a torch.FloatTensor of size 2x2 
               # tensor([[1., 2.],
               #         [3., 4.]])           
print(c.dtype) # Prints torch.float64

d = c.long()
print(d)       # Prints a torch.LongTensor of size 2x2 
               # tensor([[1, 2],
               #         [3, 4]])           
print(d.dtype) # Prints torch.int64

# Same as "d"
e = c.type('torch.LongTensor')
print(e)       # Prints a torch.LongTensor of size 2x2 
               # tensor([[1, 2],
               #         [3, 4]])          
print(e.dtype) # Prints torch.int64
```

### Automatic Differentiation

-   PyTorch Variables allow you to wrap a Tensor and record operations performed on it. This allows you to perform automatic differentiation.

```
import torch
from torch.autograd import Variable

a = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
print(a)            # Prints a torch.FloatTensor of size 2x2 
                    # tensor([[1., 2.],
                    #         [3., 4.]], requires_grad=True)

b = torch.sum(a**2) # 1 + 4 + 9 + 16
print(b)            # Prints a torch.FloatTensor of size 1
                    # tensor(30., grad_fn=<SumBackward0>)


b.backward()        # compute gradients of b wrt a
print(a.grad)       # print db/da_ij = 2*a_ij for a_11, a_12, a21, a22
                    # Prints a torch.FloatTensor of size 2x2
                    # tensor([[2., 4.], 
                    #         [6., 8.]])
```

-   This prelude should give you a sense of the things to come. PyTorch packs elegance and expressiveness in its minimalist and intuitive syntax. Make sure to familiarize yourself with some more examples from the [resources](https://aman.ai/primers/pytorch/#resources) section before moving ahead.
    
-   Thus, if you set `requires_grad` to `True` for any tensor, PyTorch will automatically track and calculate gradients for that tensor. Now, why is this needed? Setting `requires_grad=Tue` tells PyTorch that this parameter should be optimized during the training process using [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), when gradients are used to update weights. This is done with the `tensor.backward()` method; during this operation tensors with `requires_grad=True` will be used along with the tensor used to call `tensor.backward()` to calculate the gradients. As a practical example, using plain `torch.Tensors` (rather than using `torch.autograd.Variable` as above):
    

```
import torch

a = torch.tensor(1.0, requires_grad=True)
x = a ** 3    # x=a^3
b = torch.tensor(1.0, requires_grad=False)
y = b ** 3    # y=b^3
c = torch.tensor(1.0, requires_grad=False)
z = c ** a    # y=b^3

x.backward()  # Computes the gradient 
y.backward()  # Computes the gradient 
z.backward()  # Computes the gradient 

print(a.grad) # this is dx/da; prints tensor(3.)
print(b.grad) # this is dy/db; prints None
print(c.grad) # this is dz/dc; prints None
```

-   Note that `requires_grad` defaults to `False`, unless wrapped in a [`nn.Parameter()`](https://aman.ai/primers/pytorch/#parameters).
-   For more, refer [PyTorch: Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html).
-   The following sections talk about disabling automatic differentiation using `requires_grad=False` to freeze model weights (corresponding to some or all layers or even at the level of the individual weights of layers).

### Disabling Automatic Differentiation

#### Using `requires_grad=False`

```
import torch

a = torch.tensor(1.0, requires_grad=False)
x = a ** 3    # x=a^3

x.backward()  # Computes the gradient 

print(a.grad) # this is dx/da; prints None
```

-   If you want to freeze part of your model and train the rest, you can set `requires_grad` of the parameters you want to freeze to `False`.
-   For example, if you only want to keep the convolutional part of VGG16 fixed:

```
model = torchvision.models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
```

-   By switching the `requires_grad` flags to `False`, no intermediate buffers will be saved, until the computation gets to some point where one of the inputs of the operation requires the gradient.

#### Using `torch.no_grad()`

-   Using the context manager `torch.no_grad()` is a different way to achieve that goal: in the `no_grad` context, all the results of the computations will have `requires_grad=False`, even if the inputs have `requires_grad=True`.
-   Notice that you won’t be able to backpropagate the gradient to layers before the `torch.no_grad`, which are `lin1` and `lin1` in the below example.
    
    ```
      x = torch.randn(2, 2)
      x.requires_grad = True
    
      lin0 = nn.Linear(2, 2)
      lin1 = nn.Linear(2, 2)
      lin2 = nn.Linear(2, 2)
      x1 = lin0(x)
        
      with torch.no_grad():    
          x2 = lin1(x1)
        
      x3 = lin2(x2)
      x3.sum().backward()
        
      print(lin0.weight.grad, lin1.weight.grad, lin2.weight.grad)
    ```
    
    -   which outputs:
    
    ```
      (None, None, tensor([[-1.4481, -1.1789],
               [-1.4481, -1.1789]]))
    ```
    
-   Thus, `lin1.weight.requires_grad` was `True` in the above example, but the gradient wasn’t computed because the operation was done in the `no_grad` context.

-   If your goal is not to finetune, but to set your model in inference mode, the most convenient way is to use the `torch.no_grad` context manager. In this case you also have to set your model to evaluation mode, this is achieved by calling `eval()` on the `nn.Module`, for example:

```
model = torchvision.models.vgg16(pretrained=True)
model.eval()
```

-   This operation sets the attribute `self.training` of the layers to `False`, which changes the behavior of operations like Dropout or BatchNorm that behave differently at training vs. test time.

## Parameters

-   A tensor can be wrapped in a `nn.Parameter()` call to create a parameter for your `nn.module`. Note that the only difference between a `torch.tensor` and a `nn.Parameter` is that an `nn.Parameter` is tracked as a model parameter (and you may list all such parameters for the particular module using `nn.Module.parameters()`).

#### `nn.Parameter` Internals

-   The `nn.Parameter` class is shown in the below code snippet:

![](https://aman.ai/primers/assets/pytorch/parameter.png)

-   Note that since it is sub-classed from `Tensor` it is effectively a `Tensor` (with added features).

#### Difference Between Parameter vs. Tensor in PyTorch

-   Parameters that are declared using `nn.parameter` inside of a module are added to the list of the Module’s parameters. Say, if `m` is your module, `m.parameters()` will book-keep your parameters.
    
-   As an example:
    

```
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2, 2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return x @ self.weights + self.bias

m = M()
m.parameters()
list(m.parameters())
# Prints
# [Parameter containing:
#  tensor([[ 0.1506,  1.2606],
#          [-1.7916, -0.2112]], requires_grad=True), 
#  Parameter containing:
#  tensor([0., 0.], requires_grad=True)]
```

-   `nn.Module.parameters()` returns the parameters that were defined. On the flip side, if we just define a tensor within the class, using say, `self.t = torch.tensor(1)`, it will not show up in the parameters list. That is the only difference between a `torch.tensor` and a `nn.Parameter`.

## Core Training Step

-   Let’s begin with a look at what the heart of our training algorithm looks like. The five lines below pass a batch of inputs through the model, calculate the loss, perform backpropagation and update the parameters.

```
output_batch = model(train_batch)           # compute model output
loss = loss_fn(output_batch, labels_batch)  # calculate loss

optimizer.zero_grad()                       # clear previous gradients (can also be done right after optimizer.step())
loss.backward()                             # compute gradients of loss w.r.t. the inputs and parameters
                        
optimizer.step()                            # perform updates using the calculated gradients
```

-   Each of the variables `train_batch`, `labels_batch`, `output_batch` and `loss` is a PyTorch Variable which allows **derivatives to be automatically calculated**.
    
-   All the other code that we write is built around this – the exact specification of the model, how to fetch a batch of data and labels, computation of the loss and the details of the optimizer.
-   These steps are visually summarized in the image (credits to [Daniel Bourke](https://www.linkedin.com/in/ACoAABNp7YoBNWDK2rsQtsqGcN8H0_NZmHqR0uc?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BCUNHLksGRdu5GmW3t0IOLw%3D%3D)) below:

![](https://aman.ai/primers/assets/pytorch/ptl.jpeg)

-   The PyTorch training loop is a five step process:
    -   **Step 1: The Forward pass.**
        -   Here, the model takes your data, feeds it forward through your network architecture, and comes up with a prediction.
        -   First, put the model in training mode using `model.train()`.
        -   Second, make predictions: `predictions = model(training_data)`.
    -   **Step 2: Calculate the loss.**
        -   Your model will start off making errors.
        -   These errors are the difference between your prediction and the ground truth.
        -   You can calculate this as: `loss = loss_fxn(predictions, ground_truth)`.
    -   **Step 3: Zero gradients.**
        -   You need to zero out the gradients for the optimizer prior to performing back propagation.
        -   If gradients accumulate across iterations, then your model won’t train properly.
        -   You can do this via `optimizer.zero_grad()`.
    -   **Step 4: Backprop.**
        -   Next, you compute the gradient of the loss with respect to model parameter via backprop.
        -   Only parameters with `requires_grad = True` will be updated.
        -   This is where the learning starts to happen.
        -   PyTorch makes it easy, all you do is call: `loss.backward()`.
    -   **Step 5: Update the optimizer (gradient descent).**
        -   Now it’s time to update your trainable parameters so that you can make better predictions.
        -   Remember, trainable means that the parameter has requires\_grad=True.
        -   To update your parameters, all you do is call: `optimizer.step()`.
    -   **Putting it all together:**
        
        ```
          for epoch in range(epochs):
          model.train()
          preds = model(X_train)
          loss = loss_fxn(preds, truth)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        ```
        
-   **Key takeaways**
    -   The training process consists of three major components in the following order: `opt.zero_grad()`, `loss.backward()` and `opt.step()`.
    -   `optimizer.zero_grad()` clears old gradients from the last step (otherwise you’d just accumulate the gradients from all `loss.backward()` calls).
    -   `loss.backward()` computes the gradients of the loss w.r.t. the parameters (or any function requiring gradients) using backpropagation. Note that `loss.backward()` accumulates gradients (by having the `model` keep track of the previously computed gradient) from all previous calls unless cleared using `optimizer.zero_grad()`.
    -   `optimizer.step()` causes the optimizer to take a step based on the gradients of the parameters (which it access through the gradients stored in `model`).
-   Next, we’ll cover how to write a simple model in PyTorch, compute the loss and define an optimizer. The subsequent sections each cover a case of fetching data – one for [image](https://aman.ai/primers/pytorch/#vision-predicting-labels-from-images-of-hand-signs) data and another for [text](https://aman.ai/primers/pytorch/#nlp-named-entity-recognition-ner-tagging) data.

## Models in PyTorch

-   A model can be defined in PyTorch by subclassing the `torch.nn.Module` class. The model is defined using two steps. We first specify the parameters of the model, and then outline how they are applied to the inputs. For operations that do not involve trainable parameters (activation functions such as ReLU, operations like MaxPool), we generally use the `torch.nn.functional` module.
-   For a visual treatment of how to go about creating neural networks in PyTorch, check out [The StatQuest Introduction to PyTorch](https://www.youtube.com/watch?v=FHdlXe1bSe4).
-   Here’s an example of a single hidden layer neural network borrowed from [here](https://github.com/jcjohnson/pytorch-examples#pytorch-custom-nn-modules):

```
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor, we instantiate two nn.Linear modules and assign them as
        member variables.

        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H) 
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function, we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        """
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred
```

-   The `__init__` function initializes the two linear layers of the model. PyTorch takes care of the proper initialization of the parameters you specify. In the `forward` function, we first apply the first linear layer, apply ReLU activation and then apply the second linear layer. The module assumes that the first dimension of `x` is the batch size. If the input to the network is simply a vector of dimension 100, and the batch size is 32, then the dimension of `x` would be (32, 100). Let’s see an example of how to instantiate a model and compute a forward pass:

```
# N is batch size; D_in is input dimension;
# H is the dimension of the hidden layer; D_out is output dimension.
N, D_in, H, D_out = 32, 100, 50, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in)) # dim: 32 x 100

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Forward pass: Compute predicted y by passing x to the model
y_pred = model(x) # dim: 32 x 10
```

-   More complex models follow the same layout, and we’ll see two of them in the subsequent posts.

## Loss Functions

-   PyTorch comes with many standard loss functions available for you to use in the `torch.nn` [module](https://pytorch.org/docs/master/nn.html#loss-functions). From the documentation, here’s a gist of what PyTorch has to offer in terms of loss functions:

| Loss function | Description |
| --- | --- |
| `nn.L1Loss()` | Creates a criterion that measures the mean absolute error (MAE) between each element in the input $x$ and target $y$. |
| `nn.MSELoss()` | Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input $x$ and target $y$. |
| `nn.BCELoss()` | Creates a criterion that measures the Binary Cross Entropy between the target and the output. |
| `nn.BCEWithLogitsLoss()` | This loss combines a Sigmoid layer and the BCELoss in one single class. |
| `nn.CrossEntropyLoss()` | This criterion combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class. |
| `nn.CTCLoss()` | The Connectionist Temporal Classification loss. |
| `nn.NLLLoss()` | The negative log likelihood loss. |
| `nn.PoissonNLLLoss()` | Negative log likelihood loss with Poisson distribution of target. |
| `nn.KLDivLoss()` | The Kullback-Leibler divergence loss measure. |
| `nn.MarginRankingLoss()` | Creates a criterion that measures the loss given inputs $x_1, x_2$, two 1D mini-batch Tensors, and a label 1D mini-batch tensor $y$ (containing 1 or $-1)$. |
| `nn.HingeEmbeddingLoss()` | Measures the loss given an input tensor $x$ and a labels tensor $y$ (containing 1 or -1). |
| `nn.MultiLabelMarginLoss()` | Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input $x$ (a 2D mini-batch Tensor) and output yy (which is a 2D Tensor of target class indices). |
| `nn.SmoothL1Loss()` | Creates a criterion that uses a squared term if the absolute element-wise error falls below 1 and an L1 term otherwise. |
| `nn.SoftMarginLoss()` | Creates a criterion that optimizes a two-class classification logistic loss between input tensor $x$ and target tensor $y$ (containing 1 or $-1$. |
| `nn.MultiLabelSoftMarginLoss()` | Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input $x$ and target $y$ of size $(N, C)$. |
| `nn.CosineEmbeddingLoss()` | Creates a criterion that measures the loss given input tensors $x_1, x_2$ and a Tensor label $y$ with values 1 or $-1$. |
| `nn.MultiMarginLoss()` | Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input $x$ (a 2D mini-batch Tensor) and output $y$ (which is a 1D tensor of target class indices, $0 \leq y \leq \text{x.size}(1)-1$). |
| `nn.TripletMarginLoss()` | Creates a criterion that measures the triplet loss given an input tensors $x_1, x_2, x_3$ and a margin with a value greater than 0. |

-   Full API details are on PyTorch’s `torch.nn` [module](https://pytorch.org/docs/master/nn.html#loss-functions) page.
-   Here’s a simple example of how to calculate Cross Entropy Loss. Let’s say our model solves a multi-class classification problem with $C$ labels. Then for a batch of size $N$, `out` is a PyTorch Variable of dimension $N \times C$ that is obtained by passing an input batch through the model.
-   We also have a `target` Variable of size $N$, where each element is the class for that example, i.e., a label in $\text{[0, …, C-1]}$. You can define the loss function and compute the loss as follows:

```
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(out, target)
```

-   PyTorch makes it very easy to extend this and write your own custom loss function. We can write our own Cross Entropy Loss function as below (note the NumPy-esque syntax):

```
def myCrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]               # batch_size
    outputs = F.log_softmax(outputs, dim=1)      # compute the log of softmax values
    outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
    return -torch.sum(outputs)/num_examples
```

-   Note that when training multi-class classifiers in PyTorch, using `nn.CrossEntropyLoss()`, the input must be an unnormalized raw value (logits), and the target must be a class index instead of one hot encoded vector. On the other hand, when training binary classifiers in PyTorch, make sure to use the correct binary loss for your network structure. `BCELoss()` accepts the output of the sigmoid function. On the other hand, `BCEWithLogitsLoss()` improves numeric stability, but make sure you pass the unnormalized logits as an input because it will apply the sigmoid itself. Passing in the sigmoid output will lead to accidentally applying softmax twice, which many people learn the hard way. While it’s not quite clear why the double application of softmax “kills” training, it is likely due to vanishing gradients. To summarize, for binary classification, PyTorch offers `nn.BCELoss()` and `nn.BCEWithLogitsLoss()`. The former requires the input normalized sigmoid probability, an the latter can take raw unnormalized logits.

![](https://aman.ai/primers/assets/pytorch/BCEWithLogitsLoss.jpg)

-   This was a fairly trivial example of writing our own loss function. In the section on [NLP](https://aman.ai/primers/pytorch/#nlp-named-entity-recognition-ner-tagging), we’ll see an interesting use of **custom loss functions**.

## Optimizers

-   The `torch.optim` [package](https://pytorch.org/docs/master/optim.html) provides an easy to use interface for common optimization algorithms. Torch offers a bunch of in-built optimizers, such as:

-   Full API details are on PyTorch’s `torch.optim` [package](https://pytorch.org/docs/master/optim.html) page.
-   Here’s how you can instantiate your desired optimizer using `torch.optim`:

```
# pick an SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# or pick ADAM
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

-   You pass in the parameters of the model that need to be updated on every iteration. You can also specify more complex methods such as **per-layer** or even **per-parameter** learning rates.
-   Once gradients have been computed using `loss.backward()`, calling `optimizer.step()` updates the parameters as defined by the optimization algorithm.

## Training vs. Evaluation

-   Before training the model, it is imperative to call `model.train()`. Likewise, you must call `model.eval()` before testing the model.
-   This corrects for the differences in **dropout, batch normalization** during **training** and **testing**.

## Computing Metrics

-   By this stage you should be able to understand most of the code in `train.py` and `evaluate.py` (except how we fetch the data, which we’ll come to in the subsequent posts). Apart from keeping an eye on the loss, it is also helpful to monitor other metrics such as **accuracy, precision or recall**. To do this, you can define your own metric functions for a batch of model outputs in the `model/net.py` file.
-   In order to make it easier, we convert the PyTorch Variables into NumPy arrays before passing them into the metric functions.
-   For a multi-class classification problem as set up in the section on [loss functions](https://aman.ai/primers/pytorch/#loss-functions), we can write a function to compute accuracy using NumPy as:

```
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)/float(labels.size)
```

-   You can add your own metrics in the `model/net.py` file. Once you are done, simply add them to the `metrics` dictionary:

```
metrics = { 'accuracy': accuracy,
            # add your own custom metrics,
          }
```

## Saving and Loading Models

-   We define utility functions to save and load models in `utils.py`. To save your model, call:

```
state = {'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optim_dict' : optimizer.state_dict()}
utils.save_checkpoint(state,
                      is_best=is_best,      # True if this is the model with best metrics
                      checkpoint=model_dir) # path to folder
```

-   `utils.py` internally uses the `torch.save(state, filepath)` method to save the state dictionary that is defined above. You can add more items to the dictionary, such as metrics. The `model.state_dict()` stores the parameters of the model and `optimizer.state_dict()` stores the state of the optimizer (such as per-parameter learning rate).
    
-   To load the saved state from a checkpoint, you may use:
    

```
utils.load_checkpoint(restore_path, model, optimizer)
```

-   The `optimizer` argument is optional and you may choose to restart with a new optimizer. `load_checkpoint` internally loads the saved checkpoint and restores the model weights and the state of the optimizer.

## Using the GPU

-   Interspersed through the code, you will find lines such as:

```
model = net.Net(params).cuda() if params.cuda else net.Net(params)

if params.cuda:
    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
```

-   PyTorch makes the use of the GPU explicit and transparent using these commands. Calling `.cuda()` on a model/Tensor/Variable sends it to the GPU. In order to train a model on the GPU, all the relevant parameters and Variables must be sent to the GPU using `.cuda()`.

## Painless Debugging

-   With its clean and minimal design, PyTorch makes debugging a breeze. You can place breakpoints using `import pdb; pdb.set_trace()` at any line in your code. You can then execute further computations, examine the PyTorch Tensors/Variables and pinpoint the root cause of the error.
    
-   That concludes the introduction to the PyTorch code examples. Next, we take upon an example from [vision](https://aman.ai/primers/pytorch/#vision-predicting-labels-from-images-of-hand-signs) and [NLP](https://aman.ai/primers/pytorch/#nlp-named-entity-recognition-ner-tagging) to understand how we load data and define models specific to each domain.
    

### Resources

-   [Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html): an official tutorial from the PyTorch website
-   [ImageNet](https://github.com/pytorch/examples/blob/master/imagenet/main.py): Code for training on ImageNet in PyTorch

## Selected Methods

-   PyTorch provides a host of useful functions for performing computations on arrays. Below, we’ve touched upon some of the most useful ones that you’ll encounter regularly in projects.
-   You can find an exhaustive list of mathematical functions in the [PyTorch documentation](https://pytorch.org/docs/stable/torch.html).

### Tensor Shape/size

-   Unlike \[NumPy\], where `size()` returns the total number of elements in the array across all dimensions, `size()` in PyTorch returns the `shape` of an array.

```
import torch

a = torch.randn(2, 3, 5)

# Get the overall shape of the tensor
a.size()   # Prints torch.Size([2, 3, 5])
a.shape    # Prints torch.Size([2, 3, 5])

# Get the size of a specific axis/dimension of the tensor
a.size(2)  # Prints 5
a.shape[2] # Prints 5
```

### Initialization

-   Presented below are some commonly used initialization functions. A full list can be found on the PyTorch documentation’s [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) page.

#### Static

-   `torch.nn.init.zeros_()` fills the input Tensor with the scalar value 0.
-   `torch.nn.init.ones_()` fills the input Tensor with the scalar value 1.
-   `torch.nn.init.constant_()` fills the input Tensor with the passed in scalar value.

```
import torch.nn as nn

a = torch.empty(3, 5)
nn.init.zeros_(a)         # Initializes a with 0
nn.init.ones_(a)          # Initializes a with 1
nn.init.constant_(a, 0.3) # Initializes a with 0.3
```

### Standard Normal

-   Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

$$
\text{out}_{i} \sim \mathcal{N}(0, 1)
$$

```
import torch

torch.randn(4)    # Returns 4 values from the standard normal distribution
torch.randn(2, 3) # Returns a 2x3 matrix sampled from the standard normal distribution
```

#### Xavier/Glorot

##### Uniform

-   Fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feed-forward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution. The resulting tensor will have values sampled from $\mathcal{U}(-a, a)$ where,

$$
a = \text{gain} \times \sqrt{\frac{6}{\text{fan-in} + \text{fan-out}}}
$$

-   Also known as Glorot initialization.

```
import torch.nn as nn

a = torch.empty(3, 5)
nn.init.xavier_uniform_(a, gain=nn.init.calculate_gain('relu')) # Initializes a with the Xavier uniform method
```

##### Normal

-   Fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feed-forward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution. The resulting tensor will have values sampled from $\mathcal{N}(0, \text{std}^2)$ where,

$$
\text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan-in} + \text{fan-out}}}
$$

-   Also known as Glorot initialization.

```
import torch.nn as nn

a = torch.empty(3, 5)
nn.init.xavier_normal_(a) # Initializes a with the Xavier normal method
```

#### Kaiming/He

##### Uniform

-   Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a uniform distribution. The resulting tensor will have values sampled from $\mathcal{U}(-\text{bound}, \text{bound})$ where,

$$
\text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan-mode}}}
$$

-   Also known as He initialization.

```
import torch.nn as nn

a = torch.empty(3, 5)
nn.init.kaiming_uniform_(a, mode='fan_in', nonlinearity='relu') # Initializes a with the Kaiming uniform method 
```

##### Normal

-   Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a normal distribution. The resulting tensor will have values sampled from $\mathcal{N}(0, \text{std}^2)$ where,

$$
\operatorname{std}=\frac{\text { gain }}{\sqrt{\text { fan-mode }}}
$$ 

-   Also known as He initialization.

```
import torch.nn as nn

a = torch.empty(3, 5)
nn.init.kaiming_normal_(a, mode='fan_out', nonlinearity='relu') # Initializes a with the Kaiming uniform method 
```

### Send Tensor to GPU

-   To send a tensor (or model) to the GPU, you may use `tensor.cuda()` or `tensor.to(device)`:

```
import torch

t = torch.tensor([1, 2, 3])
a = t.cuda()
type(a) # Prints <class 'numpy.ndarray'>

# Send tensor to the GPU
a = a.cuda()

# Bring the tensor back to the CPU
a = a.cpu()
```

-   Note that there is no difference between the two. Early versions of PyTorch had `tensor.cuda()` and `tensor.cpu()` methods to move tensors and models from CPU to GPU and back. However, this made code writing a bit cumbersome:

```
if cuda_available:
    x = x.cuda()
    model.cuda()
else:
    x = x.cpu()
    model.cpu()
```

-   Later versions of PyTorch introduced `tensor.to()` that basically takes care of everything in an elegant way:

```
device = torch.device('cuda') if cuda_available else torch.device('cpu')
x = x.to(device)
model = model.to(device)
```

### Convert to NumPy

-   Both in PyTorch and TensorFlow, the `tensor.numpy()` method is pretty much straightforward. It converts a tensor object into an `numpy.ndarray` object. This implicitly means that the converted tensor will be now [processed on the CPU](https://stackoverflow.com/questions/63968868/what-does-the-numpy-function-do#:~:text=the%20CPU.%20Here%27s%20a-,relevant%20docstring,-supporting%20this%20statement).

```
import torch

t = torch.tensor([1, 2, 3])
a = t.numpy()               # array([1, 2, 3])
type(a)                     # Prints <class 'numpy.ndarray'>

# Send tensor to the GPU.
t = t.cuda()

b = t.cpu().numpy()          # array([1, 2, 3])
type(b)                      # <class 'numpy.ndarray'>
```

-   If you originally created a PyTorch Tensor with `requires_grad=True` (note that `requires_grad` defaults to `False`, unless wrapped in a `nn.Parameter()`), you’ll have to use `detach()` to get rid of the gradients when sending it downstream for say, post-processing with NumPy, or plotting with Matplotlib/Seaborn. Calling `detach()` before `cpu()` prevents [superfluous gradient copying](https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489/5). This greatly optimizes runtime. Note that `detach()` is not necessary if `requires_grad` is set to `False` when defining the tensor.

```
import torch

t = torch.tensor([1, 2, 3], requires_grad=True)
a = t.detach().numpy()       # array([1, 2, 3])
type(a)                      # Prints <class 'numpy.ndarray'>

# Send tensor to the GPU.
t = t.cuda()

# The output of the line below is a NumPy array.
b = t.detach().cpu().numpy() # array([1, 2, 3])
type(b)                      # <class 'numpy.ndarray'>
```

### `tensor.item()`: Convert Single Value Tensor to Scalar

-   Returns the value of a tensor as a Python int/float. This only works for tensors with one element. For other cases, see `[tolist()](#tensortolist-convert-multi-value-tensor-to-scalar)`.
-   Note that this operation is not differentiable.

```
import torch

a = torch.tensor([1.0])
a.item()   # Prints 1.0

a.tolist() # Prints [1.0]
```

### `tensor.tolist()`: Convert Multi Value Tensor to Scalar

-   Returns the tensor as a (nested) list. For scalars, a standard Python number is returned, just like with `[item()](#tensoritem-convert-single-value-tensor-to-scalar)`. Tensors are automatically moved to the CPU first if necessary.
-   Note that this operation is not differentiable.

```
a = torch.randn(2, 2)
a.tolist()      # Prints [[0.012766935862600803, 0.5415473580360413],
                #         [-0.08909505605697632, 0.7729271650314331]]
a[0,0].tolist() # Prints 0.012766935862600803
```

### Len

-   `len()` returns the size of the first dimension of the input tensor, similar to NumPy.

```
import torch

a = torch.Tensor([[1, 2], [3, 4]])
print(a) # Prints tensor([[1., 2.],
         #                [3., 4.]])
len(a)   # 2

b = torch.Tensor([1, 2, 3, 4])
print(b) # Prints tensor([1., 2., 3., 4.])
len(b)   # 4
```

### Arange

-   Return evenly spaced values within the half-open interval $[start, stop)$ (in other words, the interval including start but excluding stop).
-   For integer arguments the function is equivalent to the Python built-in `range` function, but returns an `tensor` rather than a list.

```
import torch

print(torch.arange(8))             # Prints tensor([0 1 2 3 4 5 6 7])
print(torch.arange(3, 8))          # Prints tensor([3 4 5 6 7])
print(torch.arange(3, 8, 2))       # Prints tensor([3 5 7])

# arange() works with floats too (but read the disclaimer below)
print(torch.arange(0.1, 0.5, 0.1)) # Prints tensor([0.1000, 0.2000, 0.3000, 0.4000])
```

-   When using a non-integer step, such as $0.1$, the results will often not be consistent. It is better to use `torch.linspace()` for those cases as below.

### Linspace

-   Return evenly spaced numbers calculated over the interval $[start, stop]$.
-   Starting PyTorch 1.11, `linspace` requires the `steps` argument. Use `steps=100` to restore the previous behavior.

```
import torch

print(torch.linspace(1.0, 2.0, steps=5)) # Prints tensor([1.0000, 1.2500, 1.5000, 1.7500, 2.0000])
```

### View

-   Returns a new tensor with the same data as the input tensor but of a different shape.
-   For a tensor to be viewed, the following conditions must be satisfied:
    -   The new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions.
    -   `view()` can be only be performed on contiguous tensors (which can be ascertained using `is_contiguous()`). Otherwise, a contiguous copy of the tensor (e.g., via `contiguous()`) needs to be used. When it is unclear whether a `view()` can be performed, it is advisable to use (`reshape()`)\[#reshape\], which returns a view if the shapes are compatible, and copies the tensor (equivalent to calling contiguous()) otherwise.

```
import torch

a = torch.arange(4).view(2, 2)

print(a.view(4, 1)) # Prints tensor([[0],
                    #                [1],
                    #                [2],
                    #                [3]])

print(a.view(1, 4)) # Prints tensor([[0, 1, 2, 3]])
```

-   Passing in a `-1` to `torch.view()` returns a flattened version of the array.

```
import torch

a = torch.arange(4).view(2, 2)
print(a.view(-1)) # Prints tensor([0, 1, 2, 3])
```

-   The view tensor shares the same underlying data storage with its base tensor. No data movement occurs when creating a view, view tensor just changes the way it interprets the same data. This avoids explicit data copy, thus allowing fast and memory efficient reshaping, slicing and element-wise operations.

```
import torch

a = torch.rand(4, 4)
b = a.view(2, 8)
a.storage().data_ptr() == b.storage().data_ptr() # Prints True since `a` and `b` share the same underlying data.
```

-   Note that modifying the view tensor changes the input tensor as well.

```
import torch

a = torch.rand(4, 4)
b = a.view(2, 8)
b[0][0] = 3.14

print(t[0][0]) # Prints tensor(3.14)
```

### Transpose

-   Returns a tensor that is a transposed version of input for 2D tensors. More generally, interchanges two axes of an array. In other words, the given dimensions `dim0` and `dim1` are swapped.
-   The resulting out tensor shares its underlying storage with the input tensor, so changing the content of one would change the content of the other.

```
import torch

a = torch.randn(2, 3, 5)
a.size()                 # Prints torch.Size([2, 3, 5])

a.transpose(0, -1).shape # Prints torch.Size([5, 3, 2])
```

#### Swapaxes

-   Alias for [`torch.transpose()`](https://aman.ai/primers/pytorch/#transpose). This function is equivalent to NumPy’s (`swapaxes`)\[../numpy/#swapaxes\] function.

```
import torch

a = torch.randn(2, 3, 5)
a.size()                # Prints torch.Size([2, 3, 5])

a.swapdims(0, -1).shape # Prints torch.Size([5, 3, 2])

# swapaxes is an alias of swapdims
a.swapaxes(0, -1).shape # Prints torch.Size([5, 3, 2])
```

### Permute

-   Returns a view of the input tensor with its axes ordered as indicated in the input argument.

```
import torch

a = torch.randn(2, 3, 5)
a.size()                  # Prints torch.Size([2, 3, 5])

a.permute(2, 0, 1).size() # Prints torch.Size([5, 2, 3])
```

-   Note that (i) using `view` or `reshape` to restructure the array, and (ii) `permute` or `transpose` to swap axes, can render the same output shape but does not necessarily yield the same tensor in both cases.

```
a = torch.tensor([[1, 2, 3], [4, 5, 6]])

viewed = a.view(3, 2)
perm = a.permute(1, 0)

viewed.shape   # Prints torch.Size([3, 2])
perm.shape     # Prints torch.Size([3, 2])

viewed == perm # Prints tensor([[ True, False],
               #                [False, False],
               #                [False,  True]])

viewed         # Prints tensor([[1, 2],
               #                [3, 4],
               #                [5, 6]])

perm           # Prints tensor([[1, 4],
               #                [2, 5],
               #                [3, 6]])
```

### Movedim

-   Compared to `torch.permute()` for reordering axes which needs positions of all axes to be explicitly specified, moving one axis while keeping the relative positions of all others is a common enough use-case to warrant its own syntactic sugar. This is the functionality that is offered by `torch.movedim()`.

```
import torch

a = torch.randn(2, 3, 5)
a.size()                # Prints torch.Size([2, 3, 5])

a.movedim(0, -1).shape  # Prints torch.Size([3, 5, 2])

# moveaxis is an alias of movedim
a.moveaxis(0, -1).shape # Prints torch.Size([3, 5, 2])
```

### Randperm

-   Returns a random permutation of integers from `0` to `n - 1`.

```
import torch

torch.randperm(n=4) # Prints tensor([2, 1, 0, 3])
```

-   As a practical use-case, `torch.randperm()` helps select mini-batches containing data samples randomly as follows:

```
data[torch.randperm(data.shape[0])] # Assuming the first dimension of data is the minibatch number
```

### Where

-   Returns a tensor of elements selected from either `a` or `b`, depending on the outcome of the specified condition.
-   The operation is defined as:

$$
\text{out}_i = \begin{cases} \text{x}_i & \text{if } \text{condition}_i \\ \text{y}_i & \text{otherwise} \\ \end{cases}
$$

```
import torch

a = torch.randn(3, 2) # Initializes a as a 3x2 matrix using the the standard normal distribution
b = torch.ones(3, 2)
>>> a
tensor([[-0.4620,  0.3139],
        [ 0.3898, -0.7197],
        [ 0.0478, -0.1657]])
>>> torch.where(a > 0, a, b)
tensor([[ 1.0000,  0.3139],
        [ 0.3898,  1.0000],
        [ 0.0478,  1.0000]])
>>> a = torch.randn(2, 2, dtype=torch.double)
>>> a
tensor([[ 1.0779,  0.0383],
        [-0.8785, -1.1089]], dtype=torch.float64)
>>> torch.where(a > 0, a, 0.)
tensor([[1.0779, 0.0383],
        [0.0000, 0.0000]], dtype=torch.float64)
```

### Reshape

-   Returns a tensor with the same data and number of elements as the input, but with the specified shape. When possible, the returned tensor will be a view of the input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior. It means that `torch.reshape` may return a copy or a view of the original tensor.
-   A single dimension may be `-1`, in which case it’s inferred from the remaining dimensions and the number of elements in input.

```
import torch

a = torch.arange(4*10*2).view(4, 10, 2)
b = x.permute(2, 0, 1)

# Reshape works on non-contiguous tensors (contiguous() + view())
print(b.is_contiguous())
try: 
    print(b.view(-1))
except RuntimeError as e:
    print(e)
print(b.reshape(-1))
print(b.contiguous().view(-1))
```

-   While `torch.view()` has existed for a long time, `torch.reshape()` has been [recently introduced in PyTorch 0.4](https://github.com/pytorch/pytorch/pull/5575). When it is unclear whether a `view()` can be performed, it is advisable to use `reshape()`, which returns a view if the shapes are compatible, and copies (equivalent to calling `contiguous()`) otherwise.

### Concatenate

-   Concatenates the input sequence of tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.

```
import torch

x = torch.randn(2, 3)
print(x) # Prints a 2x3 matrix: [[ 0.6580, -1.0969, -0.4614],
         #                       [-0.1034, -0.5790,  0.1497]]

print(torch.cat((x, x, x), 0)) # Prints a 6x3 matrix: [[ 0.6580, -1.0969, -0.4614],
                               #                       [-0.1034, -0.5790,  0.1497],
                               #                       [ 0.6580, -1.0969, -0.4614],
                               #                       [-0.1034, -0.5790,  0.1497],
                               #                       [ 0.6580, -1.0969, -0.4614],
                               #                       [-0.1034, -0.5790,  0.1497]]

print(torch.cat((x, x, x), 1)) # Prints a 2x9 matrix: [[ 0.6580, -1.0969, -0.4614,  
                               #                         0.6580, -1.0969, -0.4614,  
                               #                         0.6580, -1.0969, -0.4614],
                               #                       [-0.1034, -0.5790,  0.1497, 
                               #                        -0.1034, -0.5790,  0.1497, 
                               #                        -0.1034, -0.5790,  0.1497]]
```

### Squeeze

-   Similar to NumPy’s [`np.squeeze()`](https://aman.ai/primers/numpy/#squeeze), `torch.squeeze()` removes all dimensions with size one from the input tensor. The returned tensor shares the same underlying data with this tensor.
    
-   For example, if the input is of shape: $(A \times 1 \times B \times C \times 1 \times D)$ then the output tensor will be of shape: $(A \times B \times C \times D)$.
    
-   When an optional `dim` argument is given to `torch.squeeze()`, a squeeze operation is done only in the given dimension. If the input is of shape: $(A \times 1 \times B)$, `torch.squeeze(input, 0)` leaves the tensor unchanged, but `torch.squeeze(input, 1)` will squeeze the tensor to the shape $(A \times B)$.
    
-   An important bit to note is that if the tensor has a batch dimension of size 1, then `torch.squeeze()` will also remove the batch dimension, which can lead to unexpected errors.
    
-   Here is a visual representation of what `torch.squeeze()` and [`torch.unsqueeze()`](https://aman.ai/primers/pytorch/#unsqueeze) do for a 2D matrix:
    

![](https://aman.ai/primers/assets/pytorch/sq_unsq.png)

```
import torch

a = torch.zeros(2, 1, 2, 1, 2)
print(a.size()) # Prints torch.Size([2, 1, 2, 1, 2])

b = torch.squeeze(a)
print(b.size()) # Prints torch.Size([2, 2, 2])

b = torch.squeeze(a, 0)
print(b.size()) # Prints torch.Size([2, 1, 2, 1, 2])

b = torch.squeeze(a, 1)
print(b.size()) # Prints torch.Size([2, 2, 1, 2])
```

### Unsqueeze

-   `torch.unsqueeze()` is the opposite of [`torch.squeeze()`](https://aman.ai/primers/pytorch/#squeeze). It inserts a dimension of size one at the specified position. The returned tensor shares the same underlying data with this tensor.
    
-   A `dim` argument within the range `[-input.dim() - 1, input.dim() + 1)` can be used. A negative value of `dim` will correspond to `torch.unsqueeze()` applied at `dim = dim + input.dim() + 1`.
    

```
import torch

a = torch.tensor([1, 2, 3, 4])
print(x.size()) # Prints torch.Size([4])

b = torch.unsqueeze(a, 0) 
print(b)        # Prints tensor([[1, 2, 3, 4]])
print(b.size()) # Prints torch.Size([1, 4])

b = torch.unsqueeze(a, 1)
print(b)        # Prints tensor([[1],
                #                [2],
                #                [3],
                #                [4]])
print(b.size()) # torch.Size([4, 1])
```

-   Note that unlike [`torch.squeeze()`](https://aman.ai/primers/pytorch/#squeeze), the `dim` argument is required (and not optional) with `torch.unsqueeze()`.
    
-   A practical use-case of `torch.unsqueeze()` is to add an additional dimension (usually the first dimension) for the batch number as shown in the example below:
    

```
import torch

# 3 channels, 32 width, 32 height
a = torch.randn(3, 32, 32)

# 1 batch, 3 channels, 32 width, 32 height
a.unsqueeze(dim=0).shape
```

### Print Model Summary

-   Printing the model prints a summary of the model including the different **layers** involved and their **specifications**.

```
from torchvision import models
model = models.vgg16()
print(model)
```

-   The output in this case would be something as follows:

```
VGG (
  (features): Sequential (
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU (inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU (inplace)
    (4): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU (inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU (inplace)
    (9): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU (inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU (inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU (inplace)
    (16): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU (inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU (inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU (inplace)
    (23): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU (inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU (inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU (inplace)
    (30): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (classifier): Sequential (
    (0): Dropout (p = 0.5)
    (1): Linear (25088 -> 4096)
    (2): ReLU (inplace)
    (3): Dropout (p = 0.5)
    (4): Linear (4096 -> 4096)
    (5): ReLU (inplace)
    (6): Linear (4096 -> 1000)
  )
)
```

-   To get the representation [tf.keras](https://aman.ai/primers/tensorflow/#model-summary) offers, use the `pytorch-summary` package. This contains a lot more details of the model, including:
    -   Name and type of all layers in the model.
    -   Output shape for each layer.
    -   Number of weight parameters of each layer.
    -   The total number of trainable and non-trainable parameters of the model.
    -   In addition, also offers the following bits not in the Keras summary:
        -   Input size (MB)
        -   Forward/backward pass size (MB)
        -   Params size (MB)
        -   Estimated Total Size (MB)

```
from torchvision import models
from torchsummary import summary

# Example for VGG16
vgg = models.vgg16()
summary(vgg, (3, 224, 224))
```

-   The output in this case would be something as follows:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         590,080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       1,180,160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]       2,359,808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]       2,359,808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]       2,359,808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]       2,359,808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]       2,359,808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
           Linear-32                 [-1, 4096]     102,764,544
             ReLU-33                 [-1, 4096]               0
          Dropout-34                 [-1, 4096]               0
           Linear-35                 [-1, 4096]      16,781,312
             ReLU-36                 [-1, 4096]               0
          Dropout-37                 [-1, 4096]               0
           Linear-38                 [-1, 1000]       4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.59
Params size (MB): 527.79
Estimated Total Size (MB): 746.96
----------------------------------------------------------------
```

### Resources

-   [Generating Names](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#sphx-glr-intermediate-char-rnn-generation-tutorial-py): a tutorial on character-level RNN.
-   [Sequence to Sequence models](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py): a tutorial on translation.
-   [PyTorch: Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

## References

-   CS230 class notes from Spring 2019.
-   [PyTorch documentation: `torch.optim`](https://pytorch.org/docs/master/optim.html).
-   [Use GPU in your PyTorch code](https://medium.com/ai%C2%B3-theory-practice-business/use-gpu-in-your-pytorch-code-676a67faed09)
-   [Difference between torch.tensor and torch.Tensor](https://discuss.pytorch.org/t/difference-between-torch-tensor-and-torch-tensor/30786)
-   [Why we need image.to(‘CUDA’) when we have model.to(‘CUDA’)?](https://stackoverflow.com/questions/53695105/why-we-need-image-tocuda-when-we-have-model-tocuda)
-   [Is there any difference between x.to(‘cuda’) vs x.cuda()? Which one should I use?](https://discuss.pytorch.org/t/is-there-any-difference-between-x-to-cuda-vs-x-cuda-which-one-should-i-use/20137/2)
-   [What `step()`, `backward()`, and `zero_grad()` do?](https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301)
-   [How much faster is NCHW compared to NHWC in TensorFlow/cuDNN?](https://stackoverflow.com/questions/44280335/how-much-faster-is-nchw-compared-to-nhwc-in-tensorflow-cudnn)
-   [How do I check the number of parameters of a model?](https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9)
-   [How do you determine the layer type?](https://discuss.pytorch.org/t/how-do-you-determine-the-layer-type/19309)
-   [PyTorch squeeze and unsqueeze](https://stackoverflow.com/questions/61598771/pytorch-squeeze-and-unsqueeze)
-   [What does “unsqueeze” do in PyTorch?](https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch)
-   [Model summary in PyTorch](https://stackoverflow.com/questions/42480111/model-summary-in-pytorch)
-   [Difference between view, reshape, transpose and permute in PyTorch](https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/)
-   [Difference between tensor.permute and tensor.view in PyTorch?](https://stackoverflow.com/questions/51143206/difference-between-tensor-permute-and-tensor-view-in-pytorch)
-   [PyTorch: .movedim() vs. .moveaxis() vs. .permute()](https://stackoverflow.com/questions/68041894/pytorch-movedim-vs-moveaxis-vs-permute)
-   [PyTorch: What is the difference between tensor.cuda() and tensor.to(torch.device(“cuda:0”))?](https://stackoverflow.com/questions/62907815/pytorch-what-is-the-difference-between-tensor-cuda-and-tensor-totorch-device)
-   [PyTorch tensor to numpy array](https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array)
-   [PyTorch: how to set requires\_grad=False](https://stackoverflow.com/questions/51748138/pytorch-how-to-set-requires-grad-false)
-   [Difference between Parameter vs. Tensor in PyTorch](https://stackoverflow.com/questions/56708367/difference-between-parameter-vs-tensor-in-pytorch)
-   [What is the use of requires\_grad in Tensors?](https://jovian.ai/forum/t/what-is-the-use-of-requires-grad-in-tensors/17718)

## Citation

This tutorial was partially adopted from:
```
Chadha, A. (2020). PyTorch Primer. Distilled AI. https://aman.ai
```
