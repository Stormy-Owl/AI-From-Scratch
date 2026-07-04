# Neural Networks From Scratch

A collection of neural network implementations written entirely in pure
Python without using NumPy or any deep learning frameworks. Every
stage---from forward propagation to backpropagation and
optimization---is implemented manually to understand the mathematics
behind neural networks.

## Projects

### 1. Feedforward Neural Network (`RNN.py`)

> **Note:** Despite the filename, this is a fully connected feedforward
> neural network (MLP), not a recurrent neural network.

**Features** - 784 → 16 → 16 → 10 architecture - Tanh activation -
Softmax output layer - Cross-Entropy loss - Manual backpropagation -
Mini-batch gradient descent

------------------------------------------------------------------------

### 2. Adam Optimizer (`Adam.py`)

Implements the Adam optimization algorithm from scratch.

**Features** - First and second moment estimation - Bias correction -
Adaptive learning rate - Drop-in replacement for standard gradient
descent

------------------------------------------------------------------------

### 3. Convolutional Neural Network (`CNN.py`)

A CNN built completely from scratch for MNIST digit classification.

#### Architecture

``` text
28×28 Image
      │
Conv (16 Filters)
      │
ReLU
      │
Max Pool
      │
Conv (16 Filters)
      │
ReLU
      │
Max Pool
      │
Flatten
      │
Dense (10)
      │
Softmax
```

**Features** - Manual convolution - Zero padding - ReLU activation - Max
pooling - Flatten layer - Dense layer - Softmax classifier -
Cross-Entropy loss - Manual CNN backpropagation

------------------------------------------------------------------------

## Dataset Structure

``` text
train/
├── 0/
├── 1/
├── ...
└── 9/

test/
├── 0/
├── 1/
├── ...
└── 9/
```

Each folder contains grayscale images belonging to its respective digit
class.

------------------------------------------------------------------------

## Weight Initialization

Before training, initialize:

``` text
weights.json
bias.json
CNN_weights.json
CNN_Filters.json
```

These files store all learnable parameters and are updated during
training.

------------------------------------------------------------------------

## Training

Feedforward Network

``` python
batch_training(batch_size)
```

CNN

``` python
train(batch_size)
```

Adam Optimizer

``` python
Adam(batch_grad, iteration)
```

------------------------------------------------------------------------

## Testing

Feedforward Network

``` python
test()
```

CNN

``` python
forward(image_path)
```

------------------------------------------------------------------------

## Libraries Used

-   Pillow
-   math
-   json
-   random
-   os

No NumPy, TensorFlow, PyTorch, or other machine learning libraries are
used.

------------------------------------------------------------------------

## Purpose

The objective of this project is educational: to understand how neural
networks operate internally by implementing every major component
manually, including forward propagation, backpropagation, convolution,
pooling, activation functions, loss computation, and optimization
without relying on existing deep learning frameworks.
