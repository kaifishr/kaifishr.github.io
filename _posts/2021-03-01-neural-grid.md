---
layout: post
title: "NeuralGrid: A Grid-based Neural Network"
date:   2021-03-01 21:30:21
---

**TL;DR**: Grid-based neural network architecture allow neurons to form arbitrary signal processing structures during training.

---

1. TOC
{:toc}

## Introduction 

This post presents a grid-based neural network architecture which I will call NeuralGrid. This network architecture is designed to allow atificial neurons to automatically form arbitary interconnections during the training process, thereby forming a previously undefined network structure. Compared to conventional feedforward network architectures, this approach allows the grid component of the network to be more malleable. The following figure shows the basic idea of a NeuralGrid.

<p align="center"> 
<img src="/assets/images/post10/neural_grid.png" width="800"> 
<b>Figure 1:</b> Basic idea of a grid-based neural network architecture.
</p>
{: #fig:neuralgrid}

The image above shows the basic idea behand a grid-based neural network architecture. In a first step, an input $x$ is mapped to the grid's input dimension where the representation is then processed from left to right using a simple local mapping between adjacent grid layers. The signal at the grid's output is mapped to the network's output using again a fully connected layer.

The goal of this project is not to set a new SOTA for some dataset, but rather to investigate qualitatively the emerging patterns within the neural grid and how well the network performs.

## Method

The architecture of a two-dimensional grid-based neural network consists of three parts: The mapping from the input neurons $x_m$ to the grid's first layer, the mapping inside the grid, and finally the mapping from the last grid layer to the network's output. In case of a three-dimensional grid, the mapping to the grid's input can be skipped.

### Calculations outside the Grid

As has probably been noticed, the network's information processing can be split into two different types. The first type of operation, from the network's input to the grid's input, and from the grid's output to the network's output consists of a standard vector-matrix multiplication represented by fully connected layers. The information processing of a single neuron outside the grid, i.e. before and after the grid, can be described as an affine linear transformation $z$ that is follows by some nonlinear mapping represented by $h$.

From network input to grid:

\begin{equation} \label{eq:preactivation_in}
    z_l = \sum_{m} w_{lm} x_m + b_l
\end{equation}

\begin{equation} \label{eq:activation_in}
    x_m = h(z_l)
\end{equation}

From grid to network output:

\begin{equation} \label{eq:preactivation_out}
    z_i = \sum_{j} w_{ij} x_j + b_i 
\end{equation}

\begin{equation} \label{eq:activation_out}
    x_i = h(z_i)
\end{equation}

### Calculations inside the Grid

Things look a bit different for the mapping inside the grid. Here, the mapping can be described as follows:

\begin{equation} \label{eq:zgrid}
    z_p = \sum_{q \subset \Omega_p} w_{q} x_{q} + b_p
\end{equation}

\begin{equation} \label{eq:xgrid}
    x_p = h(z_p)
\end{equation}

In Equation \eqref{eq:zgrid}, $\Omega_p$ represents the receptive field over which the weighted sum is calculated. This is shown for the case of a two-dimensional grid in [Figure 1](#fig:neuralgrid) with a receptive field size (i.e. kernel size) of 3 and stride 1. As can be seen in [Figure 1](#fig:neuralgrid), the weighted sum is computed over three grid cells each. For a three-dimensional grid, the receptive field would be of size $k \times k$. It is also interesting to note, that due to the neural grid's architecture and the associated processing scheme, there are the same number of biases as there are weights.

Weights within the grid are shared by several neurons due to overlapping receptive fields. Thus, as [Figure 1](#fig:neuralgrid) shows, each weight within the grid contributes to three neurons in the consecutive layer. 

This must also be taken into account in the gradient computation insofar as the error signals for the respective weights must be accumulated. Fortunately, however, PyTorch's autograd engine takes care of this for us.

#### Formulating Grid Processes with Unfolding Operations

There is no out-of-box functionality within the PyTorch library that can perform the processes inside the grid as described above. One possibility would be to create the individual grid cells as a tensor and iterate over the grid with two for-loops. However, this would be incredibly slow (see comparison further below). Luckily, the mapping within the grid can also be represented as a special kind of vector-matrix multiplication, where the matrix is a sparse matrix with shared weights. This allows us to use PyTorch's blazingly fast unfolding operations which results in a massive speedup compared to a naive loop-based implementation. 

The following two examples show how the mapping between two consecutive grid layers can be formulated using unfolding operation for the case of a two- and three-dimensional grid.

#### Unfolding in a Two Dimensional Neural Grid

For a neural grid layer consisting of activations $\boldsymbol{x}$ and weights $\boldsymbol{w}$ each consisting of five elements, a kernel size of $k=3$, and stride 1, Equation \eqref{eq:zgrid} can be written as

$$
\label{eq:2dNeuralGridXW}
\boldsymbol{x} = 
\begin{pmatrix}
x_{1}\\
x_{2}\\
x_{3}\\
x_{4}\\
x_{5}
\end{pmatrix},
\boldsymbol{w} = 
\begin{pmatrix}
w_{1}\\
w_{2}\\
w_{3}\\
w_{4}\\
w_{5}
\end{pmatrix}
$$

$$
\label{eq:2dNeuralGridZ}
\boldsymbol{z} = 
\begin{pmatrix}
x_{1}\\
x_{2}\\
x_{3}\\
x_{4}\\
x_{5}
\end{pmatrix}
*
\begin{pmatrix}
w_{1}\\
w_{2}\\
w_{3}\\
w_{4}\\
w_{5}
\end{pmatrix}
+
\begin{pmatrix}
b_{1}\\
b_{2}\\
b_{3}\\
b_{4}\\
b_{5}
\end{pmatrix}
$$

$$
\label{eq:sparseWeightMatrix}
z = 
\begin{pmatrix}
w_{1} & w_{2} & 0 & 0 & 0\\
w_{1} & w_{2} & w_{3} & 0 & 0\\
0 & w_{2} & w_{3} & w_{4} & 0\\
0 & 0 & w_{3} & w_{4} & w_{5}\\
0 & 0 & 0 & w_{4} & w_{5}
\end{pmatrix}
\cdot
\begin{pmatrix}
x_{1}\\
x_{2}\\ x_{3}\\
x_{4}\\
x_{5}
\end{pmatrix}
+
\begin{pmatrix}
b_{1}\\
b_{2}\\
b_{3}\\
b_{4}\\
b_{5}
\end{pmatrix}
$$

In Equation \eqref{eq:sparseWeightMatrix} we see that every weight, except for the weights at the grid's border, is involved in three operations, meaning that during the backpropagation step, these weights receive three error signals / gradients. It can also be seen that every weights final gradient can be obtained by summing along the vertical axis of the weight's matrix corresponding gradient matrix.

However, in terms of memory usage and computational efficiency, Equation \eqref{eq:sparseWeightMatrix} is not very elegant and can be reformulated into a more compact representation

$$
\label{eq:unfoldedFormulation}
z = 
\begin{pmatrix}
\begin{pmatrix}
0 & w_{1} & w_{2}\\
w_{1} & w_{2} & w_{3}\\
w_{2} & w_{3} & w_{4}\\
w_{3} & w_{4} & w_{5}\\
w_{4} & w_{5} & 0
\end{pmatrix}
\odot
\begin{pmatrix}
0 & x_{1} & x_{2}\\
x_{1} & x_{2} & x_{3}\\
x_{2} & x_{3} & x_{4}\\
x_{3} & x_{4} & x_{5}\\
x_{4} & x_{5} & 0
\end{pmatrix}
\end{pmatrix}
\cdot
\begin{pmatrix}
1\\
1\\
1
\end{pmatrix}
+
\begin{pmatrix}
b_{1}\\
b_{2}\\
b_{3}\\
b_{4}\\
b_{5}
\end{pmatrix}
$$

where $\odot$ represents the element-wise product of weights and activations. At first glance, the reformulation of Equation \eqref{eq:unfoldedFormulation} does not necessarily look more compact. However, this formulation scales with $\mathcal{O}(k \cdot n)$ and is therefore computationally more efficient compared to the first formulation in Equation \eqref{eq:unfoldedFormulation} that scales with $\mathcal{O}(n^2)$. In the implementation we can express Equation \eqref{eq:unfoldedFormulation} using PyTorch's unfolding operations for blazingly fast computations within the neural grid.

#### Unfolding in a Three Dimensional Neural Grid

We can do the same for the three-dimensional case which is a bit trickier. Unlike in the case of standard 2D convolutions as implemented in standard machine learning libraries, weights are not shared accross the input (or feature map). For an image with dimensions $H \times W$ we therefore also have the same number of weights. For an input image of size $3 \times 3$ and same padding with zeros, the weight matrix of a neural grid layer looks as follows:

$$
\label{eq:Weights3d}
z =
\begin{pmatrix}
w_{1} & w_{2} & w_{3} \\
w_{4} & w_{5} & w_{6} \\
w_{7} & w_{8} & w_{9} \\
\end{pmatrix}
*
\begin{pmatrix}
x_{1} & x_{2} & x_{3} \\
x_{4} & x_{5} & x_{6} \\
x_{7} & x_{8} & x_{9} \\
\end{pmatrix}
$$


$$
\label{eq:Weights3d2}
z =
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 \\
0 & w_{1} & w_{2} & w_{3} & 0 \\
0 & w_{4} & w_{5} & w_{6} & 0 \\
0 & w_{7} & w_{8} & w_{9} & 0 \\
0 & 0 & 0 & 0 & 0 \\
\end{pmatrix}
*
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 \\
0 & x_{1} & x_{2} & x_{3} & 0 \\
0 & x_{4} & x_{5} & x_{6} & 0 \\
0 & x_{7} & x_{8} & x_{9} & 0 \\
0 & 0 & 0 & 0 & 0 \\
\end{pmatrix}
$$


$$
\Tiny
\label{eq:unfoldedWeights3d}
z = 
\begin{pmatrix}
\begin{pmatrix}
0     &   0   & 0     & 0     & w_{1} & w_{2} & 0     & w_{4} & w_{5}\\
0     &   0   & 0     & w_{1} & w_{2} & w_{3} & w_{4} & w_{5} & w_{6}\\
0     &   0   & 0     & w_{2} & w_{3} & 0     & w_{5} & w_{6} & 0    \\
0     & w_{1} & w_{2} & 0     & w_{4} & w_{5} & 0     & w_{7} & w_{8}\\
w_{1} & w_{2} & w_{3} & w_{4} & w_{5} & w_{6} & w_{7} & w_{8} & w_{9}\\
w_{2} & w_{3} & 0     & w_{5} & w_{6} & 0     & w_{8} & w_{9} & 0    \\
0     & w_{4} & w_{5} & 0     & w_{7} & w_{8} & 0     & 0     & 0    \\
w_{4} & w_{5} & w_{6} & w_{7} & w_{8} & w_{9} & 0     & 0     & 0    \\
w_{5} & w_{6} & 0     & w_{8} & w_{9} & 0     & 0     & 0     & 0    
\end{pmatrix}
\odot
\begin{pmatrix}
0     &   0   & 0     & 0     & x_{1} & x_{2} & 0     & x_{4} & x_{5}\\
0     &   0   & 0     & x_{1} & x_{2} & x_{3} & x_{4} & x_{5} & x_{6}\\
0     &   0   & 0     & x_{2} & x_{3} & 0     & x_{5} & x_{6} & 0    \\
0     & x_{1} & x_{2} & 0     & x_{4} & x_{5} & 0     & x_{7} & x_{8}\\
x_{1} & x_{2} & x_{3} & x_{4} & x_{5} & x_{6} & x_{7} & x_{8} & x_{9}\\
x_{2} & x_{3} & 0     & x_{5} & x_{6} & 0     & x_{8} & x_{9} & 0    \\
0     & x_{4} & x_{5} & 0     & x_{7} & x_{8} & 0     & 0     & 0    \\
x_{4} & x_{5} & x_{6} & x_{7} & x_{8} & x_{9} & 0     & 0     & 0    \\
x_{5} & x_{6} & 0     & x_{8} & x_{9} & 0     & 0     & 0     & 0    
\end{pmatrix}
\end{pmatrix}
\cdot
\begin{pmatrix}
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1
\end{pmatrix}
+
\begin{pmatrix}
b_{1} & b_{2} & b_{3}\\
b_{4} & b_{5} & b_{6}\\
b_{7} & b_{8} & b_{9}
\end{pmatrix}
$$

In case of three dimensional neural grids, images can be processed directly without converting them to an array.

The neural grid is not designed to achive anything close to SOTA for some dataset, rather it is designed so that local grid neurons can interact with each other and to form network structures not previously specified.


## Implementation

When I first implemented the idea for a neural grid, I took a very naive approach of iterating through the grid using two for-loops. This approach works, but required many hours to train a single epoch on the Fashion-MNIST dataset. Overall, the entire `forward()` method in the `NeuralGrid` class was a gigantic bottleneck - especially since the implementation was also only designed for a batch size of one. If at all, using the naive implementation below, the network could only crawl at best.


```python
class NeuralGrid(nn.Module):
    """
    Naive implementation of a neural grid in PyTorch
    """
    def __init__(self, n_layers, n_units):
        super().__init__()
        # Initialize trainable parameters
        self.grid_width = n_layers
        self.grid_height = n_units

        # Placeholder for activations
        self.a = [[torch.zeros(size=(1,)) for _ in range(self.grid_width + 1)] 
                   for _ in range(self.grid_height + 2)]

        # Activation function
        self.activation_function = torch.tanh

        # Trainable parameters
        fan_in = 3.0
        fan_out = 1.0
        w = xavier_uniform(size=(self.grid_height, self.grid_width), fan_in=fan_in, fan_out=fan_out)
        w = F.pad(input=w, pad=[0, 0, 1, 1], mode="constant", value=0.0)
        self.w = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.grid_height, self.grid_width)), 
                              requires_grad=True)

    def forward(self, x):

        # Assign features to grid
        for i in range(self.grid_height):
            self.a[i + 1][0] = x[i]

        # Feed features through grid
        for j in range(self.grid_width):
            for i in range(self.grid_height):
                self.a[i + 1][j + 1] = self.activation_function(self.a[i - 1][j] * self.w[i - 1][j]
                                                                + self.a[i][j] * self.w[i][j]
                                                                + self.a[i + 1][j] * self.w[i + 1][j]
                                                                + self.b[i][j])

        # Assign grid output to new vector
        x_out = torch.zeros(size=(self.grid_height,))
        for i in range(self.grid_height):
            x_out[i] = self.a[i + 1][-1]

        return x_out
```

I then wondered how to reformulate the problem on hand to increase computational speed, an realized that the mapping between layers in the grid can be formulated using a special form of vector-matrix multiplication as shown in Equation \eqref{eq:sparseWeightMatrix}. This approach resulted in the following implementation consisting of two classes. The `NeuralGrid` and the `GridLayer` class.

In `GridLayer` we reformulate the inner for-loop in the implementation above by combining unfolding with an element-wise product that is followed by a sum along the resulting matrix's rows.


```python
class GridLayer(nn.Module):
    """Class implements layer of a neural grid
    """
    def __init__(self, grid_height):
        super().__init__()
        self.kernel_size = 3  # do not change
        self.stride = 1  # do not change
        self.activation_function = torch.sin

        # Trainable parameters
        weight = xavier_uniform(size=(grid_height, ), fan_in=self.kernel_size, fan_out=1)
        self.weight = nn.Parameter(F.pad(input=weight, pad=[1, 1], mode="constant", value=0.0),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(size=(grid_height,)), requires_grad=True)

    def forward(self, x_in):
        # Same padding to ensure that input size equals output size
        x = F.pad(input=x_in, pad=[1, 1], mode="constant", value=0.0)

        # Unfold activations and weights for grid operations
        x = x.unfold(dimension=1, size=self.kernel_size, step=self.stride)
        w = self.weight.unfold(dimension=0, size=self.kernel_size, step=self.stride)

        # Compute activations
        x = self.activation_function((w * x).sum(dim=-1) + self.bias)
        return x


```

```python
class NeuralGrid(nn.Module):
    """ Implements a neural grid
    """

    def __init__(self, n_layers, n_units):
        super().__init__()

        #  self.grid_width = n_layers
        self.grid_height = n_units

        self.grid_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.grid_layers.append(GridLayer(grid_height=self.grid_height))

    def forward(self, x):
        for grid_layer in self.grid_layers:
            x = grid_layer(x)
        return x
```

### Comparison of Implementations

Network size: 784 x [128 x 128] x 10

I wondered, how both implementations above would compare in terms of their computational performance. That's why I tested how long they take for an epoch on the Fashion-MNIST dataset. Note, that the naive implementation can only process one data sample per training iteration.

Comparison of training time in seconds for a single epoch on the Fashion-MNIST dataset:

| Implementation | Accelerator | Batch size | Time per epoch | Speedup |
|:--------------:|:-----------:|:----------:|:--------------:|:-------:|
| Naive          | CPU         | 1          | $52427 \pm 607$ s (~14 h) | 1 |
| Unfold         | CPU         | 1          | $1503 \pm 288$ s          | 34 |
| Unfold         | CPU         | 16         | $404 \pm 59$ s            | 129 |
| Unfold         | CPU         | 32         | $329 \pm 33$ s            | 159 |
| Unfold         | CPU         | 64         | $257 \pm 4$ s             | 203 |
| Unfold         | GPU         | 64         | $21 \pm 1$ s              | 2469 |
| Unfold         | GPU         | 256        | $5.41 \pm 0.17$ s         | 9687 |

Compared to the vanilla implementation, I endend up with a speed-up of almost four orders.

## Experiments

The main focus of the experiments was to gather qualitative insights on how grid-based neural networks behave in a supervised classification task. For this reason, a grid-based model with a grid height of 16 cells and a depth of 32 cells was trained on the Fashion-MNIST dataset by optimising the multinomial logistic regression objective (softmax function) using the Adam optimization algorithm. The batch size during training was set to 64. The learning rate was initially set to $2e-4$, and then decreased by a factor of 10 every 100 epochs. In total, the learning was stopped after 400 epochs. The network's weights were initialized using the Xavier (TODO: cite!) initialization scheme. The biases were initialised with zero.

In order to force the grid to learn better representations, no activation function was used for the mapping from the image space to the grid's input dimension.


## Results and Discussion

This section shows some qualitative results from the grid's activation pattern. The following image shows the neural grid's activation pattern for the then different classes of the Fashion-MNIST dataset. Here, the activation patterns of 20 predictions were averaged for each class.

// Plot

It is interesting to see, that depending on the class, the signal uses different paths through the neural grid and by that is processed differently. The standard deviation shows, that the grid's deeper layers are less volatile indicating the higher level features have been detected.

// Plot

## Conclusion 

This blog post introduced a grid-based neural network architecture. Neural grids can be understood as a discretized signal processing network, where information are processed in a feedforward manner. The neural grid's placticity also allows to form arbitary network structures within the grid. The qualitative results have shown, that such pathways for the routing of signals ermerge in the grid during training.


---

```bibtex
@misc{blogpost,
  title={NeuralGrid: A Grid-based Neural Network},
  author={Fabi, Kai},
  howpublished={\url{https://kaifabi.github.io//NeuralGrid}},
  year={2021}
}
```

You find the code for this project [here][github_code].

<!-- Links -->

[github_code]: https://github.com/KaiFabi/NeuralGrid
