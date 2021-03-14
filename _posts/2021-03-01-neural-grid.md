---
layout: post
title: "[DRAFT] NeuralGrid: A Grid-based Neural Network Architecture"
date:   2021-03-01 12:52:03
---

**TL;DR**: In this post I present a grid-based neural network architecture.

---

# TODO: 

- Ask in forum how to create sparse matrices as below. Otherwise matrix vector multiplication not possible.
- Show naive implementation
- Show that implementation with combining unfolding and element-wise product represent same operation and is much faster.
- Compare training time for different implementations.
- Implement neural grid for batch size > 1
- "Weights are locally shared to distribute the signal within the neural grid."
- "We have overlapping receptive fields. It therefore follows that every weight in grid receives three gradients."
- Visualize unfolding operations (tikZ)
- Can sparse matrix representation be used to justify fan_in =3 and fan_out = 3 (1)?

- (1) Show that mapping of naive implementation can be represented as a sparse matrix operation.
- (2) Show that unfolding (for kernel size 3) followed by an element wise multiplication can be represented as a sparse matrix.

## Introduction 

In this blog post I present a grid-based neural network architecture which I will call NeuralGrid. This network architecture is designed to allow atificial neurons to automatically interconnect into a network during the training process, thereby forming a previously undefined network structure. Compared to conventional network architectures, this type of network is therfore very plastic and can learn an optimal interconnection during training. The following figure shows the basic idea of a NeuralGrid.

<p align="center"> 
<img src="/assets/images/post10/neural_grid.png" width="800"> 
<b>Figure 1:</b> Basic idea of a grid-based neural network architecture.
</p>

**TODO: Add colors to Figure 1 to show that neurons and input / output of grid are the same?**

## Method

The network architecture consists of three main parts: The mapping from the input neurons $x_m$ to the grid's first layer, the mapping inside the grid, and finally the mapping from the last grid layer to the network's output.

The network's way of processing information can be split into two different types. The first type of operation from the netowork's input to the grid, and from the grid to the output consists of standard vector-matrix multiplications.

The information processing of a single neuron outside the grid, i.e. before and after the grid, can be described as follows. 

\begin{equation} \label{eq:preactivation_in}
    z_l = \sum_{m} w_{lm} x_m + b_l
\end{equation}

\begin{equation} \label{eq:activation_in}
    x_m = h(z_l)
\end{equation}

\begin{equation} \label{eq:preactivation_out}
    z_i = \sum_{j} w_{ij} x_j + b_i 
\end{equation}

\begin{equation} \label{eq:activation_out}
    x_i = h(z_i)
\end{equation}

Things look a bit different for the mapping inside the grid. Here, the mapping can be described as follows:

\begin{equation} \label{eq:preactivation_grid}
    z_p = \sum_{q \subset \Omega} w_{pq} x_q + b_p
\end{equation}

\begin{equation} \label{eq:activation_grid}
    x_p = h(z_p)
\end{equation}

Within the grid, weights are shared by several neurons. In the figure above, each weight within the grid contributes to three neurons in the next layer. This must also be taken into account in the subsequent calculation of the gradients insofar as the error signals for the respective weights must be accumulated.

The mapping within the grid can also be represented as a special kind of vector-matrix multiplication, which allows to use certain operations provided by PyTorch that allow a massive speedup during training. For an input layer consisting of five neurons, kernel size of 3, and stride 1, Equation \eqref{eq:preactivation_grid} can be written as

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

In Equation \eqref{eq:sparseWeightMatrix} we see that every weight, except for the weights at the grid's border, is involved in three operations, meaning that these weights receive three error signals or gradients. It can also be seen that every weights final gradient can be obtained by summing along the vertical axis of the weight's matrix corresponding gradient matrix.



<!-- progress -->


### Backpropagation Gradient Descent

Now we go backwards through the network to determine the gradients. We start computing the gradients for the weight matrix $w$ connecting the grid to the network's output and the biases $b$. To do this, we first need a loss function which is given by

\begin{equation} \label{eq:loss}
L = \frac{1}{2} \sum_i (y_i - x_i)^2
\end{equation}

Using the chain rule of calculus we can determine the gradients for the weights and biases.

$$
\begin{multline} \label{eq:dw_output}
\begin{aligned}
\frac{dL}{dw_{ij}} &= \frac{dz_i}{dw_{ij}} \cdot \frac{dx_i}{dz_i} \cdot \frac{dL}{dx_i}\\
                   &= x_j \cdot \underbrace{h'(z_i) \cdot (x_i - y_i)}_{\delta_i}\\
\end{aligned}
\end{multline}
$$

\begin{equation} \label{eq:db_output}
\frac{dL}{db_i} = \frac{dz_i}{db_i} \cdot \frac{dx_i}{dz_i} \cdot \frac{dL}{dx_i}
= h'(z_i) \cdot (x_i - y_i)
\end{equation}

Now we look at the last weighst of the grid. Here we have

\begin{equation} \label{eq:dw_grid}
\frac{dL}{dw_k} = \frac{dz_j}{dw_k} \cdot \frac{dx_j}{dz_j} \cdot \frac{dz_i}{dx_j} \cdot \frac{dx_i}{dz_i} \cdot \frac{dL}{dx_i}
\end{equation}

\begin{equation}
\frac{dL}{dw_k} = x_k \cdot h'(z_j) \cdot w_{ij} \cdot h'(z_i) \cdot (x_i - y_i)
\end{equation}

Here you can see that later in the implementation you can partly use earlier results.

## Implementation

Naive implementation

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

Comparison of training time in seconds for a single epoch on the Fashion-MNIST dataset for different neural grid implementations.

| Implementation | Time per epoch | Speedup |
|:--------------:|:--------------:|:-------:|
| Loop | $52427 \pm 607$ s (~14 h) | - |
| Unfold (CPU, batch 1) | $1503 \pm 288$ s | 34 |
| Unfold (CPU, batch 16) | $404 \pm 59$ s | 129 |
| Unfold (CPU, batch 32) | $329 \pm 33$ s | 159 |
| Unfold (CPU, batch 64) | $257 \pm 4$ s | 203 |
| Unfold (GPU, batch 64) | $21 \pm 1$ s | 2469 |
| Unfold (GPU, batch 256) | $5.41 \pm 0.17$ s | 9687 |

## Experiments

pass

## Results

## Discussion

pass

## Outlook

pass

---

You find the code for this project [here][github_code].

<!-- Links -->

[github_code]: https://github.com/KaiFabi/NeuralGrid
