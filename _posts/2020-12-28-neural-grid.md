---
layout: post
title: "NeuralGrid: A Grid-based Neural Network Architecture"
date:   2020-12-28 11:55:23
---

**TL;DR**: bla foo 

---

## Introduction 

In this blog post I present a grid-based neural network architecture.

## Related Work

bla

## Method

In this section, I will first talk about the feedforward process before moving on to cover the backpropagation gradient descent part in a bit more detail.

The network architecture consists of three main parts: The mapping from the input to the grid, the processing of incoming informatino by the grid, and finally the mapping from the grid to the network's output. This feedforward process can be implemented quite easily.

### Feedforward

The mapping from inputs to outputs is a fairly simple process that consists of successive matrix-vector multiplications. During this process, the network's input is transformed in a nonlinear fashion. Here we split the process into two parts. The processing outside and the processing inside of the grid.

Information processing of a single neuron outside the grid, i.e. before and after the grid, can be described as follows.

#### Input to Grid

\begin{equation} \label{eq:preactivation_in}
    z_i = \sum_{j} w_{ij} x_j + b_i 
\end{equation}

\begin{equation} \label{eq:activation_in}
    x_i = h(z_i)
\end{equation}

#### Grid to Output

\begin{equation} \label{eq:preactivation_out}
    z_i = \sum_{j} w_{ij} x_j + b_i 
\end{equation}

\begin{equation} \label{eq:activation_out}
    x_i = h(z_i)
\end{equation}

#### Grid

For the grid itself applies:

\begin{equation} \label{eq:preactivation_grid}
    z_i = \sum_{j \subset \Omega} w_{ij} x_j + b_i 
\end{equation}

\begin{equation} \label{eq:activation_grid}
    x_i = h(z_i)
\end{equation}

Within the grid, the same weights are shared by several neurons. This must be taken into account in the subsequent calculation of the gradients insofar as the error signals for the respective weights must be accumulated.

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

```python
def f(x, y):
    pass
```

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
