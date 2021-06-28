---
layout: post
title: "Accelerating Training with Taylor Weight Extrapolation"
---

**TL;DR**: Taylor series expansion in combination with finite difference approximations can be used to perform weight extrapolation between optimization steps by taking advantage of information stored in past gradients.

---

## TODOs

- Higher-order formulations might not work as well. Reasons might be that the approximations of higher-order derivatives are numerically unstable. There might be a high volatility of $w(t)$ especially at the beginning of training. A very small learning rate might be necessary.

- Is it reasonable to assume that higher-order formulations can be used in later stages of the training where updates / gradients are not as wild as in the beginning?



## Introduction 

Training deep neural networks is becoming increasingly more expensive due to the large size of modern network architectures and every increasing amount of available data.

Reducing the costs of training these models remains a challenge and methods to accelerate the training can have a significant effect on how expensive it is to train a model.

In this post I want to show how information stored in past gradients can be used for an extrapolation step, allowing to predict a new set of parameters between two optimization steps.

Leveraging these information stored in past gradients allows to accelerate training of neural networks. In this post, an explicit formula for extrapolation steps is derived for neural networks trained with stochastic gradient descent (SGD).

The following figure shows the basic idea of extrapolation steps.

<p align="center"> 
<img src="/assets/images/post11/weight_extrapolation.png" width="500"> 
<br>
<b>Figure 1:</b> Schematic drawing of how past gradient information can be used to predict a new set of model parameters at $n+3$.
</p>
{: #fig:weightExtrapolation}

[Figure 1](#fig:weightExtrapolation) shows the basic idea how past gradient information can be used to perform an intermediate weight extrapolation step. For a second order extrapolation step the current and previous gradients are necessary to compute a parameter set based on extrapolation. The graph also shows that this is equivalent to using the information stored in the last three sets of parameters.


## Method

The method is relatively simple and consists of using the Taylor series expansion of the function $w(t)$ describing the model parameters' behavior in the parameter space to derive an expression that represents an update rule to perform an extrapolation step. Here, $w(t)$ denotes a single trainable parameter's trajectory as a function of time. This extrapolation step, which is built in between normal optimization steps, then allows to determine a new set of parameters based on past gradient information. 

The Taylor series of a function $w(t)$ at point $t_0$ is the power series

$$
w(t) = \sum_{k=0}^{\infty} \frac{w^{(k)}(t)}{k!}\bigg\rvert_{t=t_0}(t-t_0)^k
$$

Here, we are going to use the second order Taylor series expansion to derive the formula for an extrapolation step.

$$
w(t) = w(t_0) + w'(t_0)(t-t_0) + \frac{1}{2}w''(t_0)(t-t_0)^2 + \mathcal{O}(t^3)
$$

Now we want to know how $w(t)$ behaves behind point $t_0$. This is why we evaluate function $w(t)$ at $t=t_0+dt$. The step size is thus just $dt = t-t_0$. Inserting these expressions and neglecting higher order terms leads to

$$
w(t_0+dt) = w(t_0) + w'(t_0)dt + \frac{1}{2}w''(t_0)dt^2
$$

To make thinks look a little bit friendlier, I'll use the following notation $w(t_0) = w_n$ and $w(t_0+dt) = w_{n+1}$, where the index $n$ represents the current optimization step. This leads us to the following expression

$$
\label{eq:secondOrder}
w_{n+1} = w_{n} + w'_{n}dt + \frac{1}{2}w''_{n}dt^2
$$

To be able to evaluate the derivations in the expression above, we use the finite difference method that is often used as an approximation of derivatives. In particular, we make use of the backward difference method since we only have past parameters available.  The backward difference of a function $w(t)$ at point $t$ is defined by the limit

$$
\label{eq:backwardDifference}
w'(t) = \lim_{\eta \rightarrow 0} \frac{w(t)-w(t-\eta)}{\eta}
$$

For our purpose we use the following approximation to describe the derivative at step $n$. 

$$
\label{eq:firstDerivative}
w'_{n}=\frac{w_{n}-w_{n-1}}{\eta}
$$

From the expression above, we can easily derive an expression for the second derivative

$$
w''_{n}=\frac{w'_{n}-w'_{n-1}}{\eta}
=\frac{\frac{w_{n}-w_{n-1}}{\eta}-\frac{w_{n-1}-w_{n-2}}{\eta}}{\eta}
=\frac{(w_{n}-w_{n-1})-(w_{n-1}-w_{n-2})}{\eta^2}
$$

OR

$$
\begin{equation}
\begin{gathered} \label{eq:secondDerivative}
w''_{n}&=&\frac{w'_{n}-w'_{n-1}}{\eta}\\
&=&\frac{\frac{w_{n}-w_{n-1}}{\eta}-\frac{w_{n-1}-w_{n-2}}{\eta}}{\eta}\\
&=&\frac{(w_{n}-w_{n-1})-(w_{n-1}-w_{n-2})}{\eta^2}
\end{gathered}
\end{equation}
$$

Inserting Equation \eqref{eq:firstDerivative} and \eqref{eq:secondDerivative} into Equation\eqref{eq:secondOrder} yields

$$
\begin{equation}
\begin{gathered} \label{eq:secondOrder2}
w_{n+1} &=& w_{n} + w'_{n}dt + \frac{1}{2}w''_{n}dt^2\\
&=& w_{n} + \frac{dt}{\eta}(w_{n}-w_{n-1}) + \frac{dt^2}{2\eta^2}((w_{n}-w_{n-1})-(w_{n-1}-w_{n-2}))
\end{gathered}
\end{equation}
$$

At this point, we can use the standard update rule of gradient descent 

$$
w_{n} = w_{n-1} - \eta \frac{\partial L}{\partial w_{n-1}}$$

to reformulate Equation \eqref{eq:secondOrder2} into

$$
\label{eq:secondOrder3}
w_{n+1} = w_n - dt \frac{\partial L}{\partial w_{n-1}} - \frac{dt^2}{2\eta}\left( \frac{\partial L}{\partial w_{n-1}} + \frac{\partial L}{\partial w_{n-2}}\right)
$$


By looking at the first two terms of Equation \eqref{eq:secondOrder3}, we see that we recovered the standard gradient descent update rule. Interestingly, this formula suggests to perform a standard gradient descent step plus a gradient descent step influenced by the last gradients.


## Implementation

## Experiments

## Results and Discussion


Weight extrapolation as described in this post might work better for lager batch sizes and/or smaller learning rates. 

Increasing the batch size allows for better gradient approximations potentially leading to a smoother trajectory through the parameter space.
as these allow better gradient approximations
- Discussion: Should work best for small learning rates and large batches sizes as these allow smooth gradients

## Conclusion 


---

```bibtex
@misc{blogpost,
  title={Neural Weight Extrapolation},
  author={Fabi, Kai},
  howpublished={\url{https://kaifabi.github.io//NeuralWeightExtrapolation}},
  year={2021}
}
```

You find the code for this project [here][github_code].

<!-- Links -->
[github_code]: https://github.com/KaiFabi/NeuralGrid
