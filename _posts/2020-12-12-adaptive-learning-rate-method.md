---
layout: post
title: "An Adaptive Learning Rate Method for Gradient Descent Algorithms"
date:   2020-12-12 12:12:12
---

**TL;DR**: Gradient information can be used to compute adaptive learning rates that allow gradient descent algorithms to converge much faster.

---

## Introduction 

In this blog post I present a novel adaptive learning rate method for gradient descent optimization algorithms. The proposed method uses gradient information that allow to either compute parameter-wise or global adaptive learning rates.

The method can easily applied to popular optimization algorithms such as Gradient Descent, Gradient Descent with Momentum, Nestrov Accelerated Gradient, ADAM, to name just a few. 

These optimization algorithm show better performance when equipped with an adaptive learning rate compared to a static global learning rate.

This method allows us to propagate loss information back to the learning rate.

## Related Work

In recent years many learning rate schedules have been proposed to accelerate the training of machine learning models. A good comparison of these methods can be found ***here***. These methods have in common, that they do not take information into account provided by the model during the training process such as loss or gradient information. 

Furthermore, the learning rate schedule of some of these methods approach very small learning rates at the end of a training period (learning rates with exponential decay, one-cycle learning rate policy). However, these methods are not very plausible from a biological point of view, since the model will not learn new information as well at the end of training as it did at the beginning of training, when the learning rate was higher. Small learning rates prevent the model from reaching a better local minima if new information are available.

For this reason, cyclic learning rate schedules may be the better option for continous learning. However, even here it is not always easy to find an optimal learning rate. Not to mention the period length of the learning rate to be run through.

For this reason, we can use the gradient information provided by the model to determine an adaptive learning rate at every backpropagation gradient descent step.

## Method

In this section we derive an adaptive learning rate update scheme for gradient descent optimization algorithms. Stochastic gradient descent computes the gradient of a loss function $$L$$ with respect to the parameters $$\boldsymbol{w}$$. The update rule for each parameter can be written as follows:

\begin{equation} \label{eq:weight_update}
w_i^{(t+1)} = w_i^{(t)} - \eta_{i}^{(t)} \frac{dL^{(t)}}{dw_i^{(t)}} 
\end{equation}

We assume that we can backpropagate the loss information to the learning rate $$\eta$$ to adapt the learning rate for the subsequent weight update. For parameter-wise learning rates this can be written as follows:

\begin{equation} \label{eq:learning_rate_update}
    \eta_{i}^{(t)} = \eta_{i}^{(t-1)} - \alpha \frac{dL^{(t)}}{d\eta_{i}^{(t-1)}}
\end{equation}

Where $$\alpha$$ is a newly introduced hyperparameter determining the rate of change of the adaptive learning rate. We can expand the last term of Equation \eqref{eq:learning_rate_update} using the chain rule of calculus and get

\begin{equation}\label{eq:dLdeta}
    \frac{dL^{(t)}}{d\eta_{i}^{(t-1)}} = \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dw_{i}^{(t)}}{d\eta_{i}^{(t-1)}}
\end{equation}

Now we can use Equation \eqref{eq:weight_update} to rewrite the right hand side of Equation \eqref{eq:dLdeta} such that it only consists of gradients of the loss with respect to the weights. Taking the derivate of Equation \eqref{eq:weight_update} with respect to the learning rate $$\eta$$ we get

\begin{equation}\label{eq:dwdeta}
    \frac{dw_{i}^{(t)}}{d\eta_{i}^{(t-1)}} = -\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{equation}

By substituting the last term of Equation \eqref{eq:dLdeta} with the result of Equation \eqref{eq:dwdeta} we get:

\begin{equation}\label{eq:dLdeta_2}
    \frac{dL^{(t)}}{d\eta_{i}^{(t-1)}} = -\frac{dL^{(t)}}{dw^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{equation}

Now Equation \eqref{eq:dLdeta_2} can be inserted into Equation \eqref{eq:learning_rate_update} to get the final result which is an update rule that allows to compute parameter-wise adaptive learning rates

\begin{equation}\label{eq:learning_rate_update_2}
    \eta_{i}^{(t)} = \eta_{i}^{(t-1)} + \alpha \frac{dL^{(t)}}{dw^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{equation}

Here, we can make several interesting observations. The update rule uses the gradients of the current and last time step to determine the next learning rate. 

The product of the old and the new gradient allows for two differnt cases. If both gradients are either positive or negative, the result is positive resulting in an increase of the learning rate for the next weight update. This seems to make sense since subsequent gradients with the same sign indicate that there is a clear direction within the loss landscape and that the error is probably moving towards a local minimum. 

On the other hand, successive gradients with different signs may indicate that there is no clear direction for the gradient descent in the loss landscape. In this case, the product of both gradients is negative causing the learning rate for the considered parameter to decrease. At this point it is important to note that this method allows the learning rate to become negative. Decreasing the learning rate allows a more careful gradient descent or even an ascent if the learning rater becomes negative. This can help to escape a local minima.

By inserting Equation \eqref{eq:learning_rate_update_2} into Equation \eqref{eq:weight_update} we obtain the following expression for the weight update rule with adaptive learning rate:

\begin{equation}\label{eq:weight_update_2}
    w_i^{(t+1)} = w_i^{(t)} - \eta_{i}^{(t-1)} \frac{dL^{(t)}}{dw_i^{(t)}} - \alpha \left(\frac{dL^{(t)}}{dw_{i}^{(t)}} \right)^2 \frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{equation}

As can be seen, the newly introduced parameter $$\alpha$$ determines the effect of the correction term on the weight update.

With the derivation above it is also possible to derive an update scheme for a global learning rate $$\eta^{(t)}$$ that is the same for all parameters. In this case we use the gradient information of all parameters to compute a single learning rate. To do this, we can write the last term in Equation \eqref{eq:learning_rate_update} as follows

\begin{equation}\label{eq:dLdeta_global}
    \frac{dL^{(t)}}{d\eta^{(t-1)}} = \frac{1}{||\Omega||} \sum_{i \in \Omega} \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dw_{i}^{(t)}}{d\eta^{(t-1)}}
\end{equation}

Where $$\Omega$$ denotes the model's set of trainable parameters. Please note that at this point the index $i$ is no longer needed for the learning rate. Thus we write $$\eta^{(t)}$$ instead of $$\eta_{(i)}^{(t)}$$. We now insert the result of Equation \eqref{eq:dwdeta} into Equation \eqref{eq:dLdeta_global} to get:

\begin{equation}\label{eq:dLdeta_global_2}
    \frac{dL^{(t)}}{d\eta^{(t-1)}} = - \frac{1}{||\Omega||} \sum_{i \in \Omega} \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{equation}

By inserting Equation \eqref{eq:dLdeta_global_2} into Equation \eqref{eq:learning_rate_update} we get the update rule for a global learning rate.

\begin{equation}\label{eq:learning_rate_update_global}
    \eta^{(t)} = \eta^{(t-1)} + \alpha \frac{1}{||\Omega||} \sum_{i \in \Omega} \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{equation}

We can now insert Equation \eqref{eq:learning_rate_update_global} into Equation \eqref{eq:weight_update} to obtain the weight update rule for a global adaptive learning rate

\begin{equation}\label{eq:weight_update_global}
    w_i^{(t+1)} = w_i^{(t)} - \eta^{(t-1)}\frac{dL^{(t)}}{dw_i^{(t)}} - \frac{\alpha}{||\Omega||}\frac{dL^{(t)}}{dw_i^{(t)}} \sum_{j \in \Omega} \frac{dL^{(t)}}{dw_{j}^{(t)}}\frac{dL^{(t-1)}}{dw_{j}^{(t-1)}} 
\end{equation}

As we can see, the proposed method uses gradient information provided by the current and past time step to determine a learning rate for the next gradient descent step. The method allows to compute an adaptive global learning rate that determines the adjustment of all weights or parameter-wise adaptive learning rates. Moreover, the proposed method for determining adaptive learning rates is very easy to implement.

## Implementation

In this section I show how the method introduced above can be integrated into common gradient descent optimization algorithms. The implementation is fairly straightforward and can also be found on my Github repository.

First, we define the test function. In this case Beale's function:

```python
def f(x, y):
    """
    Beale's function
    """
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
```

Next, the partial derivatives with respect to $x$ and $y$ are implemented. Here I use a simple implementation of the difference quotient (see [here][numerical_differentiation] for more stable versions of the difference quotient) in order to be able to also use other test functions.

```python
def dfdx(x, y, h=1e-9):
    return 0.5 * (f(x+h, y) - f(x-h, y)) / h

def dfdy(x, y, h=1e-9):
    return 0.5 * (f(x, y+h) - f(x, y-h)) / h
```

The next step is to implement the standard gradient descent. Here the learning rate $\eta$ is constant for the entire optimization process.

```python
eta_x = eta 
eta_y = eta 

def gradient_descent():
    x -= eta * dfdx(x, y)
    y -= eta * dfdy(x, y)
```

Given the gradient descent optimizer as implemented above, we can now add the adaptive learning rate method to the algorithm by caching the new gradients to determine the new learning rate based on the last two gradients and then performing gradient descent. Afterwards, the new gradients are cached for the next optimization step.

```python
eta_x = eta 
eta_y = eta 

dx_old = 0.0
dy_old = 0.0

def gradient_descent():
    dx = dfdx(x, y)
    dy = dfdy(x, y)

    eta_x += alpha * dx * dx_old
    eta_y += alpha * dy * dy_old

    x -= eta_x * dx
    y -= eta_y * dy

    dx_old = dx
    dy_old = dy
```

The next example shows the integration of an adaptive learning step in the case of the Adam optimizer. The implementation steps are the same.

```python
def gradient_descent_adam():
    dx = dfdx(x, y)
    dy = dfdy(x, y)

    m_x = beta_1 * m_x_old + (1.0 - beta_1) * dx
    m_y = beta_1 * m_y_old + (1.0 - beta_1) * dy

    v_x = beta_2 * v_x_old + (1.0 - beta_2) * dx * dx
    v_y = beta_2 * v_y_old + (1.0 - beta_2) * dy * dy

    m_x_hat = m_x / (1.0 - beta_1**(i+1))
    m_y_hat = m_y / (1.0 - beta_1**(i+1))

    v_x_hat = v_x / (1.0 - beta_2**(i+1))
    v_y_hat = v_y / (1.0 - beta_2**(i+1))

    eta_x += alpha * dx * dx_old
    eta_y += alpha * dy * dy_old

    x -= (eta_x / (np.sqrt(v_x_hat) + epsilon)) * m_x_hat
    y -= (eta_y / (np.sqrt(v_y_hat) + epsilon)) * m_y_hat

    m_x_old = m_x
    m_y_old = m_y

    v_x_old = v_x
    v_y_old = v_y

    dx_old = dx
    dy_old = dy
```

## Experiments

Now we want to performe some experiments that allow us to compare popular gradient descent optimization algorithms such as Gradient Descent (GD), Gradient Descent with Momentum (GDM), Nestrov Accelerated Gradient (NAG), and Adam with and without the parameters-wise adaptive learning rate. Optimizers equipped with an adaptive learning rate method are marked by a plus sign (GD+, GDM+, NAG+, Adam+).

To better understand the behaviour of these methods we can visualize the gradient descent using a popular test function (see also [here][test_functions] for more such functions) for optimization algorithms such as Beale's function:

 $$f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$$

Beale's function has a global minimum at $$(x,y)=(3,0.5)$$ which is indicated by a black star in the results section.

In order to determine an optimal learning rate $$\eta$$ as well as the hyperparameter $$\alpha$$ for each optimizers, a grid search was performed beforehand. The optimal hyperparameter is determined by how fast the global minimum is reached. To meassure the speed of convergence of different optimizers I used a loss function that is defined by the Euclidean distance to the global minimum.

| Hyperparameter | GD | GD+ | GDM | GDM+ | NAG | NAG+ | Adam | Adam+ |
|----------------|----|-----|-----|------|-----|------|------|-------|
| $$\eta$$ | 0.01 | 0.01 | 0.015 | 0.01 | 0.006 | 0.006 | 0.0005 | 0.0005 |
| $$\alpha$$ | 0 | 1e-4 | 0 | 1e-5 | 0 | 1e-6 | 0 | 1e-8 |

Please note, that the initial learning rate for all optimizers with adaptive learning rate are equal or smaller compared to the standard algorithm.

For other hyperparameters, frequently used values found in the literature were used:

$$\gamma=0.5$$, $$\beta_1=0.9$$, $$\beta_2=0.99$$, $$\epsilon=1e-8$$

## Results

For better comparability, results are shown for one optimizer at a time.

### Gradient Descent

The following figure shows the behaviour of the classic Gradient Descent (GD) algorithm. We see that the algorithms equipped with an adaptive learning rate (GD+) approaches the global minimum on a similar path but with much larger steps. 

![](/assets/images/post6/gd_beale_gd_alpha_1em5.png)

This behaviour is also reflected in the loss as apparent in the next figure. The loss shows that the adaptive learning rate not only allows to approach the global minimum much faster, it also gets about an order of magnitued closer after convergence.

![](/assets/images/post6/loss_gd_beale_1em4.png)

### Gradient Descent with Momentum

In the case with momentum, the results behave similarly to those for the classic gradient descent. Due to the adaptive learning rate, the global minimum is reached in a more direct way and with larger steps.

![](/assets/images/post6/gd_beale_gdm_alpha_5em5.png)

Here we see again, that the optimizer equipped with an adaptive learning rate converges much faster and also gets much closer to the global minimum.

![](/assets/images/post6/loss_gdm_beale_5em4.png)

### Nestrov Accelerated Gradient

In case of the Nestrov Accelerated Gradient (NAG) optimizer with a fixed learning rate, the optimizer approaches the global minimum more carfully compared to the method equipped with an adaptive learning rate. Here we see that NAG+ again behaves more aggressive right at the beginning of the optimization process.

![](/assets/images/post6/gd_beale_nag_alpha_1em6.png)

Even thought the gradient descent with NAG+ does not approach the minimum right from the start, it converges with a slight delay much faster requiring only half the steps to converge compared to NAG.

![](/assets/images/post6/loss_nag_beale.png)

### Adam

The next figure shows the results for the Adam optimization algorithm. The results show that the Adam algorithm with constant learning rate approaches to global minimum in a steady and a very carfully way. If we equip the Adam optimizer with an parameter-wise adaptive learning rate, we can observe, that the gradient descent is much more aggresive meaning, that the global minimum is approached with larger steps and in a more direct way.

![](/assets/images/post6/gd_beale_adam_alpha_1em7.png)

If we look at how the loss behaves we see how much faster the optimizer with adaptive learning rate converges. Two things in particular stand out here. Adam+ get very close to the global minimum before it than oscillates much further away around the minimum. On the other hand, even though Adam takes much for time to converge, it get closer to the global minimum by almost two order of magnitudes.

![](/assets/images/post6/loss_adam_beale.png)

## Discussion

The visualization of the gradient descent already indicates that the global minimum is reached faster for optimizers that use the adaptive learning rate presented above.

Using an adaptive learning rate allows to ... less time necessary for hyperparameter search. In principle it is possible to start with a learning rate of zero.

## Outlook

The method also allows for poorly chosen hyperparameters to converge really fast.

Using our method for adaptive learning rates we demonstrate that popular optimizers such as SGD, Adam, Nestrov can significantly increase their performance.

<!-- Links -->

[numerical_differentiation]: https://en.wikipedia.org/wiki/Numerical_differentiation
[test_functions]: https://en.wikipedia.org/wiki/Test_functions_for_optimization
