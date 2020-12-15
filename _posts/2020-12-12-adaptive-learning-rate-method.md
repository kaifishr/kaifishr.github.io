---
layout: post
title: "An Adaptive Learning Rate Method for Gradient Descent Algorithms"
date:   2020-12-12 12:12:12
---

**TL;DR**: It is possible to use gradient information to compute parameter-wise adaptive learning rates.

---

### Introduction 

In this blog post I present a novel per-parameter adaptive learning rate method for gradient descent optimization algorithms. The proposed method can easily applied to popular optimization algorithms. Popular optimization algorithms such as Stochastic Gradient Descent, Momentum, Nestrov Accelerated Gradient or ADAM show superior performance if our method is applied.

% Whats the motivation for this paper, what is the problem we want to solve?

In this paper we introduce a novel adaptive learning rate update schemes for gradient descent optimization algorithms. Our method uses the information provided by the gradients of current and past time steps to determine a suitable learning rate for the next gradient descent step. The method allows to determine a global learning rate for all weights or individual learning rates for all parameters. The presented technique can be implemented easily and allows common gradient descent algorithms to converge much faster.

Using our method for adaptive learning rates we demonstrate that popular optimizers such as SGD, Adam, Nestrov can significantly increase their performance.

### Related Work

### Method

In this section we derive an adaptive learning rate update scheme for gradient descent optimization algorithms. Stochastic gradient descent computes the gradient of a loss function $$L$$ with respect to the parameters $$\bm{w}$$. The update rule for each parameter can be written as follows:

\begin{equation}
w_i^{(t+1)} = w_i^{(t)} - \eta_{i}^{(t)} \frac{dL^{(t)}}{dw_i^{(t)}} \label{eqn:weight_update}
\end{equation}

Bla foo \eqref{eqn:weight_update}

We assume that we can propagate back the loss information to the learning rate to adapt the learning rate for the subsequent weight update. For parameter-wise learning rates this can be written as follows:

\begin{eqnarray}\label{eqn:learning_rate_update}
    \eta_{i}^{(t)} = \eta_{i}^{(t-1)} - \alpha \frac{dL^{(t)}}{d\eta_{i}^{(t-1)}}
\end{eqnarray}

Where $\alpha$ is a newly introduced hyperparameter determining the rate of change during the learning rate update. We can expand the last term of Equation \ref{eqn:learning_rate_update} using the chain rule of calculus and get

\begin{eqnarray}\label{eqn:dLdeta}
    \frac{dL^{(t)}}{d\eta_{i}^{(t-1)}} = \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dw_{i}^{(t)}}{d\eta_{i}^{(t-1)}}
\end{eqnarray}

Now we can use Equation \eqref{eqn:weight_update} to rewrite the right hand side of Equation \ref{eqn:dLdeta} such that it only consists of gradients of the loss with respect to the weights. Taking the derivate of Equation \ref{eqn:weight_update} with respect to the learning rate $\eta$ we get:

\begin{eqnarray}\label{eqn:dwdeta}
    \frac{dw_{i}^{(t)}}{d\eta_{i}^{(t-1)}} = -\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{eqnarray}

By substituting the last term of Equation \ref{eqn:dLdeta} with the result of Equation \ref{eqn:dwdeta} we get:

\begin{eqnarray}\label{eqn:dLdeta_2}
    \frac{dL^{(t)}}{d\eta_{i}^{(t-1)}} = -\frac{dL^{(t)}}{dw^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{eqnarray}

Now Equation \ref{eqn:dLdeta_2} can be inserted into Equation \ref{eqn:learning_rate_update} to get an update rule that allows to compute parameter-wise adaptive learning rates:

\begin{eqnarray}\label{eqn:learning_rate_update_2}
    \eta_{i}^{(t)} = \eta_{i}^{(t-1)} + \alpha \frac{dL^{(t)}}{dw^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{eqnarray}

Here, we can make several interesting observation. The update rule uses the last and current gradients to determine the next learning rate. The product of the old and the new gradient allows for two differnt cases. If both gradients are either positive or negative the result is positive resulting in an increase of the learning rate for the next weight update. This seems to make sense because subsequent gradients with the same sign indicate that one is moving towards a local minimum\todo{Not happy with this sentence}. On the other hand, successive gradients with different signs may indicate that there are difficulties in optimizing the considered parameter and that the optimizer has problems to converge. In this case, the product of both gradients is negative causing the learning rate for the considered parameter to decrease which can help to avoid overshooting the minimum. It is important to note that with this approach the learning rate can also be negative.

By inserting Equation \ref{eqn:learning_rate_update_2} into Equation \ref{eqn:weight_update} we obtain the following expression for the weight update rule with adaptive learning rate:

\begin{eqnarray}\label{eqn:weight_update_2}
    w_i^{(t+1)} = w_i^{(t)} - \eta_{i}^{(t-1)} \frac{dL^{(t)}}{dw_i^{(t)}} - \alpha \left(\frac{dL^{(t)}}{dw_{i}^{(t)}} \right)^2 \frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{eqnarray}

Here, the newly introduced parameter $\alpha$ determines the effect of the correction term on the weight update.

With the derivation above it is also possible to derive an update scheme for a global learning rate $\eta^{(t)}$ that is the same for all parameters. In this case we use the gradient information of all parameters to compute a single learning rate. To do this, we can write the last term in Equation \ref{eqn:learning_rate_update} as follows

\begin{eqnarray}\label{eqn:dLdeta_global}
    \frac{dL^{(t)}}{d\eta^{(t-1)}} = \frac{1}{||\Omega||} \sum_{i \in \Omega} \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dw_{i}^{(t)}}{d\eta^{(t-1)}}
\end{eqnarray}

Please note that the index $i$ is no longer needed for the learning rate. Thus we write $\eta^{(t)}$ instead of $\eta_{(i)}^{(t)}$. Now we insert the result of Equation \ref{eqn:dwdeta} into Equation \ref{eqn:dLdeta_global} to get:

\begin{eqnarray}\label{eqn:dLdeta_global_2}
    \frac{dL^{(t)}}{d\eta^{(t-1)}} = - \frac{1}{||\Omega||} \sum_{i \in \Omega} \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{eqnarray}

By inserting Equation \ref{eqn:dLdeta_global_2} into Equation \ref{eqn:learning_rate_update} we get the update rule for a global learning rate.

\begin{eqnarray}\label{eqn:learning_rate_update_global}
    \eta^{(t)} = \eta^{(t-1)} + \alpha \frac{1}{||\Omega||} \sum_{i \in \Omega} \frac{dL^{(t)}}{dw_{i}^{(t)}}\frac{dL^{(t-1)}}{dw_{i}^{(t-1)}}
\end{eqnarray}

We can now insert Equation \ref{eqn:learning_rate_update_global} into Equation \ref{eqn:weight_update} to obtain the weight update rule for a global adaptive learning rate

\begin{eqnarray}\label{eqn:weight_update_global}
w_i^{(t+1)} = w_i^{(t)} - \eta^{(t-1)}\frac{dL^{(t)}}{dw_i^{(t)}} - \frac{\alpha}{||\Omega||}\frac{dL^{(t)}}{dw_i^{(t)}} \sum_{j \in \Omega} \frac{dL^{(t)}}{dw_{j}^{(t)}}\frac{dL^{(t-1)}}{dw_{j}^{(t-1)}} 
\end{eqnarray}

### Experiments

In this section we empirically evaluate the proposed parameter-wise adaptive learning rate method presented above. We apply our method to different popular machine learning models such as linear regression, logistic regression, and multilayer fully connected neural networks.

To ensure comparability, the models were initialized with the same parameters. In order to determine the optimal learning rate for the experiments, a grid search was performed beforehand. For other hyper parameters like the momentum the values frequently found in the literature were used.


![gradient_descent_beale_adam](/assets/images/post6/gd_adam.png)
![gradient_descent_beale_gd](/assets/images/post6/gd_gd.png)
![gradient_descent_beale_gdm](/assets/images/post6/gd_gdm.png)
![loss_beale](/assets/images/post6/loss_beale.png)
![loss_rosenbrock](/assets/images/post6/loss_rosenbrock.png)

<!-- Links -->
