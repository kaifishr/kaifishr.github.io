---
layout: post
title: "An Adaptive Learning Rate Method for Gradient Descent Algorithms"
date:   2020-12-12 12:12:12
---

**TL;DR**: Gradient information can be used to compute adaptive learning rates that allow gradient descent algorithms to converge much faster.

---

### Introduction 

In this blog post I present a novel adaptive learning rate method for gradient descent optimization algorithms. The proposed method uses gradient information that allow to either compute parameter-wise or global adaptive learning rates.

The method can easily applied to popular optimization algorithms such as Gradient Descent, Gradient Descent with Momentum, Nestrov Accelerated Gradient, ADAM, to name just a few. 

These optimization algorithm show better performance when equipped with an adaptive learning rate compared to a static global learning rate.

### Related Work

In recent years many learning rate schedules have been proposed to accelerate the training of machine learning models. A good comparison of these methods can be found ***here***. These methods have in common, that they do not take information into account provided by the model during the training process such as loss or gradient information. 

Furthermore, the learning rate schedule of some of these methods approach very small learning rates at the end of a training period (learning rates with exponential decay, one-cycle learning rate policy). However, these methods are not very plausible from a biological point of view, since the model will not learn new information as well at the end of training as it did at the beginning of training, when the learning rate was higher. Small learning rates prevent the model from reaching a better local minima if new information are available.

For this reason, cyclic learning rate schedules may be the better option for continous learning. However, even here it is not always easy to find an optimal learning rate. Not to mention the period length of the learning rate to be run through.

For this reason, we can use the gradient information provided by the model to determine an adaptive learning rate at every backpropagation gradient descent step.

### Method

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

### Experiments

In this section we empirically evaluate the proposed parameter-wise adaptive learning rate method presented above. We apply our method to different popular machine learning models such as linear regression, logistic regression, and multilayer fully connected neural networks.

To ensure comparability, the models were initialized with the same parameters. In order to determine the optimal learning rate for the experiments, a grid search was performed beforehand. For other hyper parameters like the momentum the values frequently found in the literature were used.

### Results

![gradient_descent_beale_adam](/assets/images/post6/gd_adam.png)
![gradient_descent_beale_gd](/assets/images/post6/gd_gd.png)
![gradient_descent_beale_gdm](/assets/images/post6/gd_gdm.png)
![loss_beale](/assets/images/post6/loss_beale.png)
![loss_rosenbrock](/assets/images/post6/loss_rosenbrock.png)

### Outlook

The method also allows for poorly chosen hyperparameters to converge really fast.

Using our method for adaptive learning rates we demonstrate that popular optimizers such as SGD, Adam, Nestrov can significantly increase their performance.

<!-- Links -->
