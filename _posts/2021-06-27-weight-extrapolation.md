---
layout: post
title: "Weight Extrapolation"
---
<!--title: "Weight Extrapolation for Accelerated Training"-->
<!--title: "Weight Extrapolation: Looking Back to go Forward"-->
<!--title: "Parameter Extrapolation"-->
<!--title: "Gradient-based Weight Extrapolation"-->
<!--Accelerating Training with Taylor Weight Extrapolation-->

**TL;DR**: Taylor series expansion in combination with finite difference approximations can be used to perform weight extrapolation between optimization steps by taking advantage of information stored in past gradients.

---

## Introduction 

Training deep neural networks is becoming increasingly more expensive due to the large size of modern network architectures and every increasing amount of available data. Reducing the costs of training these models remains a challenge and methods to accelerate the training can have a significant effect on how expensive it is to train a model.

In this post I want to show how information stored in past gradients can be used for an extrapolation step, allowing to predict a new set of parameters between two optimization steps. Leveraging these information stored in past gradients allows faster training of gradient based machine learning algorithms and thus also accelerates training of neural networks. 

In this post, an explicit formula for extrapolation steps is derived for neural networks trained with stochastic gradient descent (SGD). The following figure shows the basic idea of extrapolating model parameters.

<p align="center"> 
<img src="/assets/images/post11/weight_extrapolation.png" width="500"> 
<br>
<b>Figure 1:</b> Schematic drawing of how past gradient information can be used to predict a new set of model parameters at $n+1$.
</p>
{: #fig:weightExtrapolation}
- **TODO: Add this image to repository README.md**

[Figure 1](#fig:weightExtrapolation) shows the basic idea how past gradient information can be used to perform an intermediate weight extrapolation step. For a second order extrapolation step the current and previous gradients are necessary to compute a parameter set based on extrapolation. The graph also shows that this is equivalent to using the information stored in the last three sets of parameters.


## Method

The method is relatively simple and consists of using the Taylor series expansion of the function $w(t)$ describing the model parameters' behavior in the parameter space to derive an expression that represents an update rule to perform an extrapolation step. Here, $w(t)$ denotes a single trainable parameter's trajectory as a function of time. This extrapolation step, which is built in between normal optimization steps, then allows to determine a new set of parameters based on past gradient information. 

The Taylor series of a function $w(t)$ at point $t_{0}$ is the power series

$$
w(t) = \sum_{k=0}^{\infty} \frac{w^{(k)}(t)}{k!}\bigg\rvert_{t=t_{0}}(t-t_{0})^k
$$

Here, we are going to use the second order Taylor series expansion to derive the formula for an extrapolation step.

$$
\begin{equation}
\begin{aligned} 
\label{eq:taylorExpansion}
w(t) = w(t)\bigg\rvert_{t=t_{0}} &+ w'(t)\bigg\rvert_{t=t_{0}}(t-t_{0}) \\
&+ \frac{1}{2}w''(t)\bigg\rvert_{t=t_{0}}(t-t_{0})^2\\
&+ \frac{1}{6}w'''(t)(t-t_{0})^3\bigg\rvert_{t=t_{0}} + \mathcal{O}(t^4)
\end{aligned}
\end{equation}
$$

For the extrapolation we are trying to guess the behavior of $w(t)$ at $t=t_{0}+\Delta t$. The step size is thus just $\Delta t = t-t_{0}$. Inserting both expressions above leads to

$$
w(t_{0}+\Delta t) = w(t_{0}) + w'(t_{0})\Delta t + \frac{1}{2}w''(t_{0})\Delta t^2 + \frac{1}{6}w'''(t_{0})\Delta t^3 + \mathcal{O}(t^4)
$$

To make thinks look a little bit friendlier, I'll use the following notation $w(t_{0}) = w_{n}$ and $w(t_{0}+\Delta t) = w_{n+1}$, where the index $n$ represents the current optimization step. This leads us to the following expression

$$
\label{eq:thirdOrderExtrapolation}
w_{n+1} = w_{n} + w'_{n}\Delta t + \frac{1}{2}w''_{n}\Delta t^2 + \frac{1}{6}w'''_{n}\Delta t^3 + \mathcal{O}(\Delta t^4)
$$

Equation \eqref{eq:thirdOrderExtrapolation} tells us, how we can use higher-order derivatives to perform an extrapolation step. Now we are left with finding good approximations for the derivatives of $w$ at the point-of-interest $t_{0}$. There are several methods to do that. I'll start with a fairly straightforward approach where I use the finite backward difference method to approximate higher-order derivatives. After that, I'll derive higher-order derivatives using the backward differentiation formula.

### Finite Backward Differences

The derivative of $w(t)$ at point $t$ is then defined by the limit when we let $h$ go to zero

$$
\label{eq:definitionDerivative}
w'(t) = \lim_{h \rightarrow 0} \frac{w(t)-w(t-h)}{h}
$$

To compute the derivations in the expression above, we use finite differences. A finite difference is often used to approximate derivatives, typically in numerical differentiation. To be more precise, we use the backward difference since we only have the parameters' previous states available to work with, meaning the function values at $t$ and $t-h$.

$$
\label{eq:backwardDifference}
\nabla_{h}[w](t) = w(t) - w(t-h)
$$

If the finite difference above is divided by $h$ one gets a difference quotient. To be more precise, if $h$ has a fixed non-zero value instead of approaching zero, then the right-hand side of the Equation \ref{eq:definitionDerivative} would be written

$$
\label{eq:firstOrderBackward}
w'(t) = \frac{\nabla^1_{h}[w](t)}{h} = \frac{w(t)-w(t-h)}{h} + \mathcal{O}(h)
$$

which is already the first-order backward difference quotient. To derive the second-order backward difference we recursively insert the first-order backward difference into its own definition which leads to

$$
\begin{equation}
\begin{aligned} 
\label{eq:secondOrderBackward}
w''(t) 
&= \frac{\nabla^2_{h}[w](t)}{h^2} + \mathcal{O}(h)\\
&= \frac{w'(t)-w'(t-h)}{h} + \mathcal{O}(h)\\
&= \frac{\frac{w(t)-w(t-h)}{h}-\frac{w(t-h)-w(t-2h)}{h}}{h} + \mathcal{O}(h)\\
&= \frac{w(t)-2w(t-h)+w(t-2h)}{h^2} + \mathcal{O}(h)\\
\end{aligned}
\end{equation}
$$

For the third-order backward difference we get

$$
\begin{equation}
\begin{aligned} 
\label{eq:thirdOrderBackward}
w'''(t) 
&= \frac{\nabla^3_{h}[w](t)}{h^3} + \mathcal{O}(h)\\
&= \frac{w(t)-3w(t-h)+3w(t-2h)-w(t-3h)}{h^3} + \mathcal{O}(h)
\end{aligned}
\end{equation}
$$
 
As an aside, by examining the pattern for the expressions above more closely, it can be seen that a general finite backward difference formula exists

$$
\nabla_{h}^{n}[w](t) = \sum_{i=0}^n (-1)^i {n \choose i} w(t-ih)
$$

Aside over.

Now it remains to formulate the approximations of the derivatives in terms available to us during training. For that I'm going to use the standard gradient update rule 

$$
\label{eq:updateRule}
w_{n} = w_{n-1} - \eta \frac{\partial L}{\partial w_{n-1}}
$$

where $\frac{\partial L}{\partial w_{n-1}}$ reprents the change of the loss represented by $L$ with respect to model weight $w$ at optimization step $n-1$ to express the approximations of the derivatives using gradients computed during the optimization process. For a cleaner look, I'm going to use the notation $w_{n-1} = w(t-h)$ and $\partial_{w_{n-1}} L = \frac{\partial L}{\partial w_{n-1}}$ to present the expressions in a more tidy way. If we omitt higher-order terms, the equalities above become approximations. For the first, second, and third derivative we thus obtain

$$
\begin{equation}
\begin{aligned} 
\label{eq:firstOrderBackward2}
w'_{n} 
&= \frac{w_{n}-w_{n-1}}{h}\\
&= -\frac{\eta}{h}\partial_{w_{n-1}} L
\end{aligned}
\end{equation}
$$


$$
\begin{equation}
\begin{aligned} 
\label{eq:secondOrderBackward2}
w''_{n} 
&= \frac{(w_{n}-w_{n-1})-(w_{n-1}-w_{n-2})}{h^2}\\
&= -\frac{\eta}{h^2}(\partial_{w_{n-1}} L - \partial_{w_{n-2}} L)
\end{aligned}
\end{equation}
$$


$$
\begin{equation}
\begin{aligned} 
\label{eq:thirdOrderBackward2}
w'''_{n} 
&= \frac{(w_{n}-w_{n-1}) - 2(w_{n-1}-w_{n-2}) + (w_{n-2}-w_{n-3})}{h^3}\\
&= -\frac{\eta}{h^3}(\partial_{w_{n-1}} L - 2\partial_{w_{n-2}} L + \partial_{w_{n-3}} L)
\end{aligned}
\end{equation}
$$

By inserting Equation \ref{eq:firstOrderBackward2}, \ref{eq:secondOrderBackward2}, and \ref{eq:thirdOrderBackward2} into Equation \ref{eq:thirdOrderExtrapolation} yields

$$
\begin{equation}
\begin{aligned} 
\label{eq:thirdOrderExtrapolationBackward}
w_{n+1} = w_{n} 
&- \frac{\eta\Delta t}{h} \partial_{w_{n-1}} L \\
&- \frac{\eta\Delta t^2}{2h^2}\left( \partial_{w_{n-1}} L - \partial_{w_{n-2}} L\right) \\
&- \frac{\eta\Delta t^3}{6h^3}\left( \partial_{w_{n-1}} L - 2\partial_{w_{n-2}} L + \partial_{w_{n-3}} L \right)
\end{aligned}
\end{equation}
$$

Equation \ref{eq:thirdOrderExtrapolationBackward} now contains only expressions that are known to perform an extrapolation step starting from $w_{n}$. By looking at the first term of Equation \eqref{eq:thirdOrderExtrapolationBackward}, we see that we recovered the standard gradient descent update rule with step size $\frac{\eta \Delta t}{h}$. 

As an aside, for a small but finite $h$ we can approximate the derivative $w'(t)$ by $w'(t) \approx \frac{w_{t}-w_{t-1}}{\eta}$ and thus we have $h \equiv \eta$. This approximation is valid for small enough learning rates $\eta$. However, small values for $\eta$ during training can take our model ages to converge. On the other hand, relatively large learning rates $\eta$ lead to a poor approximation of $$w'(t)$$, $$w''(t)$$, and $$w'''(t)$$ necessary for the extrapolation. Aside over.

At this point I would like to emphasize that the formulation above for the extrapolation contains higher-order derivaitvesthat deliver good results only for very small step sizes $h$ or $\eta$. For this reason, better approximations for the derivatives of $w(t)$ are to be derived in the following section.

### Backward Differentiation Formula

In this section I derive a backward differentiation formula (BDF) that allows better approximations for the derivatives of $w(t)$. The used approach is an implicit method for the numerical integration of ordinary differential equations (ODE) of the form

$$
\label{eq:ode}
\frac{dw(t)}{dt} = f(t, w(t))
$$

The method works analogously also for higher order derivatives. As we don't have a closed form of $w(t)$ we can write the left hand side of Equation \ref{eq:ode} as a linear combination of values we know $w_{n}, w_{n-1}, \cdots$, representing past parameter states, and the values we are searching for $a_{n}, a_{n-1}, \cdots$. We start with the approximation of the first derivative and express $\frac{dw(t)}{dt}$ by

$$
\label{eq:odeAnsatz}
\frac{dw(t)}{dt}\bigg\rvert_{t=t_{n+1}} = \sum_{i=0}^n a_{n+1-i} w_{n+1-i}
$$

Here, I'm going to use three linear combinations to approximate $\frac{dw(t)}{dt}$

$$
\label{eq:bdfApproximationFirstDerivative}
\frac{dw(t)}{dt}\bigg\rvert_{t=t_{n+1}} = a_{n+1} w_{n+1} + a_{n} w_{n} + a_{n-1} w_{n-1} + a_{n-2} w_{n-2}
$$

Now we are going to determine the coefficients $a$ of the linear combinations above by replacing the known values ($w_{n}$, $w_{n-1}$, and $w_{n-2}$, i.e., the past parameter states) with suitable Taylor expansions about $\tau = t_{n+1}$. The Taylor expansion of a function $w(t)$ is given by

$$
\label{eq:taylorExpansionBDF}
w(t) = \sum_{k=0}^{\infty} \frac{w^{(k)}(t)}{k!}\bigg\rvert_{t=\tau}(t-\tau)^k
$$

To compute an approximation of $w_{n}$, we use $t = \tau - h$ or $t - tau = -h$ in Equation \ref{eq:taylorExpansionBDF}.

$$
w_{n} = \sum_{k=0}^{\infty} \frac{w^{(k)}(t)}{k!}\bigg\rvert_{t=\tau}(-h)^k
$$

To make the notation more readable, we'll use here $t_{n+1} = t+h$, $w(t_{n+1})=w_{n+1}$, and therefore $w(t_{n+1}-h)=w_{n}$. Thus we have 

$$
\begin{equation}
\begin{aligned} 
w_{n} = w_{n+1} - h w'_{n+1} + \frac{1}{2}h^2w''_{n+1} - \frac{1}{6}h^3w'''_{n+1} + \mathcal{O}(h^4)
\end{aligned}
\end{equation}
$$

Now that we have addressed the case for $t = \tau - h$ to derive $w_{n}$, let us consider the case for $t = \tau - 2h$ to derive $w_{n-1}$. By inserting $t - \tau = -2h$ into Equation \ref{eq:taylorExpansionBDF}, we get the following expression for $w_{n-1}$

$$
\begin{equation}
\begin{aligned} 
w_{n-1} 
&= \sum_{k=0}^{\infty} \frac{w^{(k)}(t)}{k!}\bigg\rvert_{t=\tau}(-2h)^k\\
&= w_{n+1} - 2hw'_{n+1} + 2h^2w''_{n+1} - \frac{4}{3}h^3w'''_{n+1} + \mathcal{O}(h^4)
\end{aligned}
\end{equation}
$$

Now only $w_{n-2}$ is missing. Here we proceed analogously. We want an expression for $w(t)$ at $t = \tau - 3h$. By inserting $t - \tau = -3h$ into Equation \ref{eq:taylorExpansionBDF} we get

$$
w_{n-2} = w_{n+1} - 3hw'_{n+1} + \frac{9}{2}h^2w''_{n+1} - \frac{9}{2}h^3w'''_{n+1} + \mathcal{O}(h^4)
$$

Since we want to find the value of $w_{n+1}$ at time $t_{n+1}$ that balances the ordinary differential equation in Equation \ref{eq:ode} we now replace the known values (past states) with suitable Taylor expansions about $\tau = t_{n+1}$ in Equation \ref{eq:bdfApproximation}. Substituting the Taylor expansions for $w_{n}$, $w_{n-1}$, and $w_{n-2}$ into Equation \ref{eq:bdfApproximation} gives

$$
\begin{equation}
\begin{aligned} 
w_{n}'
& = a_{n} w_{n}\\
& + a_{n-1} (w_{n} - hw'_{n} + \frac{1}{2}h^2w''_{n} - \frac{1}{6}h^3w'''_{n})\\
& + a_{n-2} (w_{n} - 2hw'_{n} + 2h^2w''_{n+1} - \frac{4}{3}h^3w'''_{n})\\
& + a_{n-3} (w_{n} - 3hw'_{n} + \frac{9}{2}h^2w''_{n} - \frac{9}{2}h^3w'''_{n})\\
\end{aligned}
\end{equation}
$$

We can rewrite this a bit to get

$$
\begin{equation}
\begin{aligned}
\label{eq:bdfApproximationTaylor}
w_{n}'
& = (a_{n} + a_{n-1} + a_{n-2} + a_{n-3}) w_{n}\\
& + (0a_{n} - ha_{n-1} - 2ha_{n-2} - 3ha_{n-3})w'_{n}\\
& + (0a_{n} + \frac{1}{2}h^2a_{n-1} + 2h^2a_{n-2} - \frac{9}{2}h^2a_{n-3})w''_{n}\\
& + (0a_{n} - \frac{1}{6}h^3a_{n-1} - \frac{4}{3}h^3a_{n-2} - \frac{9}{2}h^3a_{n-3})w'''_{n}
\end{aligned}
\end{equation}
$$

In order to get the best approximation for $w_{n}'$, we seek weights that zero the coefficients $a_{i}$ of the zeroth-, second-, and third-order derivatives and give a coefficient of one for the first-order derivatives. In other words, we search for coefficients $a_{i}$ so that Equation \ref{eq:bdfApproximationTaylor} is satisfied.

We can find the coefficients that satisfy Equation \ref{eq:bdfApproximationTaylor} by expressing it as a square linear system of equations.

$$
\begin{equation}
\begin{aligned}
\label{eq:linearSystemOfEquations}
a_{n} + a_{n-1} + a_{n-2} + a_{n-3} = 0\\
0 - ha_{n-1} - 2ha_{n-2} - 3ha_{n-3} = 1\\
0 + \frac{1}{2}h^2a_{n-1} + 2h^2a_{n-2} - \frac{9}{2}h^2a_{n-3} = 0\\
0 - \frac{1}{6}h^3a_{n-1} - \frac{4}{3}h^3a_{n-2} - \frac{9}{2}h^3a_{n-3} = 0
\end{aligned}
\end{equation}
$$

Which can also expressed as

$$
\begin{equation}
\begin{aligned}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
0 & -h & -2h & -3h \\
0 & \frac{1}{2}h^2  & 2h^2 & \frac{9}{2}h^2  \\
0 & -\frac{1}{6}h^3 & -\frac{4}{3}h^3 & -\frac{9}{2}h^3 
\end{pmatrix}
\cdot
\begin{pmatrix}
a_{n}\\
a_{n-1}\\
a_{n-2}\\
a_{n-3}
\end{pmatrix}
=
\begin{pmatrix}
0\\
1\\
0\\
0
\end{pmatrix}
\end{aligned}
\end{equation}
$$

Solving this system of linear equations results in

$$
a_{n} = \frac{11}{6h},\hspace{4mm}
a_{n-1} = -\frac{3}{h},\hspace{4mm}
a_{n-2} = \frac{3}{2h}, \hspace{4mm}
a_{n-3} = -\frac{1}{3h}
$$

These paramters can now be substiuted into Equation \ref{eq:bdfApproximationFirstDerivative}, giving us the approximation for $w_{n}'$ given the previous states $w_{n}$, $w_{n-1}$, $w_{n-2}$, and $w_{n-3}$.

$$
\begin{equation}
\begin{aligned}
\label{eq:bdfFirstOrderDerivative}
w_{n}'
&= \frac{11}{6h}w_{n} - \frac{3}{h}w_{n-1} + \frac{3}{2h}w_{n-2} - \frac{1}{3h}w_{n-3}\\
&= \frac{1}{h}(\frac{11}{6}w_{n} - 3w_{n-1} + \frac{3}{2}w_{n-2} - \frac{1}{3}w_{n-3})
\end{aligned}
\end{equation}
$$

We can rewrite Equation \ref{eq:bdfFirstOrderDerivative} using the gradient update rule from Equation \ref{eq:updateRule} so that we can compute $w_{n}'$ using the gradient information computed during the stochastic gradient descent optimization:

$$
\begin{equation}
\begin{aligned}
\label{eq:bdfFirstOrderDerivativeGradients}
w_{n}'
&= \frac{1}{h}(\frac{11}{6}w_{n} - \frac{18}{6}w_{n-1} + \frac{9}{6}w_{n-2} - \frac{2}{6}w_{n-3})\\
&= \frac{1}{6h}(11(w_{n} - w_{n-1}) - 7(w_{n-1}-w_{n-2}) + 2(w_{n-2}-w_{n-3}))\\
&= \frac{1}{6h}(11(-\eta \partial_{w_{n-1}}L) - 7(-\eta \partial_{w_{n-2}}L) + 2(-\eta \partial_{w_{n-3}}L))\\
&= -\frac{\eta}{6h}(11\partial_{w_{n-1}}L - 7\partial_{w_{n-2}}L + 2\partial_{w_{n-3}}L)
\end{aligned}
\end{equation}
$$

Now we have an expression for $w_{n}'$ that we can directly use in an optimizer that holds the current and past to gradients. The derivations for $w_{n}''$ and $w_{n}'''$ follow exactly the same pattern. Therefore, I will only roughly sketch these two derivations.

Solving the system of linear equations for $w_{n}''$ results in 

$$
a_{n} = \frac{2}{h^2},\hspace{4mm}
a_{n-1} = -\frac{5}{h^2},\hspace{4mm}
a_{n-2} = \frac{4}{h^2}, \hspace{4mm}
a_{n-3} = -\frac{1}{h^2}
$$

We substitute again these paramters into Equation \ref{eq:bdfApproximationFirstDerivative}, resulting in an approximation for $w_{n}''$ given the previous states $w_{n}$, $w_{n-1}$, $w_{n-2}$, and $w_{n-3}$.

$$
\begin{equation}
\begin{aligned}
\label{eq:bdfSecondOrderDerivative}
w_{n}''
&= a_{n}w_{n} + a_{n-1}w_{n-1} + a_{n-2}w_{n-2} + a_{n-3}w_{n-3}\\
&= \frac{2}{h^2}w_{n} - \frac{5}{h^2}w_{n-1} + \frac{4}{h^2}w_{n-2} - \frac{1}{h^2}w_{n-3}\\
&= \frac{1}{h^2}(2w_{n} - 5w_{n-1} + 4w_{n-2} - w_{n-3})\\
&= \frac{1}{h^2}(2(w_{n} - w_{n-1}) - 3(w_{n-1} + w_{n-2}) + (w_{n-2}-w_{n-3}))\\
&= -\frac{\eta}{h^2}(2\partial_{w_{n-1}}L - 3\partial_{w_{n-2}}L + \partial_{w_{n-3}}L)\\
\end{aligned}
\end{equation}
$$

Where I again used the gradient update rule from Equation \ref{eq:updateRule}. Finally, we are going to solve the system of linear equations for $w_{n}'''$ where we get

$$
a_{n} = \frac{1}{h^3},\hspace{4mm}
a_{n-1} = -\frac{3}{h^3},\hspace{4mm}
a_{n-2} = \frac{3}{h^3}, \hspace{4mm}
a_{n-3} = -\frac{1}{h^3}
$$

By substituting the parameters above into the standard ansatz one obtains

$$
\begin{equation}
\begin{aligned}
\label{eq:bdfThirdOrderDerivative}
w_{n}'''
&= a_{n}w_{n} + a_{n-1}w_{n-1} + a_{n-2}w_{n-2} + a_{n-3}w_{n-3}\\
&= \frac{1}{h^3}w_{n} - \frac{3}{h^3}w_{n-1} + \frac{3}{h^3}w_{n-2} - \frac{1}{h^3}w_{n-3}\\
&= \frac{1}{h^3}(w_{n} - 3w_{n-1} + 3w_{n-2} - w_{n-3})\\
&= \frac{1}{h^3}((w_{n} - w_{n-1}) - 2(w_{n-1} + w_{n-2}) + (w_{n-2}-w_{n-3}))\\
&= -\frac{\eta}{h^3}(\partial_{w_{n-1}}L - 2\partial_{w_{n-2}}L + \partial_{w_{n-3}}L)\\
\end{aligned}
\end{equation}
$$

By inserting Equation \ref{eq:bdfFirstOrderDerivative}, \ref{eq:bdfSecondOrderDerivative}, and \ref{eq:bdfThirdOrderDerivative} into Equation \ref{eq:thirdOrderExtrapolation} one obtains

$$
\begin{equation}
\begin{aligned} 
\label{eq:thirdOrderExtrapolationBDF}
w_{n+1} = w_{n} 
&- \frac{\Delta t\eta}{6h} (11\partial_{w_{n-1}}L - 7\partial_{w_{n-2}}L + 2\partial_{w_{n-3}}L)\\
&- \frac{\Delta t^2\eta}{2h^2} (2\partial_{w_{n-1}}L - 3\partial_{w_{n-2}}L + \partial_{w_{n-3}}L) \\
&- \frac{\Delta t^3\eta}{6h^3} (\partial_{w_{n-1}}L - 2\partial_{w_{n-2}}L + \partial_{w_{n-3}}L)
\end{aligned}
\end{equation}
$$

Since Equation \ref{eq:thirdOrderExtrapolationBDF} now contains only known quantities it can be used to perform extrapolation steps between gradient descent optimization steps.

---

we have to keep in mind, that Backward Differentiation Formula aren't "self-starting" methods like Runge-Kutta methods. This means, that some combination of other solvers are usually needed to generate the necessary state values before $w_{n+1}$ before using the higher-order BDF scheme.

<!--
## Implementation

We can bake this into an optimizer.

In the implementation, care must be taken that for a weight extrapolation of $N$th degree, $N$ optimization steps are waited for beforehand in order to calculate the first extrapolation step correctly. After that point there are two options when to perform extrapolation steps:

- After each optimization step. In this case the gradients used for the extrapolation are no longer connected.
- After every $N$th optimization steps. Performing an extrapolation step only every $N$ optimization steps has the advantage that the gradients used for weight extrapolation are not interrupted by the calculation of the extrapolation step (see [Figure 1](#fig:weightExtrapolation) above).

In the implementation below I used the second option to account for the interruptions introduced by the extrapolation steps.

```python
class Extrapolator(Optimizer):

    def __init__(self, params, eta, h, dt):
        defaults = dict(
            eta=eta,
            h=h,
            dt=dt,
        )
        super(Extrapolator, self).__init__(params, defaults)

        self.counter = 0

    def __setstate__(self, state):
        super(Extrapolator, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single extrapolation step."""
        self.counter += 1

        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue

                # Get gradients of parameters p
                d_p = p.grad.data

                eta = group["eta"]
                h = group["h"]
                dt = group["dt"]

                # Buffer gradients, initially same gradients for all states
                param_state = self.state[p]
                if 'grad_1' not in param_state and 'grad_2' not in param_state:
                    param_state['grad_1'] = torch.clone(d_p).detach()
                    param_state['grad_2'] = torch.clone(d_p).detach()
                else:
                    if self.counter > 3:
                        grad_1 = param_state['grad_1']
                        grad_2 = param_state['grad_2']

                        # Extrapolation step

                        # First order part of extrapolation step
                        grad = 11.0*d_p - 7.0*grad_1 + 2.0*grad_2
                        alpha = -((dt * eta) / (6.0 * h))
                        p.data.add_(grad, alpha=alpha)

                        # Second order part of extrapolation step
                        grad = 2.0*d_p - 3.0*grad_1 + grad_2
                        alpha = -((dt**2 * eta) / (2.0 * h**2))
                        p.data.add_(grad, alpha=alpha)

                        # Third order part of extrapolation step
                        grad = d_p - 2.0*grad_1 + grad_2
                        alpha = -((dt**3 * eta) / (6.0 * h**3))
                        p.data.add_(grad, alpha=alpha)

                # First in, first out gradient buffer
                param_state['grad_2'] = param_state['grad_1']
                param_state['grad_1'] = torch.clone(d_p).detach()

        return loss
```

## Experiments

For the experiments I trained two ResNet-18 convolutional neural network on the Imagewoof dataset using plain old stochastic gradient descent (SGD) without momentum. The baseline network's learning rate has been optimized using a simple grid search approach resulting in a learning rate of $0.02$ using a batch size of $64$. To compensate for the additional extrapolation steps, the baseline network has been trained for twice the number of epochs. The network equipped with weight extrapolation used extactly the same hyperparameters as the baseline model plus a weight extrapolation step size $dt = 1\mathrm{e}{-10}$.

## Results and Discussion

The results show a clear benefit coming from weight extrapolation compared to standard SGD. Not only is the test accuracy higher compared to the baseline model, but is also achieved after a shorter amount of time.

- **TODO: add two figure that compares accuracy as function of optimization step and time**

Weight extrapolation as described in this post might work better for lager batch sizes as these result in better gradient approximations potentially leading to a smoother trajectories through parameter space. Deep learning systems that are able to process very large batch sizes could thus particularly benefit from this method.

The Taylor series expansion allows for higher-order formulations of weight extrapolation that might or might not work as well. Reasons for that might be that the computation of higher-order terms can become numerically unstable. There might be a high volatility of $w(t)$ especially at the beginning of training making it difficult to extrapolate the weights. However, it is reasonable to assume that higher-order formulations can be used in later stages of the training where the gradient updates are potentially less volatile as in the beginning.

Smaller learning rates might also help the method of weight extrapolation itself, but at the same time could greatly slow down the training.

- Method allows to train same model faster!

## Conclusion

Training large neural networks can be prohibitively costly in terms of compute power. Therefore, accelerating the training of deep learning models is an important factor. In this post I tried to address this issue using a gradient-based weight extrapolation approach to speed up the learning process of machine learning models trained with gradient-based algorithms such as SGD.

Despite the simplicity of the method, adding extrapolation steps to the optimization process gives good results. The model equipped with weight extrapolation not only achieves the same or even better results in fewer steps, but also requires less time.

-->

---

- **TODO: Add to README.md of repository**
```bibtex
@misc{blogpost,
  title={Backward Weight Extrapolation},
  author={Fabi, Kai},
  howpublished={\url{https://kaifabi.github.io//WeightExtrapolation}},
  year={2021}
}
```

You find the code for this project [here][github_code].

<!-- Links -->
[github_code]: https://github.com/KaiFabi/WeightExtrapolation
