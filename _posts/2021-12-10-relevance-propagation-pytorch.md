---
layout: post
title: "Layer-wise Relevance Propagation with PyTorch"
---

**TL;DR**: A basic, unsupervised, but reasonably fast implementation of Layer-wise Relevance Propagation (LRP) in PyTorch.

---
## Introduction

Not long ago I posted an implementation for [Layer-wise Relevance Propagation with Tensorflow][lrp_tensorflow] on my blog where I also went into some of the theoretical underpinnings of LRP.

This post presents a very basic and unsupervised implementation of Layer-wise Relevance Propagation ([Bach et al.][bach2015], [Montavon et al.][montavon2019]) in PyTorch for VGG networks from PyTorch's Model Zoo. 

I used [this][montavon_gitlab] tutorial as a starting point for my implementation. I tried to make the code easy to understand but also easy to extend as this implementation is primarily intended to get you started with LRP.

I also added a novel relevance propagation filter to this implementation resulting in much crisper heat maps. If you want to use it, please don't forget to cite this implementation.


<p align="center"> 
<img src="/assets/images/post12/image_1.png" width="700"> 
<br>
<b>Figure 1:</b> bla </p>
{: #fig:weightExtrapolation}

[Figure 1](#fig:weightExtrapolation) 

## Method

$$
w(t) = \sum_{k=0}^{\infty} \frac{w^{(k)}(t)}{k!}\bigg\rvert_{t=t_{0}}(t-t_{0})^k
$$

[differences quotient][wiki_difference_quotient]


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


```python
class Bla(Optimizer):
    r"""Light extrapolator algorithm.
        
    """
```

| Function | Rosenbrock | Beale | Goldstein-Price |
|:--------:|:----------:|:-----:|:---------------:|
| Learning rate | 1e-1 | 1e-2 | 1e-3 |
| $\Delta t$ | 1e-4 | 1e-4 | 1e-5 |


<p align="center"> 
<img src="/assets/images/post12/image_0.png" width="700">
<img src="/assets/images/post12/image_1.png" width="700">
<img src="/assets/images/post12/image_2.png" width="700">
<img src="/assets/images/post12/image_3.png" width="700">
</p>


### To run

Running LRP for a VGG-like network is fairly straightforward

```python
import torch
import torchvision
from src.lrp import LRPModel

x = torch.rand(size=(1, 3, 224, 224))
model = torchvision.models.vgg16(pretrained=True)
lrp_model = LRPModel(model)
r = lrp_model.forward(x)
```

### Relevance Filter

I have found that a very effective way of directing relevance scores to important features in input space is by using a filter that allows only the highest k % of relevance scores to pass to the next layer. Such a filter method results in much crisper heatmaps supporting the idea of suppressing very small and therefore noisy relevance scores. However, sorting the relevance scores can make this filter very expensive. Especially for convolutional layers.

The code below shows how the idea was implemented in PyTorch.

```python
def relevance_filter(r: torch.tensor, top_k_percent: float = 1.0) -> torch.tensor:
    """Filter that allows largest k percent values to pass for each batch dimension.

    Filter keeps k % of the largest entries of a tensor. All tensor elements are set to
    zero except for the largest k % values.

    Args:
        r: Tensor holding relevance scores of current layer.
        top_k_percent: Proportion of top k values that is passed on.

    Returns:
        Tensor of same shape as input tensor.

    """
    assert 0.0 <= top_k_percent <= 1.0

    if top_k_percent < 1.0:
        size = r.size()
        r = r.flatten(start_dim=1)
        num_elements = r.size(-1)
        k = int(top_k_percent * num_elements)
        assert k > 0, f"Expected k to be larger than 0."
        top_k = torch.topk(input=r, k=k, dim=-1)
        r = torch.zeros_like(r)
        r.scatter_(dim=1, index=top_k.indices, src=top_k.values)
        return r.view(size)
    else:
        return r
```


## Examples

Some example images.

## Benchmark

This LRP implementation is already reasonably fast. It should therefore also be possible to use this code in projects where it is intended to work in real time. Without relevance filter and with an RTX 2080 Ti graphics card I reach 53 FPS with the VGG-16.


## Outlook

There are some open points how the implementation can be further imporved. First, the implementation should be more model agnostic. Here, implementing all network operations using the gradient trick would be an important step in this direction.Second, one would have to think about how to get a list with the activations of all operations of the original network. I tried using forward hooks but was not able to extract the activations if a torch function such as `torch.relu`, `torch.flatten`, etc., was called during the forward pass.


## Citation

```bibtex
@misc{blogpost,
  title={Layer-wise Relevance Propagation for PyTorch},
  author={Fabi, Kai},
  howpublished={\url{https://kaifabi.github.io/2021/02/02/relevance-propagation-pytorch.html}},
  year={2021}
}
```


---


You find the code for this project [here][github_code].

<!-- Links -->
[bach2015]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
[montavon2019]: https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
[montavon_gitlab]: https://git.tu-berlin.de/gmontavon/lrp-tutorial
[github_code]: https://github.com/KaiFabi/RelevancePropagationPyTorch
[lrp_tensorflow]: https://kaifabi.github.io/2021/01/20/layer-wise-relevance-propagation.html
