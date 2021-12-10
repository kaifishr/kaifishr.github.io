---
layout: post
title: "Relevance Propagation with PyTorch"
---

**TL;DR**: A basic, unsupervised, yet reasonably fast implementation of Layer-wise Relevance Propagation (LRP) in PyTorch.

---
## Introduction

Layer-wise relevance propagation is a simple but powerful method that helps us to better understand the relevance of input features on the network's classification decision.

Not long ago I posted an implementation for [Layer-wise Relevance Propagation with Tensorflow][lrp_tensorflow] on my blog where I also went into some of the theoretical underpinnings of LRP.

This post presents a very basic and unsupervised implementation of Layer-wise Relevance Propagation ([Bach et al.][bach2015], [Montavon et al.][montavon2019]) in PyTorch for VGG networks from PyTorch's Model Zoo.

I used [this][montavon_gitlab] tutorial as a starting point for my implementation. I tried to make the code easy to understand but also easy to extend as this implementation is primarily intended to get you started with LRP.

I also added a novel relevance propagation filter to this implementation resulting in much crisper heat maps. If you want to use it, please don't forget to cite this implementation.


<p align="center"> 
<img src="/assets/images/post12/image_3.png" width="700"> 
<br>
<b>Figure 1:</b> An owl. </p>
{: #fig:lrpBird}

[Figure 1](#fig:lrpOwl) 


## Method

Starting at the ouput layer, layer-wise relevance propagation assigns relevance scores to each of the network's activations until the input is reached according to some relevance propagation rule.

How the relevance of each neuron is computed is described by the following formula:

\begin{equation} \label{eq:zplus}
    R_i^{(l)} = \sum_j \frac{x_i w_{ij}^+}{\sum_{i'} x_{i'} w_{i'j}^+} R_j^{(l+1)}
\end{equation}

Equation \eqref{eq:zplus} describes relevance distribution in fully connected layers. Here, $R_i^{(l)}$ and $R_j^{(l+1)}$ represent the relevance scores of neuron $i$ and $j$ in layers $(l)$ and $(l+1)$, respectively. $x_i$ represents the $i$th neuron's activation. $w_{ij}^+$ stands for the positive weights connecting the neurons $i$ and $j$ of layers $(l)$ and $(l+1)$.

This formula is also know as the $z^+$-rule. Basically this formula describes that the contribution of a single neuron to the total activation mass of a layer is decisive for its relevance.

<p align="center"> 
<img src="/assets/images/post12/image_0.png" width="300">
<img src="/assets/images/post12/image_1.png" width="300">
<img src="/assets/images/post12/image_2.png" width="300">
<img src="/assets/images/post12/image_3.png" width="300">
</p>


{:refdef: style="text-align: center;"}
![](/assets/images/post8/lrp_network.png)
{: refdef}


## Implementation

The presentation implementation of LRP is completely unsupervised. This means, that we do not use the input's ground truth label as the starting point for the relevance propagation. 

At least in my tests, I found that starting relevance propagation from the true label (setting the class' output activation and therefore the relevance to 1) had virtually no effect on the resulting heatmap.


## Relevance Filter

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
