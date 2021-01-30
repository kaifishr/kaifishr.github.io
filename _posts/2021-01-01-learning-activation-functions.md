---
layout: post
title: "[DRAFT] Learning Activation Functions"
date:   2021-01-20 21:32:04
---

**TL;DR**: This post presents an implementation and evaluation of multilayer perceptron-based activation functions.

---


# TODO:

- Create new github repository

## Introduction 

In the area of machine learning, modern network architectures often use static activation functions such as the ReLU nonlinearity. This means that in practice we usually presume the most ideal functional form of the activation function by using predefined non-trainable nonlinearities. Due to the fixed mapping of these activation functions, there is a clear rule which signals and at what strength are allowed to pass to the next layer. However, this does not allow the network to be as flexible as it could be if the activation functions were trainable. 

This post tries to explore some implications of replacing non-trainable activation functions by small fully connected neural networks. Multilayer perceptrons have been shown to be strong function approximators. In this context, this could prove very useful to learn more complex activation functions for better feature representation.

Throughout this post I will refer to these trainable activation networks as subnetworks.

## Related Work

The idea of replacing activation functions by smaller subnetworks is not a new one. [Min Lin et al.][MinLin2014] have shown that convolutional neural networks equipped with these trainable activation functions have the potential to outperform classic neural network architectures.

One drawback of the paper, however, is that it is not really clear whether the performance increase mentioned is due to learnable activation functions or the choice of hyperparameters they made. To allow for a better comparison, this post will use the same network architectures as well as the same set of hyperparameters and only the type of activation function will be exchanged.

## Method

The following figure shows the basic idea of a fully connected neural networks equipped with subnetworks where static activation functions $h(z)$ are replaced by subnetworks $s^{(l)}_i = s(z_i; w^{(l)})$. The green color in the figure below is to indicate that there is only one subnetwork per layer with shared weights for each neuron. Subnetworks are small fully connected neural networks that are no different from ordinary densely connected networks. For computational reasons, these subnetworks consist only of a small number of neurons and a few hidden layers. Since subnetworks effectively make the network deeper, it makes sense to choose activation functions for the subnetwork that do not saturate too quickly and that are well permeable for the feedforward signal as well as the error signal during the backpropagation pass.

{:refdef: style="text-align: center;"}
![](/assets/images/post9/subnet-4.png)
{: refdef}

The implementation presented in this post uses one subnetwork for each layer (see also figure above). This is a good compromise between a subnetwork for all layers and a subnetwork for each individual neuron.

Small subnetworks consisting of two hidden layers with four hidden neurons as presented above adds only a negligible number of parameters to the network. However, it should be noted that the computational effort increases dramatically since the network’s number of activations increases eightfold.

Since all pre-activations $z$ of a given layer are being processed by the same subnetwork $s$, it is important to ensure that the subnetwork allows a fair amount of the incoming signals to pass. This is not always the case for subnetworks using ReLU nonlinearity. If subnetworks have only a few neurons, there is a high probability that these will not be activated if for example large negative biases are learned during gradient descent. This can cause the signal transmission of the entire layer to break down during the feedforward process. Once the subnetwork ends up in this state, it is unlikely to recover, since the function’s gradient at 0 is also 0. Thus, gradient descent will not change the subnetwork’s weights anymore. One attempt to address this issue and to allow the subnetwork to recover from such a state is to equip them for example with leaky ReLUs that have a small positive gradient for negative inputs.

## Implementation

This section will briefly describe how subnetworks can be implemented using Tensorflow. For more detailed information about the implementation, please see [here][github_subnetwork].

### Subnetwork

Replacing activation functions with an entire fully connected network is a fairly simple process with Tensorflow by overloading the layer class. This means that instead of using a simple nonlinearity, we substitute the activation function by a network that is treated as an additional layer with trainable weights and biases.

In practice in case of fully connected layers the input tensor with dimensions `[batch_size, n_features]` is reshaped into a single vector with dimension `[batch_size * n_features]`. The same also applies for convolutional layers where we reshape the input tensor’s dimensions `[batch_size, feature_map_height, feature_map_width, feature_map_channels]` into `[batch_size * feature_map_height * feature_map_width * feature_map_channels]` to be processed by the subnetwork. This approach treats every pre-activation as an independent data point. After processing the pre-activations, the subnetwork’s output will be reshaped back to the original dimensions to be processed by the next layer. 

The core principle can be implemented as follows

```python
class SubNetwork(tf.keras.layers.Layer):

   def __init__(self, conf):
       super(SubNetwork, self).__init__()
       . . .
       self.kernel = None
       self.bias = None

   def build(self, _):
       . . .

   def call(self, input_tensor):
       # Reshape input
       x = tf.reshape(tensor=input_tensor, shape=(-1, 1))

       # Feedforward pre-activation through network
       for kernel, bias in zip(self.kernel[:-1], self.bias[:-1]):
           x = tf.matmul(x, kernel) + bias
           x = self.activation_function(x)
       x = tf.matmul(x, self.kernel[-1]) + self.bias[-1]

       # Reshape back to input shape and return
       x = tf.reshape(tensor=x, shape=tf.shape(input_tensor))
       return x
```

Here, we receive `input_tensor` consisting of the network’s pre-activations which are reshaped into a single vector. The pre-activations come either from feature maps of convolutional layers or from fully connected layers.

### Network

In order to test models equipped with subnetworks against standard networks, I wrote one class to build fully connected neural networks and another to create VGG-like networks. For both models it applies that if `use_subnet` is `True`, only the activation function will be replaced by trainable subnetworks.

#### MLP

Given a list describing the network’s layer structure and which activation function to use, the following piece of code creates a fully connected model using Keras’ functional API.

```python
class MLP(object):

def __init__(self, conf):
   self.conf = conf
   # . . .                                                           

def build(self):
   inputs = tf.keras.layers.Input(shape=self.input_shape)
   x = tf.keras.layers.Dropout(rate=0.5)(inputs)

   if self.use_subnet:
       z = tf.keras.layers.Dense(units=self.layers_dense[0], **self.dense_config)(x)
       x = SubNetwork(self.conf)(z) #+ z
   else:
       x = tf.keras.layers.Dense(units=self.layers_dense[0], activation=self.activation_function,
                                 **self.dense_config)(x)
   x = tf.keras.layers.Dropout(rate=0.5)(x)

   for units in self.layers_dense[1:]:
       if self.use_subnet:
           z = tf.keras.layers.Dense(units=units, **self.dense_config)(x)
           x = SubNetwork(self.conf)(z) #+ z
       else:
           x = tf.keras.layers.Dense(units=units, activation=self.activation_function, **self.dense_config)(x)
       x = tf.keras.layers.Dropout(rate=0.5)(x)

   outputs = tf.keras.layers.Dense(self.n_classes)(x)
   model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp")
   return model
```

#### CNN

The convolutional neural network class follows the same logic and creates a model using Keras’ functional API. Two lists describe the network’s convolutional and fully connected layer structure. An additional parameter defines which activation functions will be used if `use_subnet` evaluates to `False`.

```python
class CNN(object):

   def __init__(self, conf):
       # . . .

   def build(self):
       inputs = tf.keras.layers.Input(shape=self.input_shape)
       x = inputs

       # Convolutional part
       for filters in self.units_conv:
           x = self.conv_block(x, filters)
       x = tf.keras.layers.Flatten()(x)

       # Dense part
       for units in self.units_dense:
           x = tf.keras.layers.Dense(units=units, **self.conf_dense)(x)
           if self.use_subnet:
               x = SubNetwork(self.conf)(x)
           else:
               x = self.activation_function(x)
       x = tf.keras.layers.Dense(units=self.n_classes, **self.conf_dense)(x)

       model = tf.keras.models.Model(inputs=inputs, outputs=x)
       return model

   def conv_block(self, inputs, filters):
       x = tf.keras.layers.Conv2D(filters=filters, **self.conf_conv)(inputs)
       if self.use_subnet:
           x = SubNetwork(self.conf)(x)
       else:
           x = self.activation_function(x)

       x = tf.keras.layers.Conv2D(filters=filters, **self.conf_conv)(x)
       if self.use_subnet:
           x = SubNetwork(self.conf)(x)
       else:
           x = self.activation_function(x)

       x = tf.keras.layers.MaxPool2D(**self.conf_pool)(x)

       return x
```

## Experiments

There are many aspects that make subnetworks very interesting. Here, two of these aspects will be examined in detail: How does the use of subnetworks affect the model’s performance and what kind of activation functions are actually learned by the subnetworks?

For the experiments two simple network types are used. A fully connected neural network and a VGG-like convolutional neural network. Both networks consist of eight hidden layers. 

The network specifications for the MLP are as follows:

```yaml
network:
 type: "mlp"
 units_dense: [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
 activation_function: "leaky_relu"
```

The following parameters were selected for the convolutional neural network

```yaml
network:
 type: "cnn"
 units_conv: [32, 64, 128]
 units_dense: [256, 256]
 activation_function: "leaky_relu"
```

For both networks above the same subnetwork has been used:

```yaml
subnetwork:
 use_subnet: True
 units: [1, 8, 8, 8, 1]
 activation_function: "leaky_relu"
```

Both networks have been trained on the Fashion-MNIST dataset for 300 epochs at a constant learning rate of 1e-4 using the Adam optimizer and a batch size of 1024. For the experimental evaluation, the results of 10 runs were averaged.

## Results

It is immediately noticeable that the networks’ loss and accuracy are less volatile if equipped with standard activation functions. 

The picture is different for network’s equipped with subnetworks. Here the variance of loss and accuracy between several runs is much higher. It is interesting to see that in some cases these networks significantly outperform the baseline model.

It is notable that after 300 epochs in all cases, models equipped with subnetworks dominate in terms of training accuracy and training loss.

Figure ? shows the resulting graphs of individual subnetworks in different layers. The functions’ definition range was chosen such that about 99.7% of the pre-activations entering the subnetwork are within three standard deviations.

Figure ? shows that there is no emerging pattern for the subnetwork in the same layer. Very different activation functions are learned during each run. Furthermore, it seems that there aren’t any noticeable differences between the functions learned by the subnetwork in different layers.

## Discussion

The results show that using subnetworks rather than standard activation functions have the potential to outperform classic network architectures. This could be due to the fact that trainable activation functions may increase the network’s overall ability of feature representations.

A major disadvantage of subnetworks is, of course, the large amount of computation required to train these kinds of networks. Depending on the network type in the experiments above, networks needed two to three times as much time to train. 
At this point it also remains unknown what the ideal size of a subnetwork with respect to the parent network is, which activation function should be used, and what a good initialization scheme for the subnetwork’s weight is.

Since adding subnetworks to a model basically adds many additional layers to the network, these kinds of networks are no longer easy to train. This problem could be addressed by adding shortcut connections that bypass the subnetworks.

It is also notable that subnetworks only increase the network’s performance if they are combined with dropout. This suggests that other regularization techniques may also have a strong influence on the performance growth that is added by subnetworks.

It is also interesting to note that networks equipped with subnetworks can be trained with larger learning rates. This could be due to the fact that subnetworks lead to smaller gradients more quickly, which is possibly compensated for by larger learning rates.

## Outlook

Despite the increased computational complexity of adding subnetworks to neural networks, the results provide evidence that these kinds of networks can easily outperform classical network architectures, where high-speed inference is not essential.

I hope to see future studies where subnetworks are used in much larger networks, as this is probably the only way to determine their true value.

---

You find the code for this project [here][github_subnetwork].

<!-- Links -->

[github_subnetwork]: https://github.com/KaiFabi/Subnetwork
[MinLin2014]: https://arxiv.org/pdf/1312.4400.pdf
