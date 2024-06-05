#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        # Initialize the learnable parameters
        self.params={}
        self.params['W1']=weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params['b1']=np.zeros(hidden_dim)
        self.params['W2']=weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2']=np.zeros(num_classes)
    
    def parameters(self):
        params = None
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return self.params
    
    def forward(self, X):
        scores, cache = None, None
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        # Forward pass
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        hidden_layer, hidden_cache = fc_forward(X, W1, b1)
        hidden_layer_relu, relu_cache = relu_forward(hidden_layer)
        scores, scores_cache = fc_forward(hidden_layer_relu, W2, b2)
        
        # Store data needed for backward pass
        cache = (hidden_cache, relu_cache, scores_cache)
        
        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        # Unpack cache
        hidden_cache, relu_cache, scores_cache = cache
        
        # Backward pass
        grad_hidden_relu, grad_W2, grad_b2 = fc_backward(grad_scores, scores_cache)
        grad_hidden = relu_backward(grad_hidden_relu, relu_cache)
        grad_X, grad_W1, grad_b1 = fc_backward(grad_hidden, hidden_cache)
        
        grads = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2}
        
        return grads


# In[ ]:




