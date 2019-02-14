import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    #for each x_i
    S = np.dot(X[i,:], W) #shape = (num_classes)
    exp_S = np.exp(S)
    P = np.exp (S[y[i]])/np.sum(exp_S)

    loss -= np.log(P)
    dW[:,y[i]] -= X[i,:]

    for j in range(num_class):
      dW[:,j] += (exp_S[j]/np.sum(exp_S))*X[i,:]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * (2 * W)

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = np.dot(X, W)
  exp_S = np.exp(S)
  norm_S = exp_S/np.sum(exp_S, axis = 1).reshape(S.shape[0], 1)

  for i in range(num_train):
    dW[:,y[i]] -= X[i,:]

  P = np.exp(S[np.arange(num_train), y[np.arange(num_train)]])/np.sum(exp_S, axis=1)
  
  #calculating sum
  loss -= np.sum (np.log(P))
  loss /= num_train
  loss += reg * np.sum(W * W)

  #calculating gradient
  dW += np.dot(X.T, norm_S)
  
  #do not know how to avoid this for loop

  dW /= num_train
  dW += reg * (2 * W)

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

