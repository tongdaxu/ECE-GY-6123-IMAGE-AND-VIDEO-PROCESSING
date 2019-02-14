import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = W.shape[0]

  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:

        dW[:, j] += np.transpose (X[i, :])
        dW[:, y[i]] -= np.transpose (X[i, :])

        #then the effect over dW is not zero!

        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  dW /= num_train
  dW += reg * (2 * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = W.shape[0]

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  scores = X.dot(W) #score matrix of shape [N,C]
  correct_class_score = scores[np.arange(num_train),y[np.arange(num_train)]]

  scores -= np.reshape(correct_class_score - 1, (num_train, 1))

  #hinge = np.vectorize(lambda x: max (0, x))
  #scores = hinge (scores)
  scores *= (scores>0)

  #finally set the j = y[j] set to 0, leave them alone at first
  scores[np.arange(num_train),y[np.arange(num_train)]] = 0

  loss = np.sum(scores)/num_train
  loss += reg * np.sum(W * W)
 
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  hinge_b = np.vectorize (lambda x: (x>0))
  scores_b = hinge_b(scores)

  dW += np.transpose(np.dot(np.transpose(scores_b), X))

  scores_v = np.dot(scores_b, np.ones(num_classes))
  correct_gradient = (scores_v.reshape(scores_v.shape[0],1)*X)
  
  #dW[:, y] -= correct_gradient.T
  for i in range(num_train):
    dW[:, y[i]] -= correct_gradient[i] 

  dW /= num_train
  dW += reg * (2 * W)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
