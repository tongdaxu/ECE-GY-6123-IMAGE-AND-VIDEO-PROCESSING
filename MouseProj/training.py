import torch
import torch.nn as nn
import torch.nn.functional as F

def shape_test(model, localdevice, localdtype):
    
    x = torch.zeros((1, 1, 64, 64, 64), dtype=localdtype)
    model = model.to(device=localdevice)
    scores = model(x)
    print(scores.size())

def overfit_test(model, localdevice, localdtype, optimizer, x, y, epochs=1):
    """    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """

    step = epochs//20 # 20 loss message only

    model = model.to(device=localdevice)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        model.train()  # put model to training mode
        x = x.to(device=localdevice, dtype=localdtype)  # move to device, e.g. GPU
        y = y.to(device=localdevice, dtype=torch.long)

        scores = model(x)
        loss = F.cross_entropy(scores, y)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()

        if e%step == 0:
            print('epoch %d, loss = %.4f' % (e, loss.item()))