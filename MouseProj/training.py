import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import dice_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

filename = 'vnet-mask-1'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def shape_test(model, localdevice, localdtype):
    
    x = torch.zeros((1, 1, 96, 128, 128), dtype=localdtype)
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
        x = x.to(device=localdevice, dtype=np.float)  # move to device, e.g. GPU
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


def train(model, traindata, valdata, optimizer, device, dtype, lossFun=dice_loss, epochs=1, print_every=1e8):
    """
    Train a model with an optimizer
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    cirrculum = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    
    for e in range(epochs):
        print('epoch %d begins: ' % (e))
        for t, batch in enumerate(traindata):
            model.train()  # put model to training mode
            x = batch['image']
            y = batch['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            scores = model(x)
            loss = lossFun(scores, y, cirrculum)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
            if t%print_every == 0:
                print('     Iteration %d, loss = %.4f' % (t, loss.item()))
        
        loss_val = check_accuracy(model, valdata, device, dtype, 
            cirrculum_index=(cirrculum), lossFun=lossFun)
        scheduler.step(loss_val)
           
        # When validation loss < 0.1,upgrade cirrculum, reset scheduler
        if loss_val < 0.1 and cirrculum <= 2:
            cirrculum += 1
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
            print('Change Currculum! Reset LR Counter!')

        if e%50 == 0:
            state = {'epoch': e + 1, 'state_dict': model.state_dict(),\
             'optimizer': optimizer.state_dict()}
            torch.save(state, filename)


def check_accuracy(model, dataloader, device, dtype, cirrculum_index, lossFun):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        loss = 0
        N = len(dataloader)
        for t, batch in enumerate(dataloader):
            x = batch['image']
            y = batch['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x)

            loss += lossFun(scores, y, cirrculum_index)

        print('     validation loss = %.4f' % (loss/N))
        return loss/N



