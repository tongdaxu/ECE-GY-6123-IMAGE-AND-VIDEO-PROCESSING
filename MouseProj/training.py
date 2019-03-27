import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import dice_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from niiutility import *

filename = 'vnet-mask-01'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def shape_test(model, localdevice, localdtype, shape):
    
    sx, sy, sz = shape

    x = torch.zeros((1, 1, sx, sy, sz), dtype=localdtype)
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


def train(model, traindata, valdata, optimizer, device, dtype, lossFun=dice_loss, epochs=1):
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=30, verbose=True)
    
    for e in range(epochs):
        epoch_loss = 0
        for t, batch in enumerate(traindata):
            model.train()  # put model to training mode
            x = batch['image']
            y = batch['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            scores = model(x)
            loss = lossFun(scores, y, cirrculum=cirrculum)

            # avoid gradient
            epoch_loss += loss.item()

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
        print('Epoch {0} finished ! Training Loss: {1}'.format(e, epoch_loss / t))
        
        loss_val = check_accuracy(model, valdata, device, dtype, 
            cirrculum=cirrculum, lossFun=lossFun)
        scheduler.step(loss_val)
           
        # When validation loss < 0.1,upgrade cirrculum, reset scheduler
        if loss_val < 0.1 and cirrculum <= 2:
            cirrculum += 1
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
            state = {'epoch': e + 1, 'state_dict': model.state_dict(),\
             'optimizer': optimizer.state_dict()}
            torch.save(state, filename)
            print('Change Currculum! Reset LR Counter!')


def check_accuracy(model, dataloader, device, dtype, cirrculum, lossFun):
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

            loss += lossFun(scores, y, cirrculum=cirrculum)

        print('     validation loss = %.4f' % (loss/N))
        return loss/N

def check_img(model, dataloader, device, dtype):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        loss = 0
        N = len(dataloader)
        for t, batch in enumerate(dataloader):
            x = batch['image']
            y = batch['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            mask_predict = model(x)
            batch_size = x.size()[0]
            
            x = x.cpu()
            y = y.cpu()
            mask_predict = mask_predict.cpu()
            
            show_batch_image(x,y,batch_size, level=4)
            show_batch_image(x,mask_predict,batch_size, level=4)

        print('     validation loss = %.4f' % (loss/N))
        return loss/N



