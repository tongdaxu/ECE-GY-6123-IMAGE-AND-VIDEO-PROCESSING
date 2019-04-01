import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import affine_transform, zoom

from loss import dice_loss
from niiutility import *
from dataset import upSampleFun

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def shape_test(model, device, dtype, lossFun, shape):
    
    sx, sy, sz = shape

    x = torch.randn((24, 3, sx, sy, sz), dtype=dtype, requires_grad=True)
    y = torch.ones((24, 3, sx, sy, sz), dtype=dtype, requires_grad=True)

    # model = model.to(device=device)
    # scores = model(x)
    # print(scores.size())
    loss = lossFun(x, y, cirrculum=2)

def loadckp (model, optimizer, scheduler, filename, device):
    model = model.to(device=device)
    if os.path.isfile(filename):
        print("loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler


def train(model, traindata, valdata, optimizer, scheduler, device, dtype, lossFun, epochs=1):
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
        # scheduler.step(loss_val)
           
        # When validation loss < 0.1,upgrade cirrculum, reset scheduler
        if loss_val < 0.1 and cirrculum <= 2:
            cirrculum += 1
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
            print('Change Currculum! Reset LR Counter!')

        if e%50 == 0:
            model_save_path = 'checkpoint' + str(datetime.datetime.now())+'.pth'
            state = {'epoch': e + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, model_save_path)
            print('Checkpoint {} saved !'.format(e + 1))
        
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



def check_img(model, dataloader, device, dtype, cirrculum, lossFun):
    model.eval()  # set model to evaluation mode

    with torch.no_grad():

        N = len(dataloader)
        for t, batch in enumerate(dataloader):

            x = batch['image']
            y = batch['label']
            
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            
            mask_predict = model(x)

            loss = lossFun(mask_predict, y, cirrculum=cirrculum)
            print('loss = %.4f' % loss)

            # Show and save image

            batch_size = x.size()[0]
            x = x.cpu()
            y = y.cpu()

            mask_predict = mask_predict.cpu()
            
            show_batch_image(x,y,batch_size)
            show_batch_image(x,mask_predict,batch_size)
            
            mask_predict_resize = upSampleFun(mask_predict.numpy()[0,1:2], 2, 0)
            mask_predict_resize = mask_predict_resize.squeeze(axis=0)
            mask_predict_resize = (mask_predict_resize > 0.5).astype(np.float32)
            shape = getniishape(t)
            mask_predict_resize = zero_padding(mask_predict_resize, shape[0], shape[1], shape[2])
            
            savenii(mask_predict_resize, str(t))
            
            pass



