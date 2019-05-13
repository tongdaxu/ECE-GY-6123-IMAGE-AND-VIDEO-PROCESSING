import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import affine_transform, zoom

from loss import dice_loss, dice_coeff_np
from niiutility import *
from dataset import upSampleFun
from scipy.ndimage import affine_transform, zoom, gaussian_filter

def weights_init(m):
	if type(m) == nn.Conv3d:
		nn.init.kaiming_normal_(m.weight)
		m.bias.data.zero_()
	elif type(m) == nn.Linear:
		# the default init for linear loos ok?
		nn.init.xavier_normal_(m.weight.data)
		nn.init.normal_(m.bias.data) # is that a good idea to init the bias as well?
		pass
	else:
		# 
		pass

def shape_test(model, device, dtype, lossFun, shape):
	
	sx, sy, sz = shape

	x = torch.randn((24, 3, sx, sy, sz), dtype=dtype, requires_grad=True)
	y = torch.ones((24, 3, sx, sy, sz), dtype=dtype, requires_grad=True)

	# model = model.to(device=device)
	# scores = model(x)
	# print(scores.size())
	loss = lossFun(x, y, cirrculum=2)

def loadckp (model, optimizer, scheduler, logger, filename, device):
	'''
	model: model structure to load
	'''
	assert (os.path.isfile(filename))
	
	print("loading checkpoint '{}'".format(filename))

	model = model.to(device=device)    
	checkpoint = torch.load(filename)
	start_epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	scheduler.load_state_dict(checkpoint['scheduler'])
	logger = checkpoint['logger']

	print("loaded checkpoint '{}' (epoch {})"
			  .format(filename, checkpoint['epoch']))

def train(model, traindata, valdata, optimizer, scheduler, device, dtype, lossFun, logger, epochs=1, startepoch=0, usescheduler=False):
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
	N = len(traindata)
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
			
		print('Epoch {0} finished ! Training Loss: {1:.4f}'.format(e + startepoch, epoch_loss/N))
		
		# Get validation loss
		loss_val = check_accuracy(model, valdata, device, dtype, 
			cirrculum=cirrculum, lossFun=lossFun)
		
		logger['train'].append(epoch_loss/N)
		logger['validation'].append(loss_val)

		# Taking a scheduler step on validation loss
		if usescheduler:
			scheduler.step(loss_val)
		# else pass

		# When validation loss < 0.1,upgrade cirrculum, reset scheduler
		if loss_val < 0.1 and cirrculum <= 2:
			cirrculum += 1
			scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
			print('Change Currculum! Reset LR Counter!')

		if (e+startepoch)%50 == 0:
			
			model_save_path = 'checkpoint' + str(datetime.datetime.now())+'.pth'
			state = {'epoch': e+startepoch + 1, 'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'logger': logger}
			torch.save(state, model_save_path)
			print('Checkpoint {} saved !'.format(e+startepoch + 1))

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

		print('     validation loss = {0:.4f}'.format(loss/N))
		return loss/N

def check_img(model, dataloader, device, dtype, cirrculum, lossFun, data_index):
	model.eval()  # set model to evaluation mode

	with torch.no_grad():

		N = len(dataloader)
		bodyDice = 0
		bvDice = 0

		for t, batch in enumerate(dataloader):

			image_index = data_index[t]

			x = batch['image']
			y = batch['label']
			
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=dtype)
			
			mask_predict = model(x)

			loss = lossFun(mask_predict, y, cirrculum=cirrculum)

			# Show and save image

			batch_size = x.size()[0]
			x = x.cpu()
			y = y.cpu()

			mask_predict = mask_predict.cpu()
			
			show_batch_image(x,y,batch_size)
			show_batch_image(x,mask_predict,batch_size)
			
			mask_predict_bd = upSampleFun(mask_predict.numpy()[0,1:2], 2, 3)
			mask_predict_bv = upSampleFun(mask_predict.numpy()[0,2:3], 2, 3)

			mask_predict_bd = mask_predict_bd.squeeze(axis=0)
			mask_predict_bv = mask_predict_bv.squeeze(axis=0)

			# mask_predict_bd = gaussian_filter(mask_predict_bd, sigma=1)
			# mask_predict_bv = gaussian_filter(mask_predict_bv, sigma=3)

			mask_predict_bd = (mask_predict_bd - np.min(mask_predict_bd)) / (np.max(mask_predict_bd) - np.min(mask_predict_bd))
			mask_predict_bv = (mask_predict_bv - np.min(mask_predict_bv)) / (np.max(mask_predict_bv) - np.min(mask_predict_bv))

			mask_predict_bd = (mask_predict_bd > 0.75).astype(np.float32)
			mask_predict_bv = (mask_predict_bv > 0.5).astype(np.float32)

			shape = getniishape(image_index)
			_, label = loadnii(image_index)

			mask_predict_bd = zero_padding(mask_predict_bd, shape[0], shape[1], shape[2])
			mask_predict_bv = zero_padding(mask_predict_bv, shape[0], shape[1], shape[2])

			mask_bv = (label > 0.75).astype(np.float32)
			mask_bd = (label > 0.25).astype(np.float32) - mask_bv

			dice_bv = dice_coeff_np(mask_bv, mask_predict_bv.reshape(1, *mask_predict_bv.shape))
			dice_bd = dice_coeff_np(mask_bd, mask_predict_bd.reshape(1, *mask_predict_bd.shape))

			print('image {0} loss is {1:.4f} \
				body dice coeff is {2:.4f}, bv dice coeff is {3:.4f}'\
				.format(image_index, loss, dice_bd, dice_bv))

			bvDice += dice_bv
			bodyDice += dice_bd

			#savenii(mask_predict_bd,str(image_index) + '_body')
			#savenii(mask_predict_bv,str(image_index) + '_bv')
			
			pass

		print('average body dice is {0:.4f}, average bv dice is {1:.4f}'.format(bodyDice/N, bvDice/N))


