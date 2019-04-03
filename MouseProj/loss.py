import torch
import numpy as np

def IoU(input, target, cirrculum=0):
	'''
	Intersect over Union for bounding box regression:
	x, y, z: upper left corner
	dx, dy, dz: length
	N, d
	'''

	N = input.size()[0]

	eplison = 1e-5

	loss = 0

	for i in range(N):

		x  = target[i,0]
		dx = target[i,1]
		y  = target[i,2]
		dy = target[i,3]
		z  = target[i,4]
		dz = target[i,5]

		xhat  = input[i,0]
		dxhat = torch.abs (input[i,1])
		yhat  = input[i,2]
		dyhat = torch.abs (input[i,3])
		zhat  = input[i,4]
		dzhat = torch.abs (input[i,5])

		zero = torch.tensor(0, dtype=torch.float, device=torch.device('cuda'))
		ix = torch.min(xhat+dxhat, x+dx) - torch.max(xhat, x)
		iy = torch.min(yhat+dyhat, y+dy) - torch.max(yhat, y)
		iz = torch.min(zhat+dzhat, z+dz) - torch.max(zhat, z)

		dice_coeff = (2*ix/(dx+dxhat+eplison)+2*iy/(dy+dyhat+eplison)+2*iz/(dz+dzhat+eplison))/3

		loss += 1 - dice_coeff

	return loss/N

def dice_coeff_np(input, target):
	'''
	numpy dice coeff for single image (1, X, Y, Z)
	'''

	assert input.shape == target.shape

	Intersect = input*target

	return 2*np.sum(Intersect)/(np.sum(input) + np.sum(target))

def dice_coeff(input, target):
	'''
	Base soft dice coeff calculation for single class as described in Vnet paper
	input, target: (N, 1, X, Y ,Z)
	Ret:
		torch tensor of size() = [1]
	'''

	assert input.size() == target.size()

	eplison = 1e-6
	batch_size = input.size()[0]

	inmin, _ = torch.min(input.view(batch_size, -1), dim=1)
	inmax, _ = torch.max(input.view(batch_size, -1), dim=1)

	inmin = inmin.view(batch_size, 1, 1, 1, 1)
	inmax = inmax.view(batch_size, 1, 1, 1, 1)
	# reshape for broadcasting

	input_remap = (input-inmin)/(inmax-inmin+eplison)
	# remap the input to [0, 1]

	dice_coefficient = torch.sum(2*input_remap*target, dim=(2,3,4))/ \
			(torch.sum(input_remap*input_remap,dim=(2,3,4))+torch.sum(target*target,dim=(2,3,4))+eplison)
	# Average dice coeff over mini batch

	return torch.mean(dice_coefficient)


def dice_loss(input, target, channel):
	'''
	Single class soft dice loss with specified channel
	input: (N, C, X, Y, Z), channel < C
	'''
	return 1 - dice_coeff(input.narrow(1, channel, 1), target.narrow(1, channel, 1))


def dice_loss_2(input, target, cirrculum):
	'''
	Multi-class dice loss for body -> bv
	'''
	if cirrculum == 0:
	# Seg the body mask first
		return dice_loss(input, target, 1)

	else:
	# Seg the body and BV altogehter
		return (dice_loss(input, target, 1) + dice_loss(input, target, 2))/2


def dice_loss_3(input, target, cirrculum):
	'''
	Multi-class dice loss
	All conversion happens explicitly in the train function
	Args: 
		input: tensor of shape (N, 3, H, W, D), dtype = float
		target: ndarray of shape (N, 3, H, W, D) dtype = float
	Ret:
		loss: 1 - DiceCoefficient
	'''

	if cirrculum == 0:
		return dice_loss(input, target, 0)

	elif cirrculum == 1:
		return (dice_loss(input, target, 0) +\
				dice_loss(input, target, 1))/2

	else:
		return (dice_loss(input, target, 0) +\
				dice_loss(input, target, 1) +\
				dice_loss(input, target, 2))/3