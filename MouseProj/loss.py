import torch

def dice_loss_single(input, target, cirrculum):
	'''
	Multi-class dice loss
	All conversion happens explicitly in the train function
	Args: 
		input: tensor of shape (N, 3, H, W, D), dtype = float
		target: ndarray of shape (N, 3, H, W, D) dtype = float
	Ret:
		loss: 1 - DiceCoefficient
	'''

	eplison = 1e-5
	input = (input - input.min())/(input.max()-input.min() + eplison)


	return 1 - (torch.sum(2*input*target.narrow(1, 0, 1))/ \
			(torch.sum(input) + torch.sum(target.narrow(1, 0, 1)) + eplison))

def dice_loss_double(input, target, cirrculum):

	intput_C1 = input.narrow(1, 0, 1)
	intput_C2 = input.narrow(1, 1, 1)
	intput_C1 = (intput_C1 - intput_C1.min())/(intput_C1.max()-intput_C1.min() + eplison)
	intput_C2 = (intput_C2 - intput_C2.min())/(intput_C2.max()-intput_C2.min() + eplison)

	if cirrculum == 0:
	# Seg the body mask first
		return 1 - (torch.sum(2*intput_C1*target.narrow(1, 0, 1))/ \
			(torch.sum(intput_C1) + torch.sum(target.narrow(1, 0, 1)) + eplison))

	else:
	# Seg the body and BV altogehter
		return 1 - (torch.sum(2*intput_C1*target.narrow(1, 0, 1))/ \
			(torch.sum(intput_C1) + torch.sum(target.narrow(1, 0, 1)) + eplison) + \
			torch.sum(2*intput_C2*target.narrow(1, 1, 1))/ \
			(torch.sum(intput_C2) + torch.sum(target.narrow(1, 1, 1)) + eplison))/2



def dice_loss(input, target, cirrculum):
	'''
	Multi-class dice loss
	All conversion happens explicitly in the train function
	Args: 
		input: tensor of shape (N, 3, H, W, D), dtype = float
		target: ndarray of shape (N, 3, H, W, D) dtype = float
	Ret:
		loss: 1 - DiceCoefficient
	'''
	print(input.size(), target.size())
	assert input.size() == target.size() # make sure input and target has same size

	eplison = 1e-5

	intput_C1 = input.narrow(1, 0, 1)
	intput_C2 = input.narrow(1, 1, 1)
	intput_C3 = input.narrow(1, 2, 1)
	intput_C1 = (intput_C1 - intput_C1.min())/(intput_C1.max()-intput_C1.min())
	intput_C2 = (intput_C2 - intput_C2.min())/(intput_C2.max()-intput_C2.min())
	intput_C3 = (intput_C3 - intput_C3.min())/(intput_C3.max()-intput_C3.min())

	if cirrculum == 0:

		return 1 - (torch.sum(2*intput_C1*target.narrow(1, 0, 1))/ \
				(torch.sum(intput_C1) + torch.sum(target.narrow(1, 0, 1)) + eplison))

	elif cirrculum == 1:

		return 1 - (torch.sum(2*intput_C1*target.narrow(1, 0, 1))/ \
			(torch.sum(intput_C1) + torch.sum(target.narrow(1, 0, 1)) + eplison) + \
			torch.sum(2*intput_C2*target.narrow(1, 1, 1))/ \
			(torch.sum(intput_C2) + torch.sum(target.narrow(1, 1, 1)) + eplison))/2


	else:

		return 1 - (torch.sum(2*intput_C1*target.narrow(1, 0, 1))/ \
			(torch.sum(intput_C1) + torch.sum(target.narrow(1, 0, 1)) + eplison) + \
			torch.sum(2*intput_C2*target.narrow(1, 1, 1))/ \
			(torch.sum(intput_C2) + torch.sum(target.narrow(1, 1, 1)) + eplison) + \
			torch.sum(2*intput_C3*target.narrow(1, 2, 1))/ \
			(torch.sum(intput_C3) + torch.sum(target.narrow(1, 2, 1)) + eplison))/3

