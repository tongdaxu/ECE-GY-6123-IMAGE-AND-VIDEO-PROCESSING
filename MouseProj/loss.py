import torch

def dice_loss(input, target):
	'''
	Multi-class dice loss
	All conversion happens explicitly in the train function
	Args: 
		input: tensor of shape (N, 3, H, W, D), dtype = float
		target: ndarray of shape (N, 3, H, W, D) dtype = float
	Ret:
		loss: 1 - DiceCoefficient
	'''
	C = input.size()[1]
	assert input.size() == target.size() # make sure input and target has same size
	
	eplison = 1e-6
	loss = 0

	for i in range (C):
		loss += torch.sum(2*input.narrow(1, i, 1)*target.narrow(1, i, 1))/ \
			(torch.sum(input.narrow(1, i, 1)) + torch.sum(target.narrow(1, i, 1)) + eplison)

	return 1-loss/(C)