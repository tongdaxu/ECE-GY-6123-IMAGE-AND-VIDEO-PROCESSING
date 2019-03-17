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

	assert input.size() == target.size() # make sure input and target has same size

	input_min = input.min()
	input_max = input.max()

	input = (input-input_min)/(input_max-input_min) # remap the output to [0, 1]

	return (torch.sum(2*input.narrow(1, 0, 1)*target.narrow(1, 0, 1))/ \
			(torch.sum(input.narrow(1, 0, 1)) + torch.sum(target.narrow(1, 0, 1)) + eplison) + \
			torch.sum(2*input.narrow(1, 1, 1)*target.narrow(1, 1, 1))/ \
			(torch.sum(input.narrow(1, 1, 1)) + torch.sum(target.narrow(1, 1, 1)) + eplison) + \
			torch.sum(2*input.narrow(1, 2, 1)*target.narrow(1, 2, 1))/ \
			(torch.sum(input.narrow(1, 2, 1)) + torch.sum(target.narrow(1, 2, 1)) + eplison))