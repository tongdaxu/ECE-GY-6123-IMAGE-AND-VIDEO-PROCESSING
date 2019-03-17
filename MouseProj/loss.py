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

	C = input.size()[1]

	eplison = 1e-6
	loss = 0
	loss = torch.tensor(0, dtype=torch.float, requires_grad=True)

	for i in range (C):
		loss += torch.sum(2*input.narrow(1, i, 1)*target.narrow(1, i, 1))/ \
			(torch.sum(input.narrow(1, i, 1)) + torch.sum(target.narrow(1, i, 1)) + eplison)

	return 1-loss/(C)