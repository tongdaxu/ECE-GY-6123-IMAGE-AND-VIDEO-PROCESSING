import torch
import numpy as np

def dice_loss(input, target):
	'''
	Multi-class dice loss
	All conversion happens explicitly in the train function
	Args: 
		input: tensor of shape (N, 3, H, W, D), dtype = float
		target: ndarray of shape (N, 1, H, W, D) dtype = float, not hotcoded yet
	Ret:
		loss:
	'''
	N = input.size()[0]
	C = input.size()[1]

	targetOH = np.zeros(input.shape, dtype=np.float32)

	targetOH[:, 0:1] = (target < 0.33).astype(np.float32)
	targetOH[:, 1:2] = (target > 0.33).astype(np.float32)
	targetOH[:, 2:3] = (target > 0.66).astype(np.float32)

	targetTensor = torch.from_numpy(targetOH)

	assert input.size() == targetTensor.size() # make sure input and target has same size
	
	eplison = 1e-6
	loss = 0

	for i in range (C):
		loss += torch.sum(2*input.narrow(1, i, 1)*targetTensor.narrow(1, i, 1))/ \
			(torch.sum(input.narrow(1, i, 1)) + torch.sum(targetTensor.narrow(1, i, 1)) + eplison)

	return 1-loss/(N*C)