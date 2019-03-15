import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class niiDataset(Dataset):
	'''
	pytorch dataset for bv segmentation
	'''
	def __init__(self, image, label, transform=None):
		'''
		Args:
			image: ndarray to image file of dtype float64
			label: ndarray to label file of dtype int8
			transform(callable, default=none): transfrom on a sample
		'''
		self.image = image
		self.label = label
		self.transform = transform

	def __len__(self):
		'''
		Override: return size of dataset
		'''
		return image.shape[0]

	def __getitem(index):
		'''
		Override: integer indexing in range from 0 to len(self) exclusive.
		'''
		sample = {'image':self.image[index], 'label':self.label[index]}

		if self.transform:
			sample = self.transform(sample)

		return sample