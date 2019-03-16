import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy.ndimage import rotate

def Rotate3D (img, x, y, z, order):
	'''
	Args:
		image: ndarray (1, H, W, D)
		flu: upper bound of degree of rotation
	Ret:
		rotated image
	'''

	img[0] = rotate(img[0], x, axes=(1, 0), reshape=False, order=order)
	img[0] = rotate(img[0], y, axes=(1, 2), reshape=False, order=order)
	img[0] = rotate(img[0], z, axes=(0, 2), reshape=False, order=order)

	return img

class RandomRotate(object):
	'''
	Random rotation
	'''
	def __init__(self, flu):
		self.flu = flu
		pass

	def __call__(self, sample):
		x = np.random.uniform(-self.flu, self.flu)
		y = np.random.uniform(-self.flu, self.flu)
		z = np.random.uniform(-self.flu, self.flu)
		image, label = sample['image'], sample['label']
		return {'image': Rotate3D(image, x, y, z, 3), 'label': Rotate3D(label, x, y, z, 0)}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']


        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class niiDataset(Dataset):
	'''
	pytorch dataset for bv segmentation
	'''
	def __init__(self, image, label, transform=None):
		'''
		Args:
			image: ndarray to image file of dtype float32
			label: ndarray to label file of dtype float32
			No Conversion of anykind in a dataset class!
			transform(callable, default=none): transfrom on a sample
		'''

		self.image = image
		self.label = label
		# converting from int8 to float 64
		
		self.transform = transform

	def __len__(self):
		'''
		Override: return size of dataset
		'''
		return self.image.shape[0]

	def __getitem__(self, index):
		'''
		Override: integer indexing in range from 0 to len(self) exclusive.
		type: converting to float tensor
		'''
		sample = {'image':np.copy(self.image[index]), 'label':np.copy(self.label[index])}

		if self.transform:
			sample = self.transform(sample)

		return sample