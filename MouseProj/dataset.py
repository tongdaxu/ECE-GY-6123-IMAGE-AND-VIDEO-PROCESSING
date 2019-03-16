import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import affine_transform

def toTensor (sample, device):
	'''
	Notes:
		Convert sample to single tensor and move to device
	Args:
		sample: dict of tensor, image in [0, 255], (N, C=1, X, Y, Z), 
			label in {0, 0.5, 1}, (N, C=1, X, Y, Z)
		device: torch.device('cuda')/torch.device('cpu')
	Ret:
		imageTensor in [0, 1], (N, C=1, X, Y, X)
		labelTensor in [0, 1], (N, C=3, X, Y, X)
	'''
	image, label = sample['image'], sample['label']
	imageTensor = torch.from_numpy(image/255)
	imageTensor = imageTensor.to(device=device)

	labelOH = np.zeros((image.shape[0], 3, image.shape[2], image.shape[3], image.shape[4]) \
		, dtype=np.float32)

	labelOH[:, 0:1] = (label < 0.33).astype(np.float32)
	labelOH[:, 1:2] = (label > 0.33).astype(np.float32)
	labelOH[:, 2:3] = (label > 0.66).astype(np.float32)

	labelTensor = torch.from_numpy(labelOH)
	labelTensor = labelTensor.to(device=device)

	return imageTensor, labelTensor


def AffineFun(img, xr, yr, zr, xm, ym, zm, order):
	'''
	Notes:
		Rotate and move
	Args:
		img: image of shape (C=1, X, Y, Z)
		xr, yr, zr: Rotate in degree
		xm, ym, zm: move as int
		order: 3 for image, 0 for label
	Ret:
		img: Transformed image of shape (C=1, X, Y, Z)
	'''
	sinx = np.sin(np.deg2rad(xr))
	cosx = np.cos(np.deg2rad(xr))

	siny = np.sin(np.deg2rad(yr))
	cosy = np.cos(np.deg2rad(yr))

	sinz = np.sin(np.deg2rad(zr))
	cosz = np.cos(np.deg2rad(zr))

	xc = img[0].shape[0]//2
	yc = img[0].shape[1]//2
	zc = img[0].shape[2]//2

	Mc = np.array([[1, 0, 0, xc],[0, 1, 0, yc],[0, 0, 1, zc],[0, 0, 0, 1]])
	Rx = np.array([[cosx, sinx, 0, 1],[-sinx, cosx, 0, 1],[0, 0, 1, 1], [0, 0, 0, 1]])
	Ry = np.array([[cosy, 0, siny, 1],[0, 1, 0, 1],[-siny, 0, cosy, 1], [0, 0, 0, 1]])
	Rz = np.array([[1, 0, 0, 1],[0, cosz, sinz, 1],[0, -sinz, cosz, 1], [0, 0, 0, 1]])
	Mb = np.array([[1, 0, 0, -xc],[0, 1, 0, -yc],[0, 0, 1, -zc],[0, 0, 0, 1]])
	MM = np.array([[1, 0, 0, xm],[0, 1, 0, ym],[0, 0, 1, zm],[0 ,0, 0, 1]])

	Matrix = np.linalg.multi_dot([Mc, Rx, Ry, Rz, Mb, MM])
	img[0] = affine_transform(img[0], Matrix, output_shape=img[0].shape, order=order)

	return img

def downSampleFun(img, level, order):
	'''
	Args:
		img: shape [1, X, Y, Z]
		level: scaling factor of downsampling
		order: 3 for image, 0 for label
	Ret:
		imgout: down sampled image of shape [1, X//level, Y//level, Z//level]
	'''
	if level == 1:
		return img
	else:
		_, x, y, z = img.shape

		imgout = np.zeros([1, x//level, y//level, z//level], dtype=np.float32)
		Matrix = np.array([[level, 0, 0, 0],[0, level, 0, 0],[0, 0, level, 0],[0, 0, 0, 1]])
		imgout[0] = affine_transform(img[0], Matrix, output_shape=imgout[0].shape, order=order)
		return imgout

class downSample(object):
	'''
	Down sample happens before affine
	'''
	def __init__(self, level):

		self.level = level
		pass

	def __call__(self, sample):

		image, label = sample['image'], sample['label']
		return {'image': downSampleFun(image, self.level, 3), \
				'label': downSampleFun(label, self.level, 0)}

class RandomAffine(object):
	'''
	Random rotation and move
	'''
	def __init__(self, fluR, fluM):

		self.fluR = fluR
		self.fluM = fluM

	def __call__(self, sample):

		xr, yr, zr = np.random.uniform(-self.fluR, self.fluR, size=3)		
		xm, ym, zm = np.random.uniform(-self.fluM, self.fluM, size=3)

		image, label = sample['image'], sample['label']
		return {'image': AffineFun(image, xr, yr, zr, xm, ym, zm, 3), \
				'label': AffineFun(label, xr, yr, zr, xm, ym, zm, 0)}

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
		type: keep as np array
		'''
		sample = {'image':np.copy(self.image[index]), 'label':np.copy(self.label[index])}

		if self.transform:
			sample = self.transform(sample)

		return sample