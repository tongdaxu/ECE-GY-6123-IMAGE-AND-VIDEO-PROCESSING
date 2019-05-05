import os
import datetime

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform, zoom
from tqdm import tqdm

image_path = 'bv_body_data/predict/'
data_path = 'img_'
label_path = 'bv_body'
appendix_str = '.nii'

def loadnii(x, xout=-1, yout=-1, zout=-1, mode='pad'):

	"""
	load the nii image and label into np array 
	input:
		x: int index of the image to read
	return: 
		tuple of (image, label)
	"""

	data_file = os.path.join(image_path, data_path + str(x) + appendix_str)
	label_file = os.path.join(image_path, label_path + str(x)+ appendix_str)

	data = ((nib.load(data_file)).get_fdata()).astype(np.float32)
	label = ((nib.load(label_file)).get_fdata()).astype(np.float32)/2

	if xout < 0:
		data = data.reshape(1, *data.shape)
		label= label.reshape(1, *label.shape)

		return (data, label) # perserve original image shape
	
	else:

		if mode == 'pad':
			data = zero_padding(data, xout, yout, zout)
			label = zero_padding(label, xout, yout, zout)

		else:
			x, y, z = data.shape
			#scale the image
			data = zoom(data, zoom=(xout/x, yout/y, zout/z))
			label = zoom(label, zoom=(xout/x, yout/y, zout/z))

		data = data.reshape(1, *data.shape)
		label= label.reshape(1, *label.shape)

		return (data, label)

def findBV(n):
	'''
	Find BV for single image of shape [1, X, Y, Z] and channel [1, X, Y, Z]
	'''
	
	stride = 8

	dict = {}

	data, label = loadnii(n) # load the original image
	label = (label > 0.66).astype(np.float32)

	_, h, w, d = data.shape
	x, y, z = 0, 0, 0

	bv = np.sum(label)

	while x < h-127:
		y = 0
		while y < w-127:
			z = 0
			while z < d-127:
				datanew = data[0:1, x:x+128, y:y+128, z:z+128]
				labelnew = label[0:1, x:x+128, y:y+128, z:z+128]
				bvnew = np.sum(labelnew)

				if bvnew > bv*0.95:
					dict[(n, x, y, z)] = 1
				elif bvnew < bv*0.6:
					dict[(n, x, y, z)] = 0
				else:
					pass

				z += stride
			y += stride
		x += stride 

	# code for debugging
	# v = list (dict.values())
	# print('image {0} generate {1} samples, with positive rate = {2:.4f}'.format (n, len(v), np.sum(v)/len(v)))

	return dict

def generateSW(nlist):
	dict = {}
	for n in tqdm (nlist):
		dictTemp = findBV(n)
		dict.update(dictTemp)
		v = list (dict.values())
	print (' In total generate {0} samples, with overall positive rate = {1:.4f}'.format (len(v), np.sum(v)/len(v)))

	return dict

def savenii(img, img_name): 
	'''
	Saving the mask to nii file in format
	Args:
		* img: ndarray of shape (1, X, Y, Z) quantized
		* img_name: str of image index
	return:
		* None
	'''
	timestamp = datetime.datetime.now()
	filename = img_name + '-' + str(timestamp) + appendix_str
	array_img = nib.Nifti1Image(img, np.eye(4))
	nib.save(array_img, filename)

	pass

def getniishape(x):
	"""
	Get the shape of an image
	input:
		* x: int of image index
	return: 
		* tuple (X, Y, X)
	"""
	label_file = os.path.join(image_path, label_path + str(x)+ appendix_str)
	label = ((nib.load(label_file)).get_fdata()).astype(np.float32)/2

	return label.shape

def zero_padding (img, target_x, target_y, target_z):
	"""
	reshaping the img to desirable shape through zero pad or crop
	Args:
		* img: input 3d nii array
		* target_x: target shape x
		* target_y: target shape y
		* target_z: target shape z
	Ret:
		img: reshaped 3d nii array
	"""

	padx = (target_x-img.shape[0])//2
	pady = (target_y-img.shape[1])//2
	padz = (target_z-img.shape[2])//2
	extrax = int(img.shape[0]%2!=0)
	extray = int(img.shape[1]%2!=0)
	extraz = int(img.shape[2]%2!=0)
    
	maxdimension = np.max(img.shape)
	if padx<0:
		img = img[-padx:padx+extrax,:,:]
	else:
		img = np.pad (img, ((extrax + padx,padx),(0, 0),(0, 0)), \
			mode='constant', constant_values=((0,0),(0,0),(0,0)))

	if pady <0:
		img = img[:,-pady:pady+extray,:]
	else:
		img = np.pad (img, ((0,0),(extray + pady, pady),(0, 0)), \
			mode='constant', constant_values=((0,0),(0,0),(0,0)))

	if padz <0:
		img = img[:,:,-padz:padz+extraz]
	else:
		img = np.pad (img, ((0,0),(0, 0),(extraz + padz, padz)), \
			mode='constant', constant_values=((0,0),(0,0),(0,0)))

	return img


def show_image(img, label=None, indice=-1):
	"""
	show a slice of image with label at certain indice
	Args:
		* img: input image (1, X, Y, Z)
		* label: input label after one hot coding (C, X, Y, Z)
		* indice: cutting indice
	Ret: 
		* None
	"""
	if indice ==-1:
		indice = img.shape[1]//2

	if type(label) != type(None):
		N = label.shape[0]
		fig, ax = plt.subplots(1, N+1, figsize=(12, 4), sharey=True)
		ax[0].imshow(img[0][indice], cmap='gray')
		# have to show the original image

		for i in range(N):
			ax[i+1].imshow(label[i][indice], cmap='gray')
	else:
		plt.imshow(img[0][indice], cmap='gray')

	plt.show()

	pass


def show_batch_image(img, batchsize, label=None, indice=-1):
	'''
	show batch of Tensor as image

	'''
	img = img.numpy()
	if type(label) != type(None):
		label = label.numpy()

	for i in range(batchsize):
		if type(label) != type(None):
			show_image(img[i], label[i], indice)
		else: 
			show_image(img[i], None, indice)
	pass

def loadallnii(x, bad_index, target_x=-1, target_y=-1, target_z=-1, verbose=False):

	"""
	DEP!
	load all nii image and label into np array
	input:
		x: number of image
		traget_shape: if preknown the target shape, else calculate
		verbose: whether print the slicing out
	return: 
		tuple of array (max x, max y, max z)
	"""

	target_shape = None

	if target_x < 0:
		target_shape = getniishape(x)
	else:
		target_shape = (target_x, target_y, target_z)

	xx = x - bad_index.shape[0]

	image = np.zeros((xx, 1, *target_shape), dtype=np.float32) # single channel image
	label = np.zeros((xx, 1, *target_shape), dtype=np.float32) # triple channel label

	j = 0
	for i in range(x):

		if np.isin(i, bad_index):
			pass
		else:
			temp_image, temp_label = loadnii(i, target_x, target_y, target_z)
			current_shape = temp_image.shape
			padx = (target_shape[0]-current_shape[0])//2

			image[j] = zero_padding(temp_image, *target_shape)
			label[j] = zero_padding(temp_label, *target_shape)/2

			print('image index loaded: ' + str(i))

			if verbose:
				show_image(image[j], label[j] , 90+padx)

			else:
				pass

			j += 1

	return (image, label)


def find_bv(label):
	'''
	input: 
		label: the unique part of file name
	output: 
		path to corresponding bv segmentation
	'''

	for r, d, f in os.walk('20180419_newdata_nii_with_filtered'):
		for file in f:
			if label in file and 'BV' in file:
				return os.path.join(r, file)

	for r, d, f in os.walk('new_data_20180522_nii'):
		for file in f:
			if label in file and 'BV' in file:
				return os.path.join(r, file)

	print('wrong label')


def find_body(label):
	'''
	input: 
		label: the unique part of file name
	output: 
		path to corresponding body segmentation
	'''
	
	for r, d, f in os.walk('nii_test'):
		for file in f:
			if label in file and 'BODY' in file:
				return os.path.join(r, file)
	print('wrong label')

def load_img_memory():
	data = np.zeros((370, 1, 256, 256, 256))
	label = np.zeros((370, 1, 256, 256, 256))
	for i in tqdm(range(370)):
		data[i], label[i] = load_img(i, mode='bv', shape=(256, 256, 256), verbose=False)
    
	return (data, label)
    
def load_img(idx, mode, shape, verbose=False):

	'''
	int idx: index of image
	str mode: mode of label
		data = 1, X, Y, Z
		label = 1, X, Y, Z
	'''
	idx = idx + 1

	if mode == 'bv':
		dataPth, labelPth = get_bv_tuple(idx)
	elif mode == 'body':
		print(idx)
		dataPth, labelPth = get_body_tuple(idx)
	else:
		print('mode {} not supported'.format(mode))

	assert dataPth != None

	data = ((nib.load(dataPth)).get_fdata()).astype(np.float32)
	label = ((nib.load(labelPth)).get_fdata()).astype(np.float32)

	if mode == 'body':
		label = label/2
	#else pass
	xout, yout, zout = shape

	data = zero_padding(data, xout, yout, zout)
	label = zero_padding(label, xout, yout, zout)

	data = data.reshape(1, *data.shape)
	label= label.reshape(1, *label.shape)

	if verbose:
		show_image(data, label)
		pass
	#else pass

	return (data, label)

def get_bv_tuple(idx):
	'''
	input: 
		idx: index into all bv segmentations
	output:
		a tuple of path to original image and path to bv 
	'''
	
	counter = 0
	
	for r, d, f in os.walk('20180419_newdata_nii_with_filtered'):
		for file in f:
			if 'BV' not in file and 'filtered' not in file and file[0]!='.':
				counter += 1
				if counter == idx: 
					label = file[:-4]
					bv_path = find_bv(label)
					return ((os.path.join(r, file), bv_path))

	for r, d, f in os.walk('new_data_20180522_nii'):
		for file in f:
			if 'BV' not in file and 'filtered' not in file and file[0]!='.':
				counter += 1
				if counter == idx: 
					label = file[:-4]
					bv_path = find_bv(label)
					return ((os.path.join(r, file), bv_path))
				
	print('index out of range')

def get_body_tuple(idx):
	'''
	input: 
		idx: index into all body segmentations
	output:
		a tuple of path to original image and path to body 
	'''
	
	counter = 0
	
	for r, d, f in os.walk('nii_test'):
		for file in f:
			if 'BODY' not in file and 'filtered' not in file and file[0]!='.':
				counter += 1
				if counter == idx: 
					label = file[:-4]
					body_path = find_body(label)
					return ((os.path.join(r, file), body_path))

				
	print('index out of range')

def bbox_scale(bbox, image):
	'''
	input: 
		(x1,x2,y1,y2,z1,z3): bounding box
		image: whole image
	output:
		a scaled image where the largest dimension of bounding box is 128
	'''
	x1, x2, y1, y2, z1, z2 = bbox
	max_len = max([x1-x2, y1-y2, z1-z2])
	
	scale = 128.0/max_len
	
	return zoom(image, scale, mode='constant', cval=0)
