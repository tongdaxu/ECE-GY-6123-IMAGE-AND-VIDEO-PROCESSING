import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

image_path = 'bv_body_data/predict/'
data_path = 'img_'
label_path = 'bv_body'
appendix_str = '.nii'

def loadnii(x):

	"""
	load the nii image and label into np array 
	input:
		x: int index of the image to read
	return: 
		tuple of (image, label)
	"""

	data_file = os.path.join(image_path, data_path + str(x) + appendix_str)
	label_file = os.path.join(image_path, label_path + str(x)+ appendix_str)
	
	data = (nib.load(data_file)).get_fdata()
	label = (nib.load(label_file)).get_fdata()
	
	return (data, label)

def getniishape(x):

	"""
	load all nii image and label into np array 
	input:
		x: number of image
	return: 
		array of tuple of (max x, max y, max z)
	"""

	#size_data = loadnii(0)[0]
	x_size = np.zeros(x)
	y_size = np.zeros(x)
	z_size = np.zeros(x)
	#label = np.zeros_like(image)

	for i in range(x):
		if i != 46:
			x_size[i], y_size[i], z_size[i] = loadnii(i)[0].shape
	return (np.max (x_size), np.max(y_size), np.max(z_size))
	#return (image, label)

def loadallnii(x, target_x=-1, target_y=-1, target_z=-1):

	"""
	load all nii image and label into np array 
	input:
		x: number of image
	return: 
		array of tuple of (max x, max y, max z)
	"""

	target_shape = None

	if target_x < 0:
		target_shape = getniishape(x)
	else:
		target_shape = (target_x, target_y, target_z)

	image = np.zeros((x, *target_shape))
	label = np.zeros_like(image)

	for i in range(x):
		if i != 46:
			temp_image, temp_label = loadnii(i)
			current_shape = temp_image.shape
			padx = (target_shape[0]-current_shape[0])//2
			pady = (target_shape[1]-current_shape[1])//2
			padz = (target_shape[2]-current_shape[2])//2

			image[i] = np.pad (temp_image, ((padx,padx),(pady, pady),(padz, padz)), mode='constant', constant_values=((0,0),(0,0),(0,0)))
			label[i] = np.pad (temp_label, ((padx,padx),(pady, pady),(padz, padz)), mode='constant', constant_values=((0,0),(0,0),(0,0)))

			"""
			fig, ax = plt.subplots(2,2)
			ax[0][0].imshow(image[i][90+padx], cmap='gray')
			ax[0][1].imshow(temp_image[90], cmap='gray')
			ax[1][0].imshow(label[i][90+padx])
			ax[1][1].imshow(temp_label[90])
			plt.show()
			"""
	return (image, label)

