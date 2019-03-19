'''
Notes:
	* N class Vnet for single channel 3d images
Args:
	* X of [BatchSize, Channel = 1, X-dim, Y-dim, Z-dim]
Out:
	* Y of [BatchSize, NumClasses, X-dim, Y-dim, Z-dim]
Source: 
	* This vnet.py implement forked from https://github.com/mattmacy/vnet.pytorch
Bug fix: 
	* Broadcasting bug in class InputTransition
	* ContBatchNorm3d substituded with native
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def passthrough(x, **kwargs):
	return x

def ELUCons(elu, nchan):
	'''
	Notes: 
		* using leaky activation function!
	Args:
		* elu: bool, using ELU activation or Not
		* nchan: int, number of channel for PReLU
	Return:
		* nn.Module
	'''
	if elu==True:
		return nn.ELU(inplace=True)
	else:
		return nn.PReLU(nchan)

class LUConv(nn.Module):
	'''
	Notes: 
		* 3d conv->bn->RELU, retain channel number
		* nchan: number of channel in and out
	'''
	def __init__(self, nchan, elu):
		super(LUConv, self).__init__()
		self.relu1 = ELUCons(elu, nchan)
		self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(nchan)

	def forward(self, x):
		out = self.relu1(self.bn1(self.conv1(x)))
		return out

def _make_nConv(nchan, depth, elu):
	'''
	Notes:
		* packaged conv3D layers
		* number = {2, 3}
	'''
	layers = []
	for _ in range(depth):
		layers.append(LUConv(nchan, elu))
	return nn.Sequential(*layers)

class InputTransition(nn.Module):
	'''
	Notes: 
		* X -> conv-> bn + X -> Relu
		* Bug fix in x16 = torch.cat
	'''
	def __init__(self, outChans, elu):
		super(InputTransition, self).__init__()
		self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(16)
		self.relu1 = ELUCons(elu, 16)

	def forward(self, x):
		# do we want a PRELU here as well?
		out = self.bn1(self.conv1(x))
		out = self.relu1(torch.add(out, x))

		return out

class DownTransition(nn.Module):
	'''
	Notes:
		* input -> conv/2-> bn -> relu -> X -> n*(conv3d->bn->relu) + X -> relu -> out
	'''
	def __init__(self, inChans, nConvs, elu, dropout=False):
		super(DownTransition, self).__init__()
		outChans = 2*inChans
		self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
		self.bn1 = nn.BatchNorm3d(outChans)
		self.do1 = passthrough
		self.relu1 = ELUCons(elu, outChans)
		self.relu2 = ELUCons(elu, outChans)
		if dropout:
			self.do1 = nn.Dropout3d()
		self.ops = _make_nConv(outChans, nConvs, elu)

	def forward(self, x):
		down = self.relu1(self.bn1(self.down_conv(x)))
		out = self.do1(down)
		out = self.ops(out)
		out = self.relu2(torch.add(out, down))
		return out

class UpTransition(nn.Module):
	def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
		super(UpTransition, self).__init__()
		self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
		self.bn1 = nn.BatchNorm3d(outChans // 2)
		self.do1 = passthrough
		self.do2 = nn.Dropout3d()
		self.relu1 = ELUCons(elu, outChans // 2)
		self.relu2 = ELUCons(elu, outChans)
		if dropout:
			self.do1 = nn.Dropout3d()
		self.ops = _make_nConv(outChans, nConvs, elu)

	def forward(self, x, skipx):
		out = self.do1(x)
		skipxdo = self.do2(skipx)
		out = self.relu1(self.bn1(self.up_conv(out)))
		xcat = torch.cat((out, skipxdo), 1)
		out = self.ops(xcat)
		out = self.relu2(torch.add(out, xcat))
		return out

class OutputTransition(nn.Module):
	def __init__(self, inChans, classnum, elu):

		'''
		Notes: 
			* converts to number of outputs
		Args:
			* inChans: input channels
			* classnum: number of classes
		Return:
			* None
		'''
		super(OutputTransition, self).__init__()
		class_num = 3
		self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(2)
		self.conv2 = nn.Conv3d(2, classnum, kernel_size=1)
		self.relu1 = ELUCons(elu, 2)

	def forward(self, x):
		# convolve 32 down to 2 channels
		out = self.relu1(self.bn1(self.conv1(x)))
		out = self.conv2(out)

		# out should have shape N, C, X, Y, Z at that time
		return out

class VNetMask(nn.Module):
	'''
	Note:
		VNet architecture As diagram of paper
	'''
	def __init__(self, elu=True):
		'''
		Args:
			* slim: using few conv layers, else as original paper
			* elu: using elu / PReLU
		'''
		super(VNetMask, self).__init__()

		self.in_tr = InputTransition(16, elu)
		self.down_tr32 = DownTransition(16, 1, elu)
		self.down_tr64 = DownTransition(32, 1, elu)
		self.down_tr128 = DownTransition(64, 2, elu, dropout=True)
		self.up_tr128 = UpTransition(128, 128, 8, elu, dropout=True)
		self.up_tr64 = UpTransition(128, 64, 2, elu)
		self.up_tr32 = UpTransition(64, 32, 1, elu)
		self.out_tr = OutputTransition(32, 3, elu) # BKG, Body Segmentation map

	def forward(self, x):

		out16 = self.in_tr(x)
		out32 = self.down_tr32(out16)
		out64 = self.down_tr64(out32)
		out128 = self.down_tr128(out64)
		out = self.up_tr128(out128, out64)
		out = self.up_tr64(out, out32)
		outfeature = self.up_tr32(out, out16)
		outmap = self.out_tr(outfeature) # this is the the Body Segmentation map

		outbkg = outmap.narrow(1, 0, 1)
		outbody = outmap.narrow(1, 1, 1)
		outbv = outmap.narrow(1, 2, 1)

		bodymsk = (outbody-outbody.min())/(outbody.max()-outbody.min()) # remap the mask to [0, 1]

		outbv = outbv*bodymsk 

		return torch.cat((outbkg, outbody, outbv), 1)


