import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

class Dcon3D(nn.Module):
    def __init__(self, img_size):
        
        super().__init__()
        # input shape = img_size**3
            
        # conv1
        self.conv1_1 = nn.Conv3d(1, 8, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        
        
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        
        self.pool1 = nn.MaxPool3d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        
        self.pool2 = nn.MaxPool3d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        
        self.pool3 = nn.MaxPool3d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/8
        
        # conv4
        self.conv4_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        
        self.pool4 = nn.MaxPool3d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        
        self.pool5 = nn.MaxPool3d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/32

        # fc6        
        self.fc6 = nn.Linear(64*(2**3), 64*(2**3))
        self.relu6 = nn.ReLU(inplace=True)
        
        # fc7
        self.fc7 = nn.Linear(64*((img_size//32)**3), 64*((img_size//32)**3))
        self.relu7 = nn.ReLU(inplace=True)
        
        # deconv5
        self.upool5 = nn.MaxUnpool3d(2, stride=2)  # 1/32
        
        self.urelu5_3 = nn.ReLU(inplace=True)
        self.uconv5_3 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.urelu5_2 = nn.ReLU(inplace=True)
        self.uconv5_2 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.urelu5_1 = nn.ReLU(inplace=True)
        self.uconv5_1 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        
        # deconv4
        self.upool4 = nn.MaxUnpool3d(2, stride=2)  # 1/16
        
        self.urelu4_3 = nn.ReLU(inplace=True)
        self.uconv4_3 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.urelu4_2 = nn.ReLU(inplace=True)
        self.uconv4_2 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.urelu4_1 = nn.ReLU(inplace=True)
        self.uconv4_1 = nn.ConvTranspose3d(64, 32, 3, padding=1)
        
        # deconv3
        self.upool3 = nn.MaxUnpool3d(2, stride=2)  # 1/8
        
        self.urelu3_3 = nn.ReLU(inplace=True)
        self.uconv3_3 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.urelu3_2 = nn.ReLU(inplace=True)
        self.uconv3_2 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.urelu3_1 = nn.ReLU(inplace=True)
        self.uconv3_1 = nn.ConvTranspose3d(32, 16, 3, padding=1)
        
        # deconv2
        self.upool2 = nn.MaxUnpool3d(2, stride=2)  # 1/4
        
        self.urelu2_2 = nn.ReLU(inplace=True)
        self.uconv2_2 = nn.ConvTranspose3d(16, 16, 3, padding=1)
        self.urelu2_1 = nn.ReLU(inplace=True)
        self.uconv2_1 = nn.ConvTranspose3d(16, 8, 3, padding=1)
        
        # deconv1
        self.upool1 = nn.MaxUnpool3d(2, stride=2)  # 1/2
        self.urelu1_2 = nn.ReLU(inplace=True)
        self.uconv1_2 = nn.ConvTranspose3d(8, 8, 3, padding=1)
        self.urelu1_1 = nn.ReLU(inplace=True)
        self.uconv1_1 = nn.ConvTranspose3d(8, 3, 3, padding=1)
        
        pass
    
    def forward(self, x):
        # the input x should have shape of 256*256
        
        scores = None
        
        batch_size = x.size()[0] # get batch size
                
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x, indice1 = self.pool1(x)
                
        # conv2
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x, indice2 = self.pool2(x)
        
        # conv3
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x, indice3 = self.pool3(x)
                
        # conv4
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x, indice4 = self.pool4(x)
        
        # conv5
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x, indice5 = self.pool5(x)
                
        x = x.view(batch_size, -1) #flatten for fully connect
                
        # fc6        
        x = self.fc6(x)
        x = self.relu6(x)
        
        # fc7
        x = self.fc7(x)
        x = self.relu7(x)
                
        x = x.view(batch_size, 64, 2, 2, 2)
                
        x = self.upool5(x, indice5)  # 1/32
        
        x = self.urelu5_3(x)
        x = self.uconv5_3(x)
        x = self.urelu5_2(x)
        x = self.uconv5_2(x)
        x = self.urelu5_1(x)
        x = self.uconv5_1(x)
        
        # deconv4
        x = self.upool4(x, indice4) # 1/16
        
        x = self.urelu4_3(x)
        x = self.uconv4_3(x)
        x = self.urelu4_2(x)
        x = self.uconv4_2(x)
        x = self.urelu4_1(x)
        x = self.uconv4_1(x)
        
        # deconv3
        x = self.upool3(x, indice3)
        
        x = self.urelu3_3(x)
        x = self.uconv3_3(x)
        x = self.urelu3_2(x)
        x = self.uconv3_2(x)
        x = self.urelu3_1(x)
        x = self.uconv3_1(x)
        
        # deconv2
        x = self.upool2(x, indice2)  # 1/4
        
        x = self.urelu2_2(x)
        x = self.uconv2_2(x)
        x = self.urelu2_1(x)
        x = self.uconv2_1(x)
        
        # deconv1
        x = self.upool1(x, indice1)  # 1/2
        x = self.urelu1_2(x)
        x = self.uconv1_2(x)
        x = self.urelu1_1(x)
        x = self.uconv1_1(x)
        
        scores = x
        
        return scores

class UNetcon3D(nn.Module):
