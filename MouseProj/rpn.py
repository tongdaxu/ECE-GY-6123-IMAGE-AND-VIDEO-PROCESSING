import torch
import torch.nn as nn
import torch.optim as optim

class ConvRB(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ConvRB, self).__init__()
        self.conv = nn.Conv3d(in_chan, out_chan, 3, padding=1)
        self.bn = nn.BatchNorm3d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class FCRB(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FCRB, self).__init__()
        self.affine = nn.Linear(in_chan, out_chan)
        self.bn = nn.BatchNorm1d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout()

    def forward(self, x):
        out = self.dp(self.relu(self.bn(self.affine(x))))
        return out


class RPN3D(nn.Module):
    def __init__(self, img_size, out_size):
        
        super(RPN3D, self).__init__()
        # input: 128*192*192
        # input: 64*96*96 #NOW
        x, y, z = img_size

        # conv1
        self.conv1_1 = ConvRB(1, 64)
        self.conv1_2 = ConvRB(64, 64)
        self.dp1 = nn.Dropout3d()
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = ConvRB(64, 128)
        self.conv2_2 = ConvRB(128, 128)
        self.dp2 = nn.Dropout3d()
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = ConvRB(128, 256)
        self.conv3_2 = ConvRB(256, 256)
        self.conv3_3 = ConvRB(256, 256)
        self.dp3 = nn.Dropout3d()
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = ConvRB(256, 512)
        self.conv4_2 = ConvRB(512, 512)
        self.conv4_3 = ConvRB(512, 512)
        self.dp4 = nn.Dropout3d()
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = ConvRB(512, 512)
        self.conv5_2 = ConvRB(512, 512)
        self.conv5_3 = ConvRB(512, 512)
        self.dp5 = nn.Dropout3d()
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/32

        self.gap = nn.AvgPool3d(kernel_size = (x//32,y//32,z//32)) # N, C, 1, 1, 1
        img_shape = 512

        self.fc1 = FCRB(img_shape, img_shape)
        self.fc2 = FCRB(img_shape, img_shape)

        self.fc3 = nn.Linear(img_shape, out_size)
        
        pass
    
    def forward(self, x):
        # the input x should have shape of 64*96*96
                
        batch_size = x.size()[0] # get batch size
        
        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.dp1(x)
        x = self.pool1(x)
                
        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.dp2(x)
        x = self.pool2(x)
        
        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.dp3(x)
        x = self.pool3(x)
                
        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.dp4(x)
        x = self.pool4(x)

        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.dp5(x)
        x = self.pool5(x)

        x = self.gap(x)  
        x = x.view(batch_size, -1) #flatten for fully connect
                
        # fc1 -> fc2 -> fc3     
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
                                
        return x
