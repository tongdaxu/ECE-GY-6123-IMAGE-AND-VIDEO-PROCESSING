import torch
import torch.nn as nn

import torchvision.transforms as T

class Convk1d1(nn.Module):
    """docstring for Convk3d1"""
    def __init__(self, in_ch, out_ch):
        super(Convk1d1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Convk3d1(nn.Module):
    """docstring for Convk3d1"""
    def __init__(self, in_ch, out_ch):
        super(Convk3d1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=0, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
class Convk3d2(nn.Module):
    """docstring for Convk3d1"""
    def __init__(self, in_ch, out_ch):
        super(Convk3d2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=0, dilation=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Convk3d3(nn.Module):
    """docstring for Convk3d1"""
    def __init__(self, in_ch, out_ch):
        super(Convk3d3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=0, dilation=3),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class AdjConv(nn.Module):
    """docstring for Convk3d1"""
    def __init__(self, in_ch, out_ch, dilation):
        super(AdjConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, dilation=dilation),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class KeepConv(nn.Module):
    """docstring for Convk3d1"""
    def __init__(self, in_ch, out_ch, dilation=1):
        super(KeepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class FCConv(nn.Module):
    """docstring for Convk3d1"""
    def __init__(self, in_ch, out_ch):
        super(FCConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class Atrous(nn.Module):
    def __init__(self, classnum):
        
        super(Atrous, self).__init__()
        '''
        expecting input size 96*128*128
        now 64*96*96
        '''

        self.F1 = AdjConv(1, 16, 1) # -2
        self.F2 = AdjConv(16, 16, 1) # -2
        self.F3 = AdjConv(16, 32, 2) # -4
        self.F4 = AdjConv(32, 32, 4) # -8
        self.F5 = AdjConv(32, 64, 8) # -16
        # self.F6 = AdjConv(64, 64, 16) # -32

        self.O1 = KeepConv(64, 16, 1) # keep size
        self.O2 = FCConv(16, classnum) # 1x1 kernel, keep size

        self.upsample = Interpolate(size=(64, 96, 96), mode='trilinear')

        pass
    
    def forward(self, x):
        # the input x should have shape of 256*256
        x = self.F1(x)
        x = self.F2(x)
        x = self.F3(x)
        x = self.F4(x)
        x = self.F5(x)
        # x = self.F6(x)
        x = self.O1(x)
        x = self.O2(x)
        x = self.upsample(x)
        
        return x
