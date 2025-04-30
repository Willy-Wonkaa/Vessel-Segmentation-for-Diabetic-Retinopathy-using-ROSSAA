import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision.transforms.functional as TF 


class  BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False) 
        self.bn1   = nn.BatchNorm2d(out_channels) 

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_channels) 

        self.conv3 =  nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False) 
        self.bn3   = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False) 
        else:
            self.proj = None 

        self.relu = nn.ReLU(inplace=True)
    def  forward(self, x):
        identity = x 

        out = self.relu(self.bn1(self.conv1(x))) 
        out = self.relu(self.bn2(self.conv2(out)))

        if self.proj is not None:
            identity = self.proj(identity) 
        
        out = self.conv3(out) 
        out += identity 
        out = self.relu(self.bn3(out)) 
        return out 


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__() 

        self.cSE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//reduction, kernel_size=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels //reduction, out_channels=in_channels, kernel_size=1), 
            nn.Sigmoid()
        ) 

        self.SSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid()) 
    
    def forward(self, x):
        out = x* self.cSE(x) + x* self.SSE(x) 
        return out
    

class DecoderBottleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers, skip_channel=None, attn=None):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  
        self.attn = attn 

        if skip_channel:
            in_channel = in_channel + skip_channel 
        else:
            in_channel = in_channel
        self.block  = nn.Sequential(BottleneckLayer(in_channels=in_channel, out_channels=out_channel),
                                    *[BottleneckLayer(in_channels=out_channel, out_channels=out_channel)
                                       for _ in range(num_layers)]) 
        

        if self.attn:
            self.attn_model = SCSEModule(in_channels= in_channel) 
    
    def forward(self, out, skip=None):
        if skip is not None:
            if skip.shape != out.shape: 
                out = TF.resize(out, skip.shape[2:]) 
            
            out = torch.cat([out, skip], dim=1)

        if self.attn:
            out = self.attn_model(out) 
        
        out = self.block(out) 
        return out 



if __name__ == '__main__':

    os.system('cls') 

    x = torch.randn(2, 1, 512, 512) 
    model = BottleneckLayer(in_channels=1, out_channels=16) 
    out = model(x) 
    print(out.shape)