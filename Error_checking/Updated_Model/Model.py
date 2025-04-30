import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import BottleneckLayer, DecoderBottleBlock
from .oc_block import APNB

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channel, num_layers):
        super().__init__()

        self.block1 = nn.Sequential(BottleneckLayer(in_channels=in_channel, out_channels=16),
                                    *[BottleneckLayer(in_channels=16, out_channels=16) for _ in range(num_layers)]) 
        
        self.block2 = nn.Sequential(BottleneckLayer(in_channels=16, out_channels=32),
                                    *[BottleneckLayer(in_channels=32, out_channels=32) for _ in range(num_layers)])
        
        self.block3 = nn.Sequential(BottleneckLayer(in_channels=32, out_channels=64),
                                    *[BottleneckLayer(in_channels=64, out_channels=64) for _ in range(num_layers)])
        
        self.block4 = nn.Sequential(BottleneckLayer(in_channels=64, out_channels=128),
                                    *[BottleneckLayer(in_channels=128, out_channels=128) for _ in range(num_layers)])
        
        self.block5 = nn.Sequential(BottleneckLayer(in_channels=128, out_channels=256),
                                    *[BottleneckLayer(in_channels=256, out_channels=256) for _ in range(num_layers)]) 
        
        self.block6 = nn.Sequential(BottleneckLayer(in_channels=256, out_channels=512),
                                    *[BottleneckLayer(in_channels=512, out_channels=512) for _ in range(num_layers)])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = [ ] 
        
        out = self.block1(x) 
        features.append(out) 
        out = self.pool(out) 

        out = self.block2(out) 
        features.append(out) 
        out = self.pool(out)

        out = self.block3(out) 
        features.append(out) 
        out = self.pool(out)

        out = self.block4(out) 
        features.append(out) 
        out = self.pool(out) 

        out = self.block5(out) 
        features.append(out) 
        out = self.pool(out) 

        out = self.block6(out)
        return out, features 


class DecoderBottleneck(nn.Module):
    def __init__(self, num_classes, num_layers, attn=True):
        super().__init__()

        self.block1 = DecoderBottleBlock(in_channel=512, out_channel=256, skip_channel=256, attn=attn, num_layers=num_layers) 
        self.block2 = DecoderBottleBlock(in_channel=256, out_channel=128, skip_channel=128, attn=attn, num_layers=num_layers)
        self.block3 = DecoderBottleBlock(in_channel=128, out_channel=64, skip_channel=64, attn=attn, num_layers=num_layers)
        self.block4 = DecoderBottleBlock(in_channel=64, out_channel=32, skip_channel=32, attn=attn, num_layers=num_layers)
        self.block5 = DecoderBottleBlock(in_channel=32, out_channel=16, skip_channel=16, attn=attn, num_layers=num_layers)

        self.block6 = DecoderBottleBlock(in_channel=16, out_channel=16, attn=attn, num_layers=num_layers)
        self.output = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1)
    

    def forward(self, out, features):

        skip = features.pop() 
        out  = self.block1(out, skip) 

        skip = features.pop() 
        out  = self.block2(out, skip) 

        skip = features.pop() 
        out  = self.block3(out, skip) 

        skip = features.pop() 
        out  = self.block4(out, skip) 

        skip = features.pop() 
        out  = self.block5(out, skip) 


        out = self.block6(out) 
        out = self.output(out) 
        return out 

class UNeT(nn.Module):
    def __init__(self, in_channels, num_classes, num_layers, attn=True):
        super().__init__()

        self.enc = EncoderBottleneck(in_channel=in_channels, num_layers=num_layers) 
        self.oc  = APNB(in_channels=512, out_channels=512, key_channels=16, value_channels=16, scale=1)
        self.dec = DecoderBottleneck(num_classes=num_classes, num_layers=num_layers, attn=attn) 
    
    def forward(self, x):

        out, features = self.enc(x) 
        # print(out.shape)
        out =  self.oc(out)
        out = self.dec(out, features) 
        return out 


if __name__ == '__main__':

    os.system('cls') 

    x = torch.randn(2, 1, 512, 512) 
    model = UNeT(in_channels=1, num_classes=1, num_layers=2) 
    out = model(x) 
    print(out.shape)