import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2): 
        super().__init__() 
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size) 
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior 
    
    def forward(self, feats):
        n, c, _, _ = feats.size() 
        priors = [stage(feats).view(n, c, -1) for stage in self.stages] 
        center = torch.cat(priors, -1)
        return center

class _SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, psp_size=(1,3,6,8)):
        super().__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.scale = scale
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale)) 
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        
        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0) 
    
    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.psp(self.f_value(x))
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.f_key(x) 
        value = value.permute(0, 2, 1) 
        key = self.psp(key)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map 
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous() 
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        return context 

class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, psp_size=(1,3,6,8)):
        super().__init__(in_channels,key_channels,value_channels, out_channels,
                                                   scale=scale,
                                                   psp_size=psp_size)


class APNB(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout=0.5, sizes=([1]),
            psp_size=(1,3,6,8), scale=1):
        super().__init__()
        self.stages = []
        self.scale= scale
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        ) 
    
    def _make_stage(self, in_channels, output_channels, key_channels, value_channels):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    scale=self.scale,
                                    psp_size=self.psp_size)
    
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        output = context + feats
        return output

        










if __name__ == '__main__':
    os.system('cls') 

    #model = SelfAttentionBlock2D(in_channels=128, out_channels=128, key_channels=16, value_channels=16) 
    model = APNB(in_channels=128, out_channels=128, key_channels=16, value_channels=16)
    print(model)

    x = torch.randn(10, 128, 16, 16) 
    out = model(x)
    print(out.shape) 