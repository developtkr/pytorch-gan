

from __future__ import print_function
#%matplotlib inline
import argparse
import os, time
import random

import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable, grad
from torch import autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_bn=True):
        super(ResBlock, self).__init__()
        self.res_option = downsample
        self.use_bn = use_bn
        
        modules = []
        
        if downsample is None:
            self.res_option = None  
            if self.use_bn:
                modules.append(nn.BatchNorm2d(out_channels))
            modules.append(nn.ReLU())
            modules.append(conv3x3(in_channels, out_channels, stride=stride))
            if self.use_bn:
                modules.append(nn.BatchNorm2d(out_channels))
            modules.append(nn.ReLU())
            modules.append(conv3x3(out_channels, out_channels, stride=stride))

            self.shortcut = conv1x1(in_channels, out_channels)
        elif 'down' in downsample:
            
            if self.use_bn:
                modules.append(nn.BatchNorm2d(out_channels))
            modules.append(nn.ReLU())
            modules.append(conv3x3(in_channels, out_channels, stride=stride))

            if self.use_bn:
                modules.append(nn.BatchNorm2d(out_channels))
            modules.append(nn.ReLU())
            modules.append(conv3x3(out_channels, out_channels, stride=stride))
            
            # Mean pooling after the second convolution.
            modules.append(nn.AvgPool2d(kernel_size=2))
            
            self.shortcut = nn.Sequential(conv1x1(out_channels, out_channels),
                                         nn.AvgPool2d(kernel_size=2))
        elif 'up' in downsample:
            if self.use_bn:
                modules.append(nn.BatchNorm2d(in_channels))
            modules.append(nn.ReLU())
            modules.append(conv3x3(in_channels, out_channels, stride=stride))
            
            # Upsample Before Second Convolution
            modules.append(nn.Upsample(scale_factor=2, mode='nearest'))

            if self.use_bn:
                modules.append(nn.BatchNorm2d(in_channels))
            modules.append(nn.ReLU())
            modules.append(conv3x3(in_channels, out_channels, stride=stride))
                
            self.shortcut = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          conv1x1(out_channels, out_channels))
        
        self.block = nn.Sequential(*modules)
        
    def forward(self, x):
        residual = self.shortcut(x)
        convs = self.block(x)
        return residual + convs


class GeneratorResNet32(nn.Module):
    def __init__(self, block, num_classes=10):
        super(GeneratorResNet32, self).__init__()
        # z <- in_channels
        self.in_channels = 128
        self.lin1_shape = 128 * 4 * 4

        # Linear Layer
        self.linear1 = nn.Linear(self.in_channels, self.lin1_shape)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.ReLU()
        
        # Res Block 1-3
        self.layer1 = self.make_layer(block, self.in_channels)
        self.layer2 = self.make_layer(block, self.in_channels)
        self.layer3 = self.make_layer(block, self.in_channels)
        
        self.conv_fin = conv3x3(self.in_channels, 3)
        self.tanh = nn.Tanh()
    
    def make_layer(self, block, out_channels, stride=1):
        return block(self.in_channels, out_channels, stride, 'up')

    def forward(self, x):
        
        out = self.linear1(x.view(x.shape[0],-1))
        out = out.view(x.shape[0], 128, 4, 4)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.conv_fin(out)
        out = self.tanh(out)
        return out


class DiscriminatorResNet32(nn.Module):
    def __init__(self, block, num_classes=10):
        super(DiscriminatorResNet32, self).__init__()
        # z <- in_channels
        self.in_channels = 128
        
        self.conv1 = conv3x3(3, self.in_channels,stride=2)
        
        # Res Block 1-4
        self.layer1 = self.make_layer(block, self.in_channels, res_opt='down')
        self.layer2 = self.make_layer(block, self.in_channels, res_opt=None)
        self.layer3 = self.make_layer(block, self.in_channels) # , res_opt='down'
        self.layer4 = self.make_layer(block, self.in_channels)
        
        self.relu_fin = nn.ReLU()
        self.avg_pool = nn.AvgPool2d((8,8))
        self.lin_fin = nn.Linear(self.in_channels, 1)

    def make_layer(self, block, out_channels, stride=1, res_opt=None):
        return block(self.in_channels, out_channels, stride, downsample=res_opt, use_bn=False)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.relu_fin(out)
        out = self.avg_pool(out)
        out = self.lin_fin(out.view(out.shape[0],-1))
        return out

