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


# Ref from 
# https://github.com/jalola/improved-wgan-pytorch/blob/master/models/conwgan.py

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)



# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, 
                 label=None, downsample=None, use_bn=True, hw=None, yb_size=None):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.use_bn = use_bn
        
        self.in_channels = in_channels
        self.input_dim = in_channels
        input_dim = in_channels
        self.out_channels = out_channels
        self.output_dim = out_channels
        output_dim = out_channels
        
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        if hw is None:
            hw = self.in_channels
            self.hw = self.in_channels
        else:
            self.hw = hw

        self.label = label
        if label is not None:
            self.add_dim = self.label.shape[1]
            self.yb1 = label.view([self.label.shape[0],self.add_dim, 1, 1])
            self.yb2 = label.view([self.label.shape[0],self.add_dim, 1, 1])
            if downsample=='down':
                yb1_size = yb_size
                yb2_size = yb1_size 
            elif downsample=='up':
                yb1_size = yb_size
                yb2_size = int(yb1_size * 2)
            elif downsample==None:
                yb1_size = yb_size
                yb2_size = yb1_size 
                
            
    
            self.yb1 = self.yb1*torch.ones([self.label.shape[0], self.add_dim, yb1_size, yb1_size]).cuda()
            self.yb2 = self.yb2*torch.ones([self.label.shape[0], self.add_dim, yb2_size, yb2_size]).cuda()
        else:
            self.add_dim=0
            
        if downsample == 'down':
            self.bn1 = nn.LayerNorm([in_channels, hw, hw])
            self.bn2 = nn.LayerNorm([in_channels, hw, hw])
        elif downsample == 'up':
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif downsample == None:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.LayerNorm([in_channels, hw, hw])
        else:
            raise Exception('invalid resample value')
        
        if downsample is None:
            self.shortcut = None
        elif 'down' in downsample:
            self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2), 
                                          conv1x1(in_channels, out_channels))
        elif 'up' in downsample:
            self.shortcut = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          conv1x1(in_channels, out_channels))
            
        if downsample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim+ self.add_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim+ self.add_dim, output_dim, kernel_size = kernel_size)
        elif downsample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim+ self.add_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim+ self.add_dim, output_dim, kernel_size = kernel_size)
        elif downsample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim+ self.add_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim+ self.add_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input, input_y=None):
        if self.input_dim == self.output_dim and self.downsample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        if self.label is not None:
            output = torch.cat([output, self.yb1], 1)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        if self.label is not None:
            output = torch.cat([output, self.yb2], 1)
        output = self.conv_2(output)
        return shortcut + output



class Generator64(nn.Module):
    def __init__(self, block, num_classes=10, in_channels=64, nz_dim=128, labels=None,):
        super(Generator64, self).__init__()
    
        self.nz_dim = nz_dim
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.lin1_shape = in_channels * 4 * 4 * 8 
        self.labels = labels
        
        # Linear Layer
        self.linear1 = nn.Linear(self.nz_dim, self.lin1_shape)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.ReLU()

        self.block = block
        # Res Block
        self.resblock1 = self.block(in_channels=self.in_channels * 8, label=self.labels,
                         out_channels=self.out_channels * 8, downsample='up', yb_size=4)
        self.resblock2 = self.block(in_channels=self.in_channels * 8, label=self.labels,
                         out_channels=self.out_channels * 4, downsample='up', yb_size=8)
        self.resblock3 = self.block(in_channels=self.in_channels * 4, label=self.labels,
                         out_channels=self.out_channels * 2, downsample='up', yb_size=16)
        self.resblock4 = self.block(in_channels=self.in_channels * 2, label=self.labels,
                         out_channels=self.out_channels * 1, downsample='up', yb_size=32)
        
        self.bn_fin = self.bn1 = nn.BatchNorm2d(self.out_channels * 1)
        self.relu_fin = nn.ReLU()
        
        self.conv_fin = conv3x3(self.in_channels, 3)
        self.tanh = nn.Tanh()

    def forward(self, input, input_y=None):
        input_z = input.view(input.shape[0],-1)
        out = self.linear1(input_z)
        out = out.view(out.shape[0], self.in_channels * 8, 4, 4)
        
        out = self.resblock1(out,input_y)
        out = self.resblock2(out,input_y)
        out = self.resblock3(out,input_y)
        out = self.resblock4(out,input_y)
        
        out = self.bn_fin(out)
        out = self.relu_fin(out)
        out = self.conv_fin(out)
        out = self.tanh(out)
#         output = output.view(-1, OUTPUT_DIM) ??
        return out

class Discriminator64(nn.Module):
    def __init__(self, block, num_classes=10, conditional=None, labels=None, in_channels=64):
        super(Discriminator64, self).__init__()

        self.dim = in_channels
        self.num_classes = num_classes
        self.conv1 = conv3x3(3, self.dim, stride=1)

        self.labels = labels
        self.block = block
        
        self.resblock1 =self.block(in_channels=self.dim, use_bn=False, hw=self.dim, yb_size=64,
                         out_channels=self.dim * 2, label=self.labels, downsample='down')
        self.resblock2 =self.block(in_channels=self.dim * 2, use_bn=False, hw=int(self.dim/2), yb_size=32,
                         out_channels=self.dim * 4, label=self.labels, downsample='down')
        self.resblock3 =self.block(in_channels=self.dim * 4, use_bn=False, hw=int(self.dim/4), yb_size=16,
                         out_channels=self.dim * 8, label=self.labels, downsample='down')
        self.resblock4 =self.block(in_channels=self.dim * 8, use_bn=False, hw=int(self.dim/8), yb_size=8,
                         out_channels=self.dim * 8, label=self.labels, downsample='down')
        fin_shape = 8 * 4 * 4 * self.dim
        self.lin_fin1 =nn.Linear(fin_shape, 1)
        self.lin_fin2 =nn.Linear(fin_shape, self.num_classes)
        

    def forward(self, input, input_y):
        output = input.contiguous()
        output = self.conv1(output)
        output = self.resblock1(output, input_y)
        output = self.resblock2(output, input_y)
        output = self.resblock3(output, input_y)
        output = self.resblock4(output, input_y)
        output = output.view([output.shape[0],-1])
        output_wgan = self.lin_fin1(output)
        return output_wgan
