from matplotlib import pyplot as plt
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
from PIL import Image
from PIL import ImageEnhance
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
import argparse 
from barbar import Bar

'''Discriminator1'''

class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.zero_pad1 = nn.ZeroPad2d(1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.zero_pad2 = nn.ZeroPad2d(1)
        self.last_conv = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)
        self.drop = nn.Dropout2d(p = 0.4)

    def forward(self, inp, tar):

        print("Entering Discriminator1")
        x_1 = torch.cat([inp, tar], dim=1)
        x_2 = self.leaky_relu(self.bn1(self.conv1(x_1)))
        x_3 = self.drop(x_2)
        x_4 = self.leaky_relu(self.bn2(self.conv2(x_3)))
        x_5 = self.drop(x_4)
        x_6 = self.leaky_relu((self.conv3(x_5)))
        x_7 = self.drop(x_6)
        x_8 = self.zero_pad1(x_7)
        x_9 = (self.last_conv(x_8))
        x_10 = self.drop(x_9)
        print("Leaving Discriminator1")
        return torch.sigmoid(x_10)
    

'''Discriminator2'''

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.zero_pad1 = nn.ZeroPad2d(1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.zero_pad2 = nn.ZeroPad2d(1)
        self.last_conv = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)
        self.drop = nn.Dropout2d(p = 0.4)
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inp, tar):
        
        inp1 = self.maxpool_layer(inp)
        tar1 = self.maxpool_layer(tar)
        print("Entering Discriminator2")
        x_1 = torch.cat([inp1, tar1], dim=1)
        x_2 = self.leaky_relu(self.bn1(self.conv1(x_1)))
        x_3 = self.drop(x_2)
        x_4 = self.leaky_relu(self.bn2(self.conv2(x_3)))
        x_5 = self.drop(x_4)
        x_6 = self.leaky_relu((self.conv3(x_5)))
        x_7 = self.drop(x_6)
        x_8 = self.zero_pad1(x_7)
        x_9 = (self.last_conv(x_8))
        x_10 = self.drop(x_9)
        print("Leaving Discriminator2")
        return torch.sigmoid(x_10)

'''Local Generator'''

class Local_Generator(nn.Module):
    def __init__(self):
        super(Local_Generator, self).__init__()

        def layer(in_down, out_channels, kernel_size=4, stride=2):
            model = nn.Sequential(
                        nn.Conv2d(in_down, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
            return model
        
        #Downstack
        self.down_stack = []
        in_down = 3  # Initial input channels
        out_down_list = [8, 16, 32, 64, 128, 256]
        

        for out_channels in out_down_list:
            self.down_stack.append(layer(in_down, out_channels))
            in_down = out_channels
       
       #UpStack
        self.deconv1 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(32, 8, 4, stride=2, padding=1, bias=False) 
        self.deconv_last = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False)
  
    def forward(self, x):
        
        #Downsampling and forming the skip connections
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips_list = skips[::-1]
        
        #Upsampling and adding the skip connections
        out1 = torch.cat([x, skips_list[0]], dim=1)
        out_2 = self.deconv1(out1)
        out2 = torch.cat([out_2, skips_list[1]], dim=1)
        out_3 = self.deconv2(out2)
        out3 = torch.cat([out_3, skips_list[2]], dim=1)
        out_4 = self.deconv3(out3)
        out4 = torch.cat([out_4, skips_list[3]], dim=1)
        out_5 = self.deconv4(out4)
        out5 = torch.cat([out_5, skips_list[4]], dim=1)
        out_6 = self.deconv5(out5)
        out6 = torch.cat([out_6, skips_list[5]], dim=1)
        out7 = self.deconv_last(out6)
        return out7
    

'''Global Generator'''

class Generator(nn.Module):
    def __init__(self, local_gen):
        super(Generator, self).__init__()
        self.local_gen = local_gen

        def layer(in_down, out_channels, kernel_size=4, stride=2):
            model = nn.Sequential(
                        nn.Conv2d(in_down, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
            return model

        #Downstack
        self.down_stack = []
        in_down = 3  # Initial input channels
        out_down_list = [8, 16, 32, 64, 128, 256, 256, 256]
        
        for out_channels in out_down_list:
            
            self.down_stack.append(layer(in_down, out_channels))
            in_down = out_channels

        #maxpooling input and seding it into local generator.
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
    
        #UpStack
        self.deconv1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(16)
        self.deconv7 = nn.ConvTranspose2d(32, 8, 4, stride=2, padding=1, bias=False) 
        self.bn7 = nn.BatchNorm2d(8)    
        self.deconv_last = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1, bias=False)
    
    def forward(self, x):
        
        #Maxpooling input by 2 and passing it into Local Generator
        x_downsampled = self.maxpool_layer(x)
        x_local = self.local_gen(x_downsampled)
        print("x_local size", x_local.size())
        #Downsampling and forming the skip connections
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
       
        skips_list = list(reversed(skips[:-1]))

        #Upsampling and adding the skip connections
        out_1 = self.bn1(self.deconv1(x))
        out1 = torch.cat([out_1, skips_list[0]], dim=1)
        out_2 = self.bn2(self.deconv2(out1))
        out2 = torch.cat([out_2, skips_list[1]], dim=1)
        out_3 = self.bn3(self.deconv3(out2))
        out3 = torch.cat([out_3, skips_list[2]], dim=1)
        out_4 = self.bn4(self.deconv4(out3))
        out4 = torch.cat([out_4, skips_list[3]], dim=1)
        out_5= self.bn5(self.deconv5(out4))
        out5 = torch.cat([out_5, skips_list[4]], dim=1)
        out_6= self.bn6(self.deconv6(out5))
        out6 = torch.cat([out_6, skips_list[5]], dim=1)
        out_7= self.bn7(self.deconv7(out6))
        #Concating output of local gen class here
        out7 = torch.cat([out_7, x_local], dim=1)
        out8 = F.tanh(self.deconv_last(out7))
        print("Leavinng Generator")
        
        return out8
