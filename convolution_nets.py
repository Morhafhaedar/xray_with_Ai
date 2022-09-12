import torch.nn  as nn
import torch

class Convolution_NxN(nn.Module):
    def __init__(self, kernel_size):
        super(Convolution_NxN, self). __init__()  # need to do init for the nn.Module

        # self.conv1kernel = nn.Conv2d(1, 1, kernel_size, padding= (int(kernel_size/2), int(kernel_size/2)), bias =True)
        # self.sig         = nn.Sigmoid()
        self.conv1kernel = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size, padding=(int(kernel_size/2), int(kernel_size/2)), bias =True), nn.PReLU(),
            nn.Conv2d(10, 10, kernel_size, padding=(int(kernel_size/2), int(kernel_size/2)), bias =True), nn.PReLU(),
            nn.Conv2d(10,  1, kernel_size, padding=(int(kernel_size/2), int(kernel_size/2)), bias =True), nn.Sigmoid(),

        )

    def forward(self,x):
        x = self.conv1kernel(x)
        return x


#----------------
class Convolution_3(nn.Module):
    def __init__(self, kernel_size):
        super(Convolution_3, self). __init__()  # need to do init for the nn.Module

    
        n_feature = 8

        self.conv1kernel = nn.Sequential(
            nn.Conv2d(1, n_feature, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
            nn.Conv2d(n_feature, n_feature, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
            nn.Conv2d(n_feature, 1, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1)

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ConvTranspose2d(48, 48, 2, 2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ConvTranspose2d(96, 96, 2, 2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ConvTranspose2d(96, 96, 2, 2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(97, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        # CNN
        x     = self.conv1kernel(x)
        
        # Autoencoder

        pool1     = self.block1(x)
        # print("pool1: ", pool1.shape)
        pool2     = self.block2(pool1)
        # print("pool2: ", pool2.shape)
        pool3     = self.block2(pool2)
        # print("pool3: ", pool3.shape)
        pool4     = self.block2(pool3)
        # print("pool4: ", pool4.shape)
        pool5     = self.block2(pool4)
        # print("pool5: ", pool5.shape)
        upsample5 = self.block3(pool5)
        # print("upsample5: ", upsample5.shape)
        concat5   = torch.cat((upsample5, pool4),1)
        # print("concat5: ", concat5.shape)
        upsample4 = self.block4(concat5)
        # print("upsample4: ", upsample4.shape)
        concat4   = torch.cat((upsample4, pool3), 1)
        # print("concat4: ", concat4.shape)
        upsample3 = self.block5(concat4)
        # print("upsample3: ", upsample3.shape)
        concat3   = torch.cat((upsample3, pool2), 1)
        # print("concat3: ", concat3.shape)
        upsample2 = self.block5(concat3)
        # print("upsample2: ", upsample2.shape)
        concat2   = torch.cat((upsample2, pool1), 1)
        # print("concat2: ", concat2.shape)
        upsample1 = self.block5(concat2)
        # print("upsample1: ", upsample1.shape)
        concat1   = torch.cat((upsample1, x), 1)
        # print("concat1: ", concat1.shape)
        output    = self.block6(concat1)
        # print("output: ", output.shape)
        return output



""" Full assembly of the parts to form the complete network """
from unit_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor     = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        