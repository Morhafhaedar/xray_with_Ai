from json import load
import numpy as np
import torch
from   torchmetrics import StructuralSimilarityIndexMeasure
import torchvision.transforms as transforms
from   skimage.transform import resize, rescale
import torch.nn  as nn
from   imageio import imread, imwrite
import cv2
import glob
from   tqdm import tqdm
from   torch.utils.data import DataLoader
import scipy.fftpack as fp
from   os.path import *

class Convolution_35(nn.Module):
    def __init__(self, kernel_size):
        super(Convolution_35, self). __init__()  # need to do init for the nn.Module
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


        for name, module in self.named_children():
          # print(name, module)
          for name, mo in module.named_children():
            if  isinstance(mo, nn.Conv2d):
              mo.bias.requires_grad= False

    def change_biases(val):
      for name, module in model.named_children():
        # print(name, module)
        for name, mo in module.named_children():
          if  isinstance(mo, nn.Conv2d):
            t = torch.zeros(mo.bias.shape)+ val   
            mo.bias = torch.nn.Parameter(t)


    def forward(self,input):
        x   = input["img"]
        val = input["val"]
        self.change_biases(val)
        # CNN
        x     = self.conv1kernel(x)
        
        # Autoencoder

        pool1 = self.block1(x)
        # print("pool1: ", pool1.shape)
        pool2 = self.block2(pool1)
        # print("pool2: ", pool2.shape)
        pool3 = self.block2(pool2)
        # print("pool3: ", pool3.shape)
        pool4 = self.block2(pool3)
        # print("pool4: ", pool4.shape)
        pool5 = self.block2(pool4)
        # print("pool5: ", pool5.shape)
        upsample5 = self.block3(pool5)
        # print("upsample5: ", upsample5.shape)
        concat5 = torch.cat((upsample5, pool4),1)
        # print("concat5: ", concat5.shape)
        upsample4 = self.block4(concat5)
        # print("upsample4: ", upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), 1)
        # print("concat4: ", concat4.shape)
        upsample3 = self.block5(concat4)
        # print("upsample3: ", upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), 1)
        # print("concat3: ", concat3.shape)
        upsample2 = self.block5(concat3)
        # print("upsample2: ", upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), 1)
        # print("concat2: ", concat2.shape)
        upsample1 = self.block5(concat2)
        # print("upsample1: ", upsample1.shape)
        concat1 = torch.cat((upsample1, x), 1)
        # print("concat1: ", concat1.shape)
        output = self.block6(concat1)
        # print("output: ", output.shape)
        return output


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class Convolution_40(nn.Module):
    def __init__(self, kernel_size):
        super(Convolution_40, self). __init__()  # need to do init for the nn.Module
        n_feature = 8
        self.conv1kernel = nn.Sequential(
            nn.Conv2d(1, n_feature, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
            nn.Conv2d(n_feature, n_feature, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
            nn.Conv2d(n_feature, 1, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
        )

        for name, module in self.named_children():
          # print(name, module)
          for name, mo in module.named_children():
            if  isinstance(mo, nn.Conv2d):
              mo.bias.requires_grad= False

    def change_biases(self,val):
        for name, module in self.named_children():
        # print(name, module)
            for name, mo in module.named_children():
                if  isinstance(mo, nn.Conv2d):
                    t = torch.zeros(mo.bias.shape).to(device)+ val   
                    mo.bias = torch.nn.Parameter(t)
    
    def forward(self,input):
        x   = input["img"]
        val = input["val"]
        self.change_biases(val)
        # CNN
        x     = self.conv1kernel(x)
        return x




