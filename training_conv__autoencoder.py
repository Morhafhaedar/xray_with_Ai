# -*- coding: utf-8 -*-

import torch.nn  as nn
import torch
import numpy as np
import torch
from   torch.utils.data import Dataset
from   skimage import io
from   sklearn.model_selection import train_test_split
import cv2
import glob
from   json import load
import numpy as np
import torch
from   torchmetrics import StructuralSimilarityIndexMeasure
import torchvision.transforms as transforms
from   skimage.transform import resize, rescale
import torch.nn  as nn
import cv2
import glob
from   tqdm import tqdm
import matplotlib.pyplot as plt
# from   convolution_net import UNet, Convolution_NxN, Convolution_3
# from   customDatset import Custom_Data
from   torch.utils.data import DataLoader
import copy
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("cuda")
else:
    device = torch.device('cpu')
    print("cpu")

torch.manual_seed(0)

class Convolution_4(nn.Module):
    def __init__(self, kernel_size):
        super(Convolution_4, self). __init__()  # need to do init for the nn.Module

    
        n_feature = 8

        self.conv1kernel = nn.Sequential(
            nn.Conv2d(1, n_feature, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
            nn.Conv2d(n_feature, n_feature, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
            nn.Conv2d(n_feature, 1, kernel_size, 1, padding=(int(kernel_size/2), int(kernel_size/2)), bias=True),
            nn.PReLU(),
        )


    def forward(self,x):
        # CNN
        x     = self.conv1kernel(x)
        
        # Autoencoder
        
        return x

class Convolution_xray(nn.Module):
    def __init__(self, kernel_size):
        super(Convolution_xray, self). __init__()  # need to do init for the nn.Module

    
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

class Custom_Data(Dataset):
    def __init__(self, truth_dir: str, blured_dir: str, val, test, transform=None) -> None:
        self.truth_dir       = truth_dir
        self.blured_dir      = blured_dir
        self.transform       = transform
        print("Data Loading ... ")
        self.target_image    = self.data_loader(self.truth_dir, val, test)
        self.input_image     = self.data_loader(self.blured_dir, val, test)

    def data_loader(self,path, val=False, test=False):
        paths = glob.glob(path)
        if test:
            file  = sorted(paths)[:60]

        else:
          if val :
            file  = sorted(paths)[:2]#[int(len(paths)*0.8):]
          else:
            file  = sorted(paths)[:6]#[:int(len(paths)*0.8)]
        data_list = []
        for image in tqdm(file):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = np.array(image, dtype= np.float64)
            image = image/255
            image = image.reshape(1,image.shape[0], image.shape[1])
            data_list.append(image)
        
        return np.array(data_list)
    
    def __len__(self):
        return len(self.target_image)
    
    def __getitem__(self, index):
        if self.transform:
            self.input_image[index] = self.transform(self.input_image[index])
            self.target_image[index]= self.transform(self.target_image[index])
        sample = {'target':self.target_image[index], 'input':self.input_image[index]}
        return sample
    
    def normalized(self):
        self.target_image     = self.target_image/255
        self.input_image      = self.input_image/255

def test_loader(path):
    file      = sorted(glob.glob(path))
    data_list = []
    for image in file:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        data_list.append(image)
    return data_list

def load_data(truth_d, deblured_d, val=False, test=False):
    data_set      = Custom_Data(truth_d, deblured_d, val = val, test=test
                          ) 
    data_loader = DataLoader(dataset= data_set ,batch_size=1, shuffle=True)
    return data_loader

class MyModel():
  def __init__(self, model, loss_func, optimizer):
    self.model = model
    self.loss_func = loss_func
    self.optimizer = optimizer
    self.old_val_loss = 10000
    self.best_model_val = 10000
    self.best_model = None
    self.epoch = 0
    self.train_losses = []
    self.val_losses = []

  def train(self, epochs, train_data, val_data, val_wait=10):
      stamp = time.time()
      train_loss = 0.0
      val_step = 0

      for epoch in tqdm(range(self.epoch, self.epoch+epochs)):
        for i, sample in enumerate(train_data):

            x      = sample['input'].to(device) 
            y      = sample['target'].to(device)
            y      = y.float() 
            x      = x.float()
            output = self.model(x)
            optimiser.zero_grad() 
            loss   = self.loss_func(output, y)
            train_loss += loss
            loss.backward()                                                              
            optimiser.step() 
            # output_cpu = output.to('cpu')
            # cnn_output = np.squeeze(output_cpu.detach().numpy())

        val_loss = self.test(val_data)

        if self.old_val_loss < val_loss:
          val_step +=1
        else:
          val_step = 0 

        self.old_val_loss = val_loss

        print( f'epoch {epoch+1}: train_loss = {train_loss:.8f}, val_loss = {val_loss:.8f}')
        train_loss = train_loss.to('cpu')
        self.train_losses.append(np.squeeze(train_loss.detach().numpy()))
        val_loss   = val_loss.to('cpu')
        self.val_losses.append(val_loss.detach().numpy())
        train_loss = 0.0

        if val_loss < self.best_model_val:
          self.best_model_val = val_loss
          self.best_model = copy.deepcopy(self.model)
          torch.save({"model_state_dict":self.best_model.state_dict(), "epoch":epoch+1, "optimizer_state_dict": self.optimizer.state_dict(),
                      "loss_func":self.loss_func, "best_model_val":self.best_model_val}, f"the best model--{self.best_model_val}--{str(stamp)}.pth") 
          print(f"saving model.... model_val {self.best_model_val}")


          

        if val_step >=val_wait:
          print(f"Break because the val_loss not decreasing since {val_step} epoch, the best model achieve {self.best_model_val} on validation data.")
          
          break

      self.epoch =epoch+1
      # self.train_losses = self.train_losses.detach().numpy()
      # self.val_losses = self.val_losses.detach().numpy()
      # print(self.train_losses)
      # print(self.val_losses)
      
      plt.plot(range(len(self.train_losses)), self.train_losses,label='train_loss')
      plt.plot(range(len(self.val_losses)), self.val_losses,label='val_loss')
      plt.savefig(f"train{epoch}--{str(stamp)}.png")
      plt.show()
      print('Training is done !!')

  def test(self, test_data, plot=False):
    test_loss = 0.0
    with torch.no_grad():
      for i, sample in enumerate(test_data):   
        x = sample["input"].to(device)
        y = sample["target"].to(device)
        x = x.float()
        y = y.float()
        output = self.model(x)
        loss = self.loss_func(output, y)
        test_loss += loss
        output_cpu = output.to('cpu')
        cnn_output = np.squeeze(output_cpu.detach().numpy())
        cv2.imwrite(outpath + f'{str(1+i)}.png', (cnn_output*255).astype(np.uint8))
        if plot:
          for i, out in enumerate(output):
            fig = plt.figure(figsize=(15, 15))

            ax = fig.add_subplot(1, 3, 1)
            imgplot = plt.imshow(x.to('cpu')[i][0] * 255, cmap="gray")
            ax.set_title('input')
            out = out.to('cpu')
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow((out*255)[0], cmap="gray")        
            ax.set_title('cnn_output')

            ax = fig.add_subplot(1, 3, 3)
            plt.imshow(y.to('cpu')[i][0] * 255, cmap="gray")        
            ax.set_title('expected output')

            plt.show()
    
    return  test_loss 


  def load_model(self, path):
    checkpoint = torch.load(path)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.epoch = checkpoint['epoch']
    self.loss_func = checkpoint['loss_func']
    self.best_model_val = checkpoint['best_model_val']

"""# Load Data"""

outpath               = 'C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/xray_Ai_retoation/output/Prep' 

gt_dir                = 'C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Kernel Training/Kernel_Training-1/xray/target_images/*.png'
filter_dir            = 'C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Kernel Training/Kernel_Training-1/xray/training/*.png'

test_dir              = "C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Kernel Training/Kernel_Training-1/xray/training/*.*"
test_gt_dir           = "C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Kernel Training/Kernel_Training-1/xray/target_images/*.png"

epochs                = 550
training_data         = load_data(gt_dir, filter_dir, val=True)
val_data              = load_data(gt_dir, filter_dir, val=True)
test_data             = test_loader(test_dir)        #load_data(test_gt_dir, test_dir, test=True)

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(1, 2, 1)
data = iter(training_data).next()
imgplot = plt.imshow(data["input"][0][0], cmap="gray")
ax.set_title('input ..... training')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(data["target"][0][0], cmap="gray")
ax.set_title('target .... training')

fig = plt.figure(figsize=(15, 15))
data = iter(val_data).next()

ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(data["input"][0][0], cmap="gray")
ax.set_title('input ... val')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(data["target"][0][0], cmap="gray")
ax.set_title('target .... val')

fig = plt.figure(figsize=(15, 15))
#data = iter(test_data).next()

ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(data["input"][0][0], cmap="gray")
ax.set_title('input ... test')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(data["target"][0][0], cmap="gray")
ax.set_title('target .... test')

"""# Initial model and train"""

model                 = Convolution_xray(3).to(device)
loss_func             = nn.MSELoss()
optimiser             = torch.optim.Adam(model.parameters(), lr = 0.007)

my_model = MyModel(model, loss_func, optimiser)

my_model.train(epochs=55, train_data=training_data, val_data=val_data, val_wait=2)

my_model.test(test_data, plot=False)

# #---------------------------loss function------------------------#
# loss_func = nn.MSELoss() 
# def custom_loss_function(a, pred, truth):
#     l1    = nn.L1Loss()
#     ssim  = StructuralSimilarityIndexMeasure()
#     l_mix = a*l1(pred, truth)+(1-a)*ssim(pred,truth)
#     return l_mix