import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import cv2
import glob
import json


# this is a templat do not put a finger on it
class Custom_Data(Dataset):
    def __init__(self, truth_dir: str, blured_dir: str, transform=None) -> None:
        self.truth_dir       = truth_dir
        self.blured_dir      = blured_dir
        self.transform       = transform
        self.target_image    = self.data_loader(self.truth_dir)
        self.input_image     = self.data_loader(self.blured_dir)

   
    def data_loader(self,path):
        file      = sorted(glob.glob(path))
        data_list = []
        for image in file:
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = np.array(image, dtype= np.float64)
            image = image/255
            image = image.reshape(1,image.shape[0], image.shape[1])
            data_list.append(image)
        return data_list
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

# this class is for custom conv with angle

class Custom_Data2(Dataset):
    def __init__(self, truth_dir: str, blured_dir: str, angle, transform=None) -> None:
        self.truth_dir       = truth_dir
        self.blured_dir      = blured_dir
        self.transform       = transform
        self.angle           = angle
        self.target_image    = self.data_loader(self.truth_dir)
        self.input_image     = self.data_loader(self.blured_dir) 
        self.kernel_angle    = self.angle_loader(self.angle)

    def angle_loader(self, json_file: str):
        f      = open(json_file)
        ang    = json.load(f)
        angles = {}
        for item in ang["angle"]:
            angles = item
        return angles

    def data_loader(self,path):
        file      = sorted(glob.glob(path))
        data_list = []
        for image in file:
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = np.array(image, dtype= np.float64)
            image = image/255
            image = image.reshape(1,image.shape[0], image.shape[1])
            data_list.append(image)
        return data_list
    def __len__(self):
        return len(self.target_image)
    def __getitem__(self, index):
        if self.transform:
            self.input_image[index] = self.transform(self.input_image[index])
            self.target_image[index]= self.transform(self.target_image[index])
        sample = {'target':self.target_image[index], 'input':self.input_image[index], "angle": self.kernel_angle[index] }
        return sample
    def normalized(self):
        self.target_image     = self.target_image/255
        self.input_image      = self.input_image/255


