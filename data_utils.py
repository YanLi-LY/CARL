import os
import glob
import h5py
import random
from PIL import Image
from matplotlib import pyplot as plt

import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

from metrics import *
from option import opt


crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

class Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.jpg'):
        super(Dataset,self).__init__()
        self.size=size
        self.train=train
        self.format=format
        self.haze_imgs_dir = os.path.join(path, 'hazy')  # haze_img dir path
        self.haze_imgs_list = os.listdir(self.haze_imgs_dir)  # haze_img name list
        self.haze_imgs = [os.path.join(self.haze_imgs_dir, img) for img in self.haze_imgs_list]  # haze_img path list
        self.clear_dir = os.path.join(path, 'clear')  # clean_img dir path

        self.DCP_dir = ''
        self.length = len(self.haze_imgs_list)
    def __getitem__(self, index):
        haze_1 = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze_1.size[0]<self.size or haze_1.size[1]<self.size :
                index = random.randint(0,self.length)
                haze_1 = Image.open(self.haze_imgs[index])
        haze_1_name = self.haze_imgs[index].split('/')[-1]
        name, extention = os.path.splitext(haze_1_name)
        id = haze_1_name.split('_')[0]
        if self.train:
            clear_name = id + '.png'
        else:
            clear_name = id
        if os.path.exists(os.path.join(self.clear_dir, id + '.jpg')):
            clear = Image.open(os.path.join(self.clear_dir, id + '.jpg'))
        else:
            clear = Image.open(os.path.join(self.clear_dir, id + '.png'))

        clear = tfs.CenterCrop(haze_1.size[::-1])(clear)
        if self.train:
            DCP_dehaze = Image.open(os.path.join(self.DCP_dir, haze_1_name))
            DCP_dehaze = tfs.CenterCrop(haze_1.size[::-1])(DCP_dehaze)
            haze_2_path_list = glob.glob(os.path.join(self.haze_imgs_dir, id + '_*' + self.format))
            haze_2_name = haze_2_path_list[0].split('/')[-1]
            i = 1
            while haze_2_name == haze_1_name:
                haze_2_name = haze_2_path_list[i].split('/')[-1]
                i += 1
            haze_2 = Image.open(os.path.join(self.haze_imgs_dir, haze_2_name))
            haze_2 = tfs.CenterCrop(haze_1.size[::-1])(haze_2)
            if not isinstance(self.size,str):
                i,j,h,w=tfs.RandomCrop.get_params(haze_1,output_size=(self.size,self.size))
                haze_1=FF.crop(haze_1,i,j,h,w)
                haze_2=FF.crop(haze_2, i, j, h, w)
                clear=FF.crop(clear,i,j,h,w)
                DCP_dehaze = FF.crop(DCP_dehaze, i, j, h, w)
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze_1 =self.augData_haze(haze_1.convert("RGB"), rand_hor, rand_rot)
            haze_2 = self.augData_haze(haze_2.convert("RGB"), rand_hor, rand_rot)
            clear = self.augData_clear(clear.convert("RGB"), rand_hor, rand_rot)
            DCP_dehaze = self.augData_clear(DCP_dehaze.convert("RGB"), rand_hor, rand_rot)
            return haze_1, haze_2, clear, DCP_dehaze, haze_1_name, haze_2_name
        else:
            if not isinstance(self.size,str):
                i,j,h,w=tfs.RandomCrop.get_params(haze_1,output_size=(self.size,self.size))
                haze_1=FF.crop(haze_1,i,j,h,w)
                clear=FF.crop(clear,i,j,h,w)
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze_1 = self.augData_haze(haze_1.convert("RGB"), rand_hor, rand_rot)
            clear = self.augData_clear(clear.convert("RGB"), rand_hor, rand_rot)
            return haze_1, clear

    def augData_haze(self, haze, rand_hor, rand_rot):
        if self.train:
            haze=tfs.RandomHorizontalFlip(rand_hor)(haze)
            if rand_rot:
                haze=FF.rotate(haze,90*rand_rot)
        haze=tfs.ToTensor()(haze)
        haze=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(haze)
        return haze
    def augData_clear(self, clear, rand_hor, rand_rot):
        if self.train:
            clear=tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                clear=FF.rotate(clear,90*rand_rot)
        clear=tfs.ToTensor()(clear)
        return clear
    def __len__(self):
        return self.length

path='./data/'

train_loader=DataLoader(dataset=Dataset(path+'/RESIDE/ITS',train=True,size=crop_size),
                            batch_size=opt.bs, shuffle=True)
test_loader=DataLoader(dataset=Dataset(path+'/RESIDE/SOTS-Indoor',train=False,size='whole img'),batch_size=1,shuffle=False)

