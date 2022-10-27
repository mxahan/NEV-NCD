# Import libraries 


import os
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch 

import pickle 
import matplotlib.pyplot as plt

import numpy as np

import cv2

import  time
import glob

from scipy.io import loadmat

import random

from random import seed, randint

from sklearn.model_selection import train_test_split

import pandas as pd

from imutils.video import FPS

import imutils

from torch.utils.data import DataLoader, Dataset


from threading import Thread
import sys 

from queue import Queue

from imutils.video import FileVideoStream

import numpy as np
import argparse
import imutils
import time
#%% Augmentations


# Brighness augmentation video, input shape # frame number, H, W, C

def brightness_augment(img, factor=0.5): 
    rgb = img.copy()
    for i in range(img.shape[0]):
        hsv = cv2.cvtColor(img[i], cv2.COLOR_RGB2HSV) #convert to hsv
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor) #scale channel V uniformly
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
        rgb[i] = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb




# Salt and Papper Noise 
import PIL

class SaltAndPepperNoise(object):
    r""" Implements 'Salt-and-Pepper' noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)
    """
    def __init__(self,
                 treshold:float = 0.005,
                 imgType:str = "cv2",
                 lowerValue:int = 5,
                 upperValue:int = 250,
                 noiseType:str = "SnP"):
        self.treshold = treshold
        self.imgType = imgType
        self.lowerValue = lowerValue # 255 would be too high
        self.upperValue = upperValue # 0 would be too low
        if (noiseType != "RGB") and (noiseType != "SnP"):
            raise Exception("'noiseType' not of value {'SnP', 'RGB'}")
        else:
            self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img1):
        img = img1.copy()
        if self.imgType == "PIL":
            img = np.array(img)
        if type(img) != np.ndarray: raise TypeError("Image is not of type 'np.ndarray'!")
        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0],img.shape[1])
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)      
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        if self.imgType == "cv2": return img
        elif self.imgType == "PIL": return PIL.Image.fromarray(img)

# Define the SNP noises for video 

def snp_RGB(vid_fr):
    vid_sn = vid_fr.copy()
    RGB_noise = SaltAndPepperNoise(noiseType="RGB")
    for i in range(vid_fr.shape[0]): vid_sn[i] = RGB_noise(vid_fr[i])
    return vid_sn 

#%% Pytorch DataLoader [very Crutial]
# mode_no = 'InfoNCE', 'Triplet', 'SimCLR', 'AlUn', TCLR, NCDSUP

class data_set_Prep(Dataset):
    def __init__(self, vid_file, frm_no = 16, mode_no = 'InfoNCE', samp_siz = 5, S = set([0,1, 2]), l_info = (None, None), SS = None):
        self.frm_no = frm_no
        self.temp = vid_file
        self.mini =  min(len(vid_file[0]), len(vid_file[2]), len(vid_file[1])) - 150 
        self.mode_no = mode_no
        self.samp_siz =  samp_siz
        self.S = S
        self.SS = SS
        self.gt_id =  l_info[0]
        self.gt_name =  l_info[1]
        self.LabelDict= {'sitting':0, 'standing':1, 'lying_d':2, 'lying_u':3, 'walking':4, 'push_up':5,
                         'object_walk':6, 'object_pick':7, 'hand_wave':8, 'leg_exer':9, 'what':10}

        
    def __len__(self):
        return self.mini
        
    def __getitem__(self,idx):
        if self.mode_no == 'Triplet':
            return self.get_data_infoNCE(neg_num = 1)
        elif self.mode_no =='InfoNCE':
            return self.get_data_infoNCE(neg_num = self.samp_siz)
        elif self.mode_no == 'AlUn':
            return self.get_al_un(samp_siz =self.samp_siz)
        elif self.mode_no == 'SupSet':
            return self.get_sup_data(batch = self.samp_siz)
        elif self.mode_no == 'AdvNCE':
            return self.get_data_info_cam(neg_num = self.samp_siz)
        elif self.mode_no == 'TCLR':
            return self.get_data_infoNCE_TCLR(neg_num = self.samp_siz)
        elif self.mode_no == 'NCDSUP':
            return self.get_NCD_sup(batch = self.samp_siz, SS = self.SS)
        elif self.mode_no == 'NCD_info':
            return self.get_data_info_cam(neg_num = self.samp_siz, SS = self.SS)
        elif self.mode_no == 'Neg_learn':
            return self.get_data_neg_learn(neg_num = self.samp_siz, SS = self.SS)
        else:
            return self.get_SIMCLR_data(samp_siz = self.samp_siz)
    
    def get_data_infoNCE(self, neg_num = 8):
        aps = random.sample(self.S,2)
        sample = []
        anchor, idx = self.get_anchor(aps = aps[0])
        sample.append(anchor)
        if np.random.uniform()>(0.1-np.exp(-10)): sample.append(self.get_anchor(aps = aps[1], idx = idx))
        else:
            sample.append(self.pos_aug(anchor))
        for i in range(neg_num):
            sample.append(self.info_neg(idx = idx, aps = None))
        #ret = np.moveaxis(np.stack((Anchor, Pos, Neg, Neg01,  Neg02, Neg1, Neg11, Neg2, Neg3, Neg4, Neg5, Neg51), axis = 0), -1, -4)    
        ret = np.moveaxis(np.stack(sample), -1, -4)
        return ret.astype(np.float32)/255.
    
    def get_data_infoNCE_TCLR(self, neg_num = 8):
        aps = random.sample(self.S,2)
        sample, cam = [], []
        anchor, idx = self.get_anchor(aps = aps[0]) 
        cam.append(aps[0])
        sample.append(anchor)
        if np.random.uniform()>(0.1-np.exp(-10)): sample.append(self.get_anchor(aps = aps[0], idx = idx+random.randint(1, 60))); cam.append(aps[0])
        else:
            sample.append(self.pos_aug(anchor)); cam.append(aps[0])
        for i in range(neg_num):
            sample.append(self.info_neg(idx = idx, aps = aps[0])); cam.append(aps[0])
            
        ret = np.moveaxis(np.stack(sample), -1, -4)
        cam = np.stack(cam)
        return ret.astype(np.float32)/255., cam
    
    def info_neg(self, idx, aps = None):
        idx_n =  randint(0, self.mini)
        while abs(idx-idx_n)<1200: idx_n = randint(0, self.mini)
        if aps == None: aps = random.sample(self.S,1)[0]
        return self.get_anchor(aps = aps, idx = idx_n)
    
    def pos_aug(self, anchor):
        ps = randint(0,2)
        if ps ==1: Pos = np.flip(anchor, axis = 2)
        elif ps == 2: Pos = brightness_augment(anchor, factor = 1.5)
        else: Pos = snp_RGB(anchor)
        return Pos
    
    def get_anchor(self, aps = 0, idx = None, smp_rte = 1):
        if np.random.uniform()>0.95: smp_rte = 2
        if idx == None:
            idx = randint(0,self.mini)
            anchor  = self.temp[aps][idx:idx+self.frm_no*smp_rte: smp_rte]
            return anchor, idx
        else:
            return self.temp[aps][idx:idx+self.frm_no *smp_rte : smp_rte]
    
    def get_SIMCLR_data(self, samp_siz):
        a = self.random_sample(0, self.mini, 1200, samp_siz = samp_siz); s1, s2 = [], []
        s1, s2, cam = self.create_two_set(a)
        s1, s2 = np.moveaxis(np.stack(s1), -1, -4), np.moveaxis(np.stack(s2), -1, -4)
        cam = np.stack(cam)
        return s1.astype(np.float32)/255., s2.astype(np.float32)/255., cam
        
    def random_sample(self, mn, mx, df = 1200, samp_siz= 2):
        a = np.sort((random.sample(range(mn, mx), samp_siz))); count = 0;
        while sum(np.diff(a)<df)>0:
            b = np.where(np.diff(a)<1200)
            for i in b: a[i] = randint(mn,mx)
            a  = np.sort(a); count = count +1  
            if count > 50: print("Did Not get the right Negatives for SIMCLR"); break      
        return a
    
    def get_al_un(self, samp_siz):
        a = random.sample(range(self.mini), samp_siz); s1, s2 = [], []
        s1, s2, _ = self.create_two_set(a)
        s1, s2 = np.moveaxis(np.stack(s1), -1, -4), np.moveaxis(np.stack(s2), -1, -4)
        return s1.astype(np.float32)/255., s2.astype(np.float32)/255.
    
    
    def create_two_set(self, a):
        s1, s2, cam = [], [], []
        for i in a:
            aps = random.sample(self.S, 2)
            anchor = self.get_anchor(aps = aps[0], idx = i)
            s1.append(anchor); cam.append(aps[0])       
            if np.random.uniform()>(0.1-np.exp(-10)): s2.append(self.get_anchor(aps = aps[1], idx = i)); cam.append(aps[1])
            else: 
                s2.append(self.pos_aug(anchor)); cam.append(aps[0])
        return s1, s2, cam
    
    def get_sup_data(self, batch):
        sample, idx = [], []
        for _ in range(batch):
            aps = random.sample(self.S,1)
            anchor, idx_v = self.get_anchor(aps[0])
            if np.random.uniform()>(0.9): anchor = self.pos_aug(anchor)
            sample.append(anchor)
            if (self.gt_id == None).all():
                idx.append(idx_v)
            else:
                idx.append(self.get_label_name(idx_v))
        sample = np.moveaxis(np.stack(sample), -1, -4)
        idx = np.stack(idx)
        return sample.astype(np.float32)/255., idx

    def get_NCD_sup(self, batch, SS):
        sample, idx = [], []
        if SS == None:
            print("SS is NONE")
            return None
        for _ in range(batch):
            aps = random.sample(self.S,1)
            idx_v = randint(0,self.mini)
            while self.get_label_name(idx_v) not in SS:
                idx_v = randint(0,self.mini)
                
            anchor = self.get_anchor(aps[0], idx_v)
            
            if np.random.uniform()>(0.9): anchor = self.pos_aug(anchor)
            sample.append(anchor)
            if (self.gt_id == None).all():
                idx.append(idx_v)
            else:
                idx.append(self.get_label_name(idx_v))
        sample = np.moveaxis(np.stack(sample), -1, -4)
        idx = np.stack(idx)
        return sample.astype(np.float32)/255., idx
    
    def get_label_name(self, idx): # send [idx[0]]
        return self.LabelDict[self.gt_name[np.where(self.gt_id == self.gt_id[self.gt_id<=idx][-1])[0][0]]]
    
    def get_data_info_cam(self, neg_num= 8, SS = None):    
        aps = random.sample(self.S,2)
        sample, cam, gt = [], [], []
        
        
        if self.mode_no != 'NCD_info':
            anchor, idx = self.get_anchor(aps = aps[0])
        else:
            idx = randint(0,self.mini)
            while self.get_label_name(idx) not in SS:  
                idx = randint(0,self.mini)
            anchor = self.get_anchor(aps[0], idx)
        
        gt.append(self.get_label_name(idx))
            
        cam.append(aps[0])
        sample.append(anchor)
        if np.random.uniform()>(0.1-np.exp(-10)): sample.append(self.get_anchor(aps = aps[1], idx = idx)); cam.append(aps[1]);
        else:
            sample.append(self.pos_aug(anchor))
            cam.append(aps[0])
            
        gt.append(self.get_label_name(idx))
        
        for i in range(neg_num):
            aps = random.sample(self.S, 1)[0]
            sample.append(self.info_neg(idx = idx, aps = aps))
            cam.append(aps)
            
        ret = np.moveaxis(np.stack(sample), -1, -4)
        cam = np.stack(cam)
        gt = np.stack(gt)
        return ret.astype(np.float32)/255., cam, gt
    
    def get_data_neg_learn(self, neg_num= 8, SS = None):    
        aps = random.sample(self.S,2)
        sample, cam, gt = [], [], []
        
        
        if np.random.uniform()>(0.5-np.exp(-10)):
            idx = randint(0,self.mini)
            while self.get_label_name(idx) not in SS:  
                idx = randint(0,self.mini)
            anchor = self.get_anchor(aps[0], idx)
        
            gt.append(self.get_label_name(idx))

            cam.append(aps[0])
            sample.append(anchor)
            if np.random.uniform()>(0.1-np.exp(-10)): sample.append(self.get_anchor(aps = aps[1], idx = idx)); cam.append(aps[1]);
            else:
                sample.append(self.pos_aug(anchor))
                cam.append(aps[0])

            gt.append(self.get_label_name(idx))

            for i in range(neg_num):
                aps = random.sample(self.S, 1)[0]
                sample.append(self.info_neg(idx = idx, aps = aps))
                cam.append(aps)
        else: 
            if np.random.uniform()>(0.5-np.exp(-10)):
                idx = randint(0,self.mini)
                while self.get_label_name(idx) in SS:  
                    idx = randint(0,self.mini)
                anchor = self.get_anchor(aps[0], idx)

                gt.append(1000)

                cam.append(aps[0])
                sample.append(anchor)
                if np.random.uniform()>(0.95-np.exp(-10)): sample.append(self.get_anchor(aps = aps[1], idx = idx)); cam.append(aps[1]);
                else:
                    sample.append(self.pos_aug(anchor))
                    cam.append(aps[0])

                gt.append(1000)

                for i in range(neg_num):

                    aps = random.sample(self.S,2)

                    idx = randint(0,self.mini)
                    while self.get_label_name(idx) not in SS:  
                        idx = randint(0,self.mini)

                    sample.append(self.get_anchor(aps[0], idx))
                    cam.append(aps[0])
                    
            else:
                for i in range(24):
                    aps = random.sample(self.S,2)
                    idx = randint(0,self.mini)
                    while self.get_label_name(idx) in SS:  
                        idx = randint(0,self.mini)

                    sample.append(self.get_anchor(aps[0], idx))
                    cam.append(aps[0])
                    gt.append(10000)
                
            
            
        
        ret = np.moveaxis(np.stack(sample), -1, -4)
        cam = np.stack(cam)
        gt = np.stack(gt)
        return ret.astype(np.float32)/255.0, cam, gt

    
# change the data shape using torch.moveaxis, numpy.moveaxis
# https://pytorch.org/docs/stable/generated/torch.moveaxis.html
# https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html

class test_data():
    def __init__(self, temp, frm_no = 16):
        self.frm_no = frm_no
        self.temp = temp
        self.mini =  min(len(temp[0]), len(temp[2]), len(temp[1]))- 80
    def get_data_test(self, idx= None):
        if idx == None: idx = randint(0,self.mini)    
        x_v1, x_v2, x_v3 = self.temp[0][idx:idx+self.frm_no], self.temp[1][idx:idx+self.frm_no], self.temp[2][idx:idx+self.frm_no]
        ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3), axis = 0), -1, -4)
        return ret.astype(np.float32)/255.0