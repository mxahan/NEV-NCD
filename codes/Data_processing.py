#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:33:09 2021

@author: zahid
"""
#%% libraries load


#import tensorflow as tf

import os
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#os.environ['KMP_DUPLICATE_LIB_OK']='True'


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



from threading import Thread
import sys 

from queue import Queue

from imutils.video import FileVideoStream

import numpy as np
import argparse
import imutils
import time

#%% Directory load data

files = []
# change the path_dir file
path_dir = '../../../../Dataset/ARL_MULTIVIEW_AR/nirandi_8_26_21_2/' # change to folder

# No change here

save_path = path_dir + '_160_160.pkl'




dataPath = os.path.join(path_dir, '*.mp4')
files = glob.glob(dataPath)  # care about the serialization


dataPath = os.path.join(path_dir, '*.MP4')

files.extend(glob.glob(dataPath))  # care about the serialization
# check files [ac, sp, dr]

#%% Import data


# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# https://data-flair.training/blogs/python-project-real-time-human-detection-counting/


def get_all_frame(files, im_size = (400,400),  slice_info = None, pdb_info = None):
    data = []
    
    cap = cv2.VideoCapture(files)
    
    import pdb
    
    time.sleep(1.0)
    fps = FPS().start()
    
    while(cap.isOpened()):
        ret, gray = cap.read()
        
        if ret==False:
            break
        
        
        if pdb_info!=None:
            pdb.set_trace()
            
            
            
        if slice_info==None:
            gray =  gray[:,:,:]
            
        else: 
            gray = gray[slice_info]
 
        gray = cv2.resize(gray, im_size) 
        
        # pdb.set_trace()
        
        # gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        
        # boxes, weights = hog.detectMultiScale(gray, winStride=(4,4), scale = 1.03 ) #may add padding
        # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        # for (xA, yA, xB, yB) in boxes:
        #     cv2.rectangle(gray, (xA, yA), (xB, yB),
        #                       (0, 255, 0), 2)
        
        
        data.append(gray)
        
        # pdb.set_trace([]
    
        cv2.imshow('frame', gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    
    
    fps = cap.get(cv2.CAP_PROP_FPS)
        
    cap.release()
    cv2.destroyAllWindows()
    data =  np.array(data)
    return data

#%% Debug mode

# data = get_all_frame(files[0], slice_info=None, pdb_info=None)
### use pdb true to get the crop information
# plt.imshow(data[120])
# plt.imshow(gray)

#%% Run the Main section

data =[]

stFr = [1287, 372, 457] # starting frames (ac, sp, dr)

slice_info =  [(slice(100, 950), slice(300, 1700), slice(0,3)), None, (slice(150, 950), slice(200,1600), slice(0,3))]


for i in range(3):
    data.append(get_all_frame(files[i], im_size=(160,160), slice_info=slice_info[i]))
    
    

for i in range(3):
    data[i]= data[i][stFr[i]:]

#plt.imshow(data[1][5000])

#%% Data Save Option

# https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/

## Quick save an load "PICKLE"
input("pickle save ahead")
# Saving 



with open(save_path, 'wb') as f:
    pickle.dump(data, f)  #NickName: SAD


# Loading 

with open(save_path, 'rb') as f:
    sp, ac, dr = pickle.load(f)
    