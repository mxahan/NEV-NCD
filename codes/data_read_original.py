# Import libraries 

#%% libraries


#import tensorflow as tf

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

#%% Directory load data

files = []

path_dir = '../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27/'

dataPath = os.path.join(path_dir, '*.mp4')
files = glob.glob(dataPath)  # care about the serialization


dataPath = os.path.join(path_dir, '*.MP4')

files.extend(glob.glob(dataPath))  # care about the serialization

#%% Import data


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# https://data-flair.training/blogs/python-project-real-time-human-detection-counting/


def get_all_frame(files, im_size = (400,400),  slice_info = None):
    data = []
    
    cap = cv2.VideoCapture(files)
    
    import pdb
    
    time.sleep(1.0)
    fps = FPS().start()
    
    while(cap.isOpened()):
        ret, gray = cap.read()
        
        if ret==False:
            break
    
        # pdb.set_trace()
        if slice_info==None:
            gray =  gray[:,:,:]
        else: 
            gray  = gray[slice_info]
        # gray[150:900,410:1200,:]     
       
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


data =[]
slice_info =  [(slice(10, 1050), slice(600, 1800)), None, (slice(150, 900), slice(410,1200))]

for i in range(3):
    data.append(get_all_frame(files[i], im_size=(160,160), slice_info=slice_info[i]))
    
    

stFr = [5808, 1016, 382]

for i in range(3):
    data[0]= data[0][stFr[i]:]

#%% Data Analysis

print(data.nbytes)
print(data.shape) 

# dr_data = data[382:]


#sp_data
#ac_data
#dr_data


# data_pre =  [sp_data, ac_data, dr_data] # Keep the order as it is


#%% Data Save Option

# https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/

## Quick save an load "PICKLE"
input("pickle save ahead")
# Saving 


with open('../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27_160_160.pkl', 'wb') as f:
    pickle.dump(data_temp, f)  #NickName: SAD


# Loading 

with open('../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27_160_160.pkl', 'rb') as f:
    sp, ac, dr = pickle.load(f)
    
    
## Data Frame Option (not good option as CSV turns things to string)

# dict = {'sp': [sp_data], 'ac': [ac_data], 'dr': [dr_data]}
# df = pd.DataFrame(dict)

# df.to_csv('file_name.csv')

# df1 =  pd.read_csv('file_name.csv')

# sp = np.array(df1['sp'])


#device = torch.device('cuda:0')

#%% Prepare Data Loader

#%% Cuda Determinstic 

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
seed = 42 # any number 
set_deterministic(seed=seed)

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
    Adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values = white, low values = black
    
    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with 
                                               noise added
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
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")
        
        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0],img.shape[1])
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)      
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        
        

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)

# Define the SNP noises for video 

def snp_RGB(vid_fr):
    vid_sn = vid_fr.copy()
    RGB_noise = SaltAndPepperNoise(noiseType="RGB")
    for i in range(vid_fr.shape[0]):
        vid_sn[i] = RGB_noise(vid_fr[i])
    return vid_sn 
    
#%% Pytorch DataLoader [very Crutial]


class dataPrep(Dataset):
    def __init__(self, root_dir):
        self.all_cam = root_dir
        
    def __len__(self):
        return len(self.all_cam[0])
        
    def __getitem__(self,idx):
        # x = self.all_cam[randint(0,2)][idx:idx+16]
        # x = self.all_cam[0][idx:idx+16]
        # y = self.all_cam[1][idx:idx+16]
        # z = self.all_cam[2][idx:idx+16]
        xx = self.get_data1()
        return xx
    
    def get_data1(self):
        
        # Time Positives

        idx = randint(0,len(temp[0])-100)

        
        
        x_v1 = temp[0][idx:idx+16]
        x_v2 = temp[1][idx:idx+16]
        x_v3 = temp[2][idx:idx+16]
        
        # Augmentation Positive
        
        # Horizontal flip 
        x_v1_hf = np.flip(x_v3, axis = 2)
        x_v1_br = brightness_augment(x_v1, factor = 1.5)
        
        x_v1_snp =  snp_RGB(x_v1)
        
        
        # Time Negatives 
        idx_n =  randint(0,len(temp[0])-100)
        while abs(idx-idx_n)<1200: idx_n = randint(0,len(temp[0])-100)
        
        xNIra =  temp[0][idx_n:idx_n+16] # intra negative 
        
        # Augmentation Negative 
        
        # use tensor append option
        
        # ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3, x_v1_hf, x_v1_br, x_v1_snp, xNIra), axis = 0), -1, -4)
        
        ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3), axis = 0), -1, -4)
        return ret.astype(np.float32)/255.
    
# change the data shape using torch.moveaxis, numpy.moveaxis
# https://pytorch.org/docs/stable/generated/torch.moveaxis.html
# https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html




# sampling for SIMCLR



#%% Dataset Loader 

with open('../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27_160_160.pkl', 'rb') as f:
    temp = pickle.load(f)

dataSet =  dataPrep(temp)

data_loader = DataLoader(dataSet, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


#%% Dataloader Check 

sample = next(iter(data_loader))


for i in data_loader:
    break

#%% Random Loss function (add all the loss functions)
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()   




class ContrastiveLoss2(nn.Module):
    None
    
    
    
    
loss = ContrastiveLoss(0.5)


#%% Network C3D

from C3D_model import C3D
from p3D_net import P3D63, P3D131, P3D199
from torchsummary import  summary

#%% Input Model name

ModName = input("Input your model name C3D, P3D199, P3D63, P3D131 \n")

def str_to_class(ModelName):
    return getattr(sys.modules[__name__], ModelName)

ModelName = str_to_class(ModName)

if ModName == 'C3D':
    net = ModelName()
    net.load_state_dict(torch.load('../../Saved_models/c3d.pickle'))
    net = net.cuda()
    input_shape = (3,16,112,112)
    print(summary(net, input_shape))

elif ModName == 'P3D199':
    net = ModelName(True, 'RGB',num_classes=400)
    # net = net.cuda()
    # input_shape = (3,16,160,160)
    # print(summary(net, input_shape))

elif ModName == 'P3D63' or ModName == 'P3D131':
    net = ModelName(num_classes=400)
    # net = net.cuda()
    # input_shape = (3,16,160,160)
    # print(summary(net, input_shape))
    
    more network here!!
    SLOWFAST


#%% Removing layer from the mdoel 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.in_features=2048

    def forward(self, x):
        return x
# net.avgpool = Identity() // OR linear!! 
    
net.fc = Identity()

## good link : https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
## https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6


#%% Optimizer 

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

#%% Data inference

# jj =  (i[1].float()/255.0).cuda()



# https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/15

# with torch.no_grad():
#%% 

z= torch.from_numpy(np.array(1))

z = z.cuda()



for sample in data_loader:    
    sample = sample.cuda()
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(sample[0])
    loss1 = loss(output[:1], output[1:2],z)
    loss1.backward()
    optimizer.step()   #%%
    print('jj /n')





#%% Image Ploting  (nothing to do with the code)

sample = next(iter(data_loader))

jj =  (sample[0]).cuda()
plt.imshow(np.moveaxis((jj[0,:,0,:,:]).cpu().numpy(), 0,2), vmin=0., vmax=1.)

#%% Clear Memory for torch (GPU memory release) (nothing to do with the code)
# del output
# del jj 
# del net

torch.cuda.empty_cache()

# check  devices 
next(net.parameters()).is_cuda
next(net.parameters()).device

#%% Manual test DataLoader (May Not useful)

def get_data_test(idx= None):
    
    # Time Positives
    if idx ==None:
        idx = randint(0,len(temp[0])-100)
        
    x_v1 = temp[0][idx:idx+16]
    x_v2 = temp[1][idx:idx+16]
    x_v3 = temp[2][idx:idx+16]
    
    # Augmentation Positive
    # Horizontal flip 
    x_v1_hf = np.flip(x_v3, axis = 2)
    x_v1_br = brightness_augment(x_v1, factor = 1.5)
    
    x_v1_snp =  snp_RGB(x_v1)
    # Time Negatives 
    idx_n =  randint(0,len(temp[0])-100)
    while abs(idx-idx_n)<1200: idx_n = randint(0,len(temp[0])-100)
    
    xNIra =  temp[0][idx_n:idx_n+16] # intra negative 
    
    # Augmentation Negative 
    # use tensor append option
    # ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3, x_v1_hf, x_v1_br, x_v1_snp, xNIra), axis = 0), -1, -4)
    
    ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3), axis = 0), -1, -4)
    return ret.astype(np.float32)/255.0

#%% Preparing test dataset (change for every new dataset)
# test result
a = np.int16([0, 1650 , 3240, 4530, 6240,7980,9180,10980, 11580,12030])



data_t =[]
data_lab = []


for i in range(a.shape[0]-1):
    for _ in range(20):
        sample = torch.from_numpy(get_data_test(randint(a[i], a[i+1])))
        sample = sample.cuda()
        with torch.no_grad():
            output = net(sample)
        data_t.append(output.cpu().numpy())
        data_lab.append([i,i,i])


data_t = np.array(data_t)
data_lab = np.int16(data_lab)
h,w,l = data_t.shape
data_t = data_t.reshape((h*w,l))
data_lab = data_lab.reshape((h*w,1))


#%% TSNE PLOT

from sklearn.manifold import TSNE

label3 = np.int16([ i for i in range(h*w)])%3

def tsne_plot(data = data_t, n_comp = 2, label = label3):
    X_embedded = TSNE(n_components=n_comp, verbose=1).fit_transform(data)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    if n_comp == 3:ax = fig.add_subplot(projection ='3d')
    
    # cdict = {0: 'red', 1: 'blue', 2: 'green'}
    
    markers = ['v', 'x', 'o', '.', '>', '<', '1', '2', '3']
    
    for i, g in enumerate(np.unique(label3)):
        ix = np.where(label == g)
        if n_comp==3:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], X_embedded[ix,2], marker = markers[i], label = g, alpha = 0.8)
        else:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], marker = markers[i], label = g, alpha = 0.8)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if n_comp==3:ax.set_zlabel('Z Label')
    
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    if n_comp == 3:ax = fig.add_subplot(projection ='3d')
    
    # cdict = {0: 'red', 1: 'blue', 2: 'green'}
    markers = ['v', 'x', 'o', '.', '>', '<', '1', '2', '3']
    
    for i, g in enumerate(np.unique(label)):
        ix = np.where(label == g)
        if n_comp==3:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], X_embedded[ix,2], marker = markers[i], label = g, alpha = 0.8)
        else:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], marker = markers[i], label = g, alpha = 0.8)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if n_comp==3:ax.set_zlabel('Z Label')
    
    
    
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

    

    
    
tsne_plot(data_t, 2, data_lab)
    
    