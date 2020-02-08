# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:47:49 2020

@author: Nicky
"""

### preprocessing
## method 1: for Keras (not recommended as prof asked for not using high level API)

from keras.preprocessing.image import ImageDataGenerator
import tensorflow
import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import Callback




train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    'C:/Users/Nicky/Desktop/Old Com/Queens Master/MMAI894/Project/dataset/',
    target_size=(128, 128),
    batch_size=64,
    class_mode='binary',
    subset='training') # set as training data
# train_generator size = (1-validation_split)*train_datagen

validation_generator = train_datagen.flow_from_directory(
    'C:/Users/Nicky/Desktop/Old Com/Queens Master/MMAI894/Project/dataset/', # same directory as training data
    target_size=(128, 128),
    batch_size=64,
    class_mode='binary',
    subset='validation') # set as validation data
# validation_generator size = validation_split*train_datagen


## method 2: general (for Tensorflow and Pytorch)
import tensorflow
import keras

import os
##pip install opencv-python
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

data_directory = "C:/Users/Nicky/Desktop/Old Com/Queens Master/MMAI894/Project/dataset/" #insert the directory 
img_size = 128
categories = ["Positive", "Negative"]
training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(data_directory, category)
        class_num = categories.index(category)
        
        # read and resize the images and append to training_data a list with the image itself and its class number
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_num])

create_training_data()
random.shuffle(training_data)
X_data = []
y = []
for features, label in training_data:
    X_data.append(features)
    y.append(label)
X = np.array(X_data).reshape(len(X_data), img_size, img_size, 1)



X = X / 255

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

# ensure data balance in training

import collections
c = collections.Counter(y_train)
c

from tqdm import tqdm
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import torch


final_train_data = []
final_target_train = []
for i in tqdm(range(X_train.shape[0])):
    final_train_data.append(X_train[i])
    final_train_data.append(rotate(X_train[i], angle=45, mode = 'wrap'))
    final_train_data.append(np.fliplr(X_train[i]))
    final_train_data.append(np.flipud(X_train[i]))
    final_train_data.append(random_noise(X_train[i],var=0.2**2))
    for j in range(5):
        final_target_train.append(y_train[i])
        
len(final_target_train), len(final_train_data)
##final_train = np.array(random.sample(final_train_data,28000))
##final_target_train = np.array(final_target_train)

##final_train, final_target_train = zip(*random.sample(list(zip(final_train_data, final_target_train)), 28000))

## stratified sample due to memory error
final_train, xxx, final_target_train, yyy = train_test_split(final_train_data, final_target_train, test_size=0.8, random_state=42, stratify = final_target_train)

###======final_train and final_target_train can be used in tensorflow
###======the below are for pytorch

# converting training images into torch format
final_train = np.array(final_train).reshape(28000, 1, 128, 128)
final_train  = torch.from_numpy(final_train)
final_train = final_train.float()

# converting the target into torch format
final_target_train = np.array(final_target_train).astype(int)
final_target_train = torch.tensor(final_target_train, dtype=torch.long)

X_test_sample, xxxx, y_test_sample, yyyy = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify = y_test)
X_test_sample, xxxx, y_test_sample, yyyy = train_test_split(X_test_sample, y_test_sample, test_size=0.8, random_state=42, stratify = y_test_sample)

# converting validation images into torch format
val_x = np.array(X_test).reshape(12000, 1, 128, 128)
val_x  = torch.from_numpy(val_x)
val_x = val_x.float()

# converting the target into torch format
val_y = np.array(y_test).astype(int)
val_y = torch.tensor(val_y, dtype=torch.long)




## method 3: for Pytorch only (notice that the dataset are not 100% balanced in each batch)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = 'C:/Users/Nicky/Desktop/Old Com/Queens Master/MMAI894/Project/dataset/'
def train_test_split(datadir, valid_size = .2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
        transforms.Resize(size=128),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transform)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader
trainloader, testloader = train_test_split(data_dir)
print(trainloader.dataset.classes)
print(trainloader.dataset)
# very batch should have 40000*(1-valid_size)/batch_size
len(trainloader)
print(testloader.dataset)
# very batch should have 40000*valid_size/batch_size
len(testloader)

# making sure each batch is roughly balanced
for i, (x, y) in enumerate(trainloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
    
for i, (x, y) in enumerate(testloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))