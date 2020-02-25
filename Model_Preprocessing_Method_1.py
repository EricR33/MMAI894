# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:15:05 2020

@author: Nicky
"""

## method 1: general with balanced dataset
import tensorflow
import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
import collections
from tqdm import tqdm
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import torch
import torchvision
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.utils.tensorboard import SummaryWriter

data_directory = ".../MMAI894/Project/dataset/" #insert the directory 
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
c = collections.Counter(y_train)
c


# data augmentation
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


## stratified sample due to CPU memory error - this way can ensure balanced dataset in the training set
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

#X_test_sample, xxxx, y_test_sample, yyyy = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify = y_test)
#X_test_sample, xxxx, y_test_sample, yyyy = train_test_split(X_test_sample, y_test_sample, test_size=0.8, random_state=42, stratify = y_test_sample)

# converting validation images into torch format
val_x = np.array(X_test).reshape(12000, 1, 128, 128)
val_x  = torch.from_numpy(val_x)
val_x = val_x.float()

# converting the target into torch format
val_y = np.array(y_test).astype(int)
val_y = torch.tensor(val_y, dtype=torch.long)

torch.manual_seed(0)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
            # Defining another 2D convolution layer
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
            # Defining another 2D convolution layer
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
            # Defining another 2D convolution layer 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
        )

        self.linear_layers = Sequential(
            Linear(8192, 512),
            ReLU(inplace=True),
            Dropout(),
            Linear(512, 256),
            ReLU(inplace=True),
            Dropout(),
            Linear(256,10),
            ReLU(inplace=True),
            Dropout(),
            Linear(10,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)
    
# batch size of the model
batch_size = 64

# number of epochs to train the model
n_epochs = 2

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
        
    permutation = torch.randperm(final_train.size()[0])

    training_loss = []
    for i in tqdm(range(0,final_train.size()[0], batch_size)):

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = final_train[indices], final_target_train[indices]
        
        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs,batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
    
# prediction for training set
prediction = []
target = []
permutation = torch.randperm(final_train.size()[0])
for i in tqdm(range(0,final_train.size()[0], batch_size)):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = final_train[indices], final_target_train[indices]

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)
    
# training accuracy
accuracy = []
precision = []
recall = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i].cpu(),prediction[i]))
    precision.append(precision_score(target[i].cpu(),prediction[i]))
    recall.append(recall_score(target[i].cpu(),prediction[i]))
    
print('training accuracy: \t', np.average(accuracy))
print('training precision: \t', np.average(precision))
print('training recall: \t', np.average(recall))

# checking the performance on validation set
# checking it in batches due to CPU memory error
batch_size = 32

prediction_test = []
target_test = []
permutation_test = torch.randperm(val_x.size()[0])
for i in tqdm(range(0,val_x.size()[0], batch_size)):
    indices = permutation_test[i:i+batch_size]
    batch_x, batch_y = val_x[indices], val_y[indices]

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_test.append(predictions)
    target_test.append(batch_y)
    
accuracy_test = []
precision_test = []
recall_test = []
for i in range(len(prediction_test)):
    accuracy_test.append(accuracy_score(target_test[i].cpu(),prediction_test[i]))
    precision_test.append(precision_score(target_test[i].cpu(),prediction_test[i]))
    recall_test.append(recall_score(target_test[i].cpu(),prediction_test[i]))
    
print('testing accuracy: \t', np.average(accuracy_test))
print('testing precision: \t', np.average(precision_test))
print('testing recall: \t', np.average(recall_test))



## Transferrability
## Only a portion is tested due to time constraint and memory error (following the same method to split training set (not trained) and test set)
# clear data to save memory
X = []
X_data = []
X_train =[]
X_test =[]
final_train_data=[]

data_directory = ".../MMAI894/Project/pavementtest/P/" #insert the directory 
# data_directory = ".../MMAI894/Project/bridgetest/P/"
# data_directory = ".../MMAI894/Project/walltest/P/"
img_size = 128
categories = ["Positive", "Negative"]
test_data = []

def create_test_data():
    for category in categories:
        path = os.path.join(data_directory, category)
        class_num = categories.index(category)
        
        # read and resize the images and append to training_data a list with the image itself and its class number
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            test_data.append([new_array, class_num])

create_test_data()

X_deck = []
y_deck = []
for features, label in test_data:
    X_deck.append(features)
    y_deck.append(label)
X_deck = np.array(X_deck).reshape(len(X_deck), img_size, img_size, 1)
X_deck = X_deck / 255

# converting validation images into torch format
val_deck = np.array(X_deck).reshape(len(X_deck), 1, 128, 128)
val_deck  = torch.from_numpy(val_deck)
val_deck = val_deck.float()



# converting the target into torch format
val_y_deck = np.array(y_deck).astype(int)
val_y_deck = torch.tensor(val_y_deck, dtype=torch.long)




#method : testing it in batches due to CPU memory error

batch_size = 32

prediction_test = []
target_test = []
permutation_test = torch.randperm(val_deck.size()[0])
for i in tqdm(range(0,val_deck.size()[0], batch_size)):
    indices = permutation_test[i:i+batch_size]
    batch_x, batch_y = val_deck[indices], val_y_deck[indices]

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_test.append(predictions)
    target_test.append(batch_y)
    

    
accuracy_test = []
precision_test = []
recall_test = []
for i in range(len(prediction_test)):
    accuracy_test.append(accuracy_score(target_test[i].cpu(),prediction_test[i]))
    precision_test.append(precision_score(target_test[i].cpu(),prediction_test[i]))
    recall_test.append(recall_score(target_test[i].cpu(),prediction_test[i]))
    
print('testing accuracy: \t', np.average(accuracy_test))
print('testing precision: \t', np.average(precision_test))
print('testing recall: \t', np.average(recall_test))