# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:36:49 2020

@author: Nicky
"""
## training set are not 100% balanced, however more data augmentation is integrated
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_dir = ".../MMAI894/Project/dataset/"
def train_test_split(datadir, valid_size = .2):
    ## data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=64),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
        transforms.Resize(size=128),
        transforms.CenterCrop(size=64),
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
# every batch should have 40000*(1-valid_size)/batch_size
len(trainloader)
print(testloader.dataset)
# every batch should have 40000*valid_size/batch_size
len(testloader)

# making sure each batch is roughly balanced
for i, (x, y) in enumerate(trainloader):
    print("batch index {}, 0/1: {}/{}".format(
     i, (y == 0).sum(), (y == 1).sum()))
    
    
for i, (x, y) in enumerate(testloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
    
torch.manual_seed(0)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
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
            Linear(2048, 512),
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
    


for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

prediction = []
target = []

for i, data in enumerate(trainloader, 0):
    
    inputs, labels = data

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(inputs)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(labels)
    
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


for i, data in enumerate(testloader, 0):
    
    inputs, labels = data

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(inputs)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_test.append(predictions)
    target_test.append(labels)
    
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

# the below is to show the network in pytorch
writer = SummaryWriter('runs/')
# get some random training images
dataiter = iter(trainloader)
dataiter
images, labels = dataiter.next()
images

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# write to tensorboard
writer.add_image('example', img_grid)

writer.add_graph(model, images)
writer.close()


## Transferrability
## Only a portion is tested due to time constraint and memory error (following the same method to split training set (not trained) and test set)

data_directory = ".../MMAI894/Project/pavementtest/P/" #insert the directory 
# data_directory = ".../MMAI894/Project/bridgetest/P/"
# data_directory = ".../MMAI894/Project/walltest/P/"

def train_test_split(datadir, valid_size = 1):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=64),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
        transforms.Resize(size=128),
        transforms.CenterCrop(size=64),
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
trainloader, testloader = train_test_split(data_directory)
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
    
batch_size = 32

prediction_test = []
target_test = []


for i, data in enumerate(testloader, 0):
    
    inputs, labels = data

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(inputs)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_test.append(predictions)
    target_test.append(labels)
    
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