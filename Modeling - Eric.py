import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datetime
import torch

from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json, Sequential
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from tqdm import tqdm
#%matplotlib inline

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./227,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
#Divide the pixels of images by 255 so that the pixel values of images comes in the range [0,1]. This step helps in optimizing the performance of our model.
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '/Users/ericross/School/Queens_MMAI/MMAI/MMAI_894/Team_Project/Data/Train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '/Users/ericross/School/Queens_MMAI/MMAI/MMAI_894/Team_Project/Data/Test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

print('hey how are you 1 ')