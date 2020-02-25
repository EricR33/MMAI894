# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:37:32 2019

@author: Nicky
"""

import tensorflow
import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import os
##pip install opencv-python
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

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

## save the dataset

X = X / 255

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)
 
import collections
c = collections.Counter(y_train)
c
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
input_size = (128, 128)
classifier.add(Conv2D(32, (2, 2), input_shape=(*input_size, 1), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2))) # 2x2 is optimal
##classifier.add(Dropout(.2))  
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (2, 2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
##classifier.add(Dropout(.2)) 
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (2, 2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
##classifier.add(Dropout(.2)) 
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Part 2 - Fitting the CNN to the images
batch_size = 28
##train_datagen = ImageDataGenerator(rescale=1. / 255,
##                                   shear_range=0.2,
##                                   zoom_range=0.2,
##                                   horizontal_flip=True)
 
##test_datagen = ImageDataGenerator(rescale=1. / 255)
 
##training_set = train_datagen.flow_from_directory(training_set_path,
##                                                 target_size=input_size,
##                                                 batch_size=batch_size,
##                                                 class_mode='binary')
 
##test_set = test_datagen.flow_from_directory(test_set_path,
##                                            target_size=input_size,
##                                            batch_size=batch_size,
##                                            class_mode='binary')
 
# Create a loss history
#history = LossHistory()
 
classifier.fit(X_train, y_train, batch_size = batch_size, epochs = 5, verbose=1, validation_split = .2)
 

score = classifier.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 
# Save model


 
