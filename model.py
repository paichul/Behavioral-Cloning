# Author: Paichun Lin
# MIT License

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tools import preprocess
from tools import get_samples

#loading training image/steering angle data using the driving log csv files
X1, y1 = get_samples('./data/driving_log_track2_clockwise1.csv')
X2, y2 = get_samples('./data/driving_log_track2_clockwise2.csv')
X3, y3 = get_samples('./data/driving_log_track2_counterclockwise1.csv')
X4, y4 = get_samples('./data/driving_log_track2_counterclockwise2.csv')
X = np.concatenate((X1, X2, X3, X4), axis=0)
y = np.concatenate((y1, y2, y3, y4), axis=0)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

from sklearn.utils import shuffle

#randomly choose left, right or center images and load images
def load_image(X_sample, angle, is_training):
  if is_training and np.random.random() > 0.8:
    c = np.random.randint(1, 3) # randomly choose left or right image
    name = './data/IMG/' + X_sample[c].strip().split('/')[-1]
    if c == 1:
      angle = float(angle) + 0.2
    elif c == 2:
      angle = float(angle) - 0.2

    image = cv2.imread(name)
    image = preprocess(image)
    return image, angle
  else: #chose center images most of the time
    name = './data/IMG/' + X_sample[0].strip().split('/')[-1]
    angle = float(angle)
    image = cv2.imread(name)
    image = preprocess(image)
    return image, angle

#load images and do data augmentation
def load_data(X_sample, angle, is_training):
  image, angle = load_image(X_sample, angle, is_training)

  return image, angle

#generator that is used to generate batches for training and testing/validation
def generator(X, y, batch_size, is_training):
  assert(len(X) == len(y))
  num_samples = len(X)
  while 1:
    shuffle(X, y)
    for offset in range(0, num_samples, batch_size):
      X_samples = X[offset:offset+batch_size]
      y_samples = y[offset:offset+batch_size]
 
      images = []
      angles = []
      for X_sample, y_sample in zip(X_samples, y_samples):
        image, angle = load_data(X_sample, y_sample, is_training) 
        images.append(image)
        angles.append(angle)
        
      X_data = np.array(images)
      y_data = np.array(angles)
      yield shuffle(X_data, y_data)

train_generator = generator(train_X, train_y, batch_size=32, is_training=True)
validation_generator = generator(test_X, test_y, batch_size=32, is_training=False)

from keras.layers import Dense, Input, Conv2D, merge, Activation
from keras.layers import BatchNormalization, Dropout, Flatten
from keras.models import Model

img = Input(shape=(90, 260, 3))
conv1 = Conv2D(24, (5, 5), strides=(2, 2), name='conv1')(img)
conv1 = BatchNormalization(axis=-1)(conv1)
conv1 = Activation('relu', name='conv1_a')(conv1)
conv2 = Conv2D(36, (5, 5), strides=(2, 2), name='conv2')(conv1)
conv2 = BatchNormalization(axis=-1)(conv2)
conv2 = Activation('relu', name='conv2_a')(conv2)
conv3 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', name='conv3')(conv2)
conv4 = Conv2D(64, (3, 3), activation='relu', name='conv4')(conv3)
conv5 = Conv2D(64, (3, 3), activation='relu', name='conv5')(conv4)
conv5 = Dropout(0.5)(conv5)
conv5 = Flatten()(conv5)
fc1 = Dense(1164, activation='relu')(conv5)
fc2 = Dense(100, activation='relu')(fc1)
fc3 = Dense(50, activation='relu')(fc2)
fc4 = Dense(10, activation='relu')(fc3)
concat = merge([fc4, conv5], mode='concat');
steering_angle = Dense(1, name='output')(concat)

model = Model(inputs=img, outputs=steering_angle)
model.compile(loss='mse', optimizer="adam")
model.fit_generator(train_generator, samples_per_epoch=len(train_X), validation_data=validation_generator, nb_val_samples=len(test_X), nb_epoch=1)

model.save('model.h5')
