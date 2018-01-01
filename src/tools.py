# Author: Paichun Lin
# MIT License

import cv2
import pandas as pd

#get samples from the driving log
def get_samples(filename):
  df = pd.read_csv(filename)
  X = df[['center','left','right']].values
  y = df['steering'].values
  return X, y

#data preprocessing
def preprocess(image):
  yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV) #convert from BGR to YUV
  hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  yuv[:,:,2] = hsv[:,:,2] #replace yuv's v channel with hsv's v channel
  image = image[50:-20, 30:-30,:] #cropping
  image = image/127.5 - 1.0 #normalization
  return image
