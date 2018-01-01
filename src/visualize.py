# Author: Paichun Lin
# MIT License

import tensorflow as tf
from keras.models import load_model
from keras import __version__ as keras_version
from keras.layers.core import Lambda
from keras.models import Sequential
import keras.backend as K

import argparse
import cv2
import scipy.misc
import numpy as np
import h5py
import csv

from tools import preprocess
from tools import get_samples

K.set_learning_phase(0) #set learning phase

#Mean Squared Error metric
def mse_loss(x, steering_angle_gt):
    return tf.reduce_mean(tf.square(tf.subtract(x, steering_angle_gt)))

def mse_loss_output_shape(input_shape):
    return input_shape

# utility function to normalize a tensor by its L2 norm
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def visualize_attention(input_model, image, steering_angle_gt, layer_name):
    model = Sequential()
    model.add(input_model)

    target_layer = lambda x: mse_loss(x, steering_angle_gt)
    model.add(Lambda(target_layer,
                     output_shape = mse_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output = input_model.get_layer(layer_name).output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    image_array = preprocess(image)
    output, grads_val = gradient_function([image_array[None, :, :, :]])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    vismap = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        vismap += w * output[:, :, i]

    image = image[50:-20, 30:-30,:]
    vismap = cv2.resize(vismap, tuple(image.shape[0:2][::-1]))
    vismap = np.exp(vismap) - 1
    heatmap = vismap / np.max(vismap)

    vismap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    vismap = np.float32(vismap) + np.float32(image)
    vismap = 255 * vismap / np.max(vismap)
    vismap = cv2.resize(vismap, (360, 360))
    return np.uint8(vismap), heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize layers')
    parser.add_argument('--model', '-m', action='store', dest='model',
                        default='model.h5', help='Input model name')
    parser.add_argument('--input-file', '-log', action='store', dest='input_file',
                        default='driving_log.csv', help='test log file')
    parser.add_argument('--data-dir', '--data', action='store', dest='data_dir',
                        default='./data')
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    X, y = get_samples(args.data_dir + '/' + args.input_file)
    sess = tf.Session()
    
    for x, steering_angle_gt in zip(X, y):
        filename = './data/IMG/' + x[0].strip().split('/')[-1]
        image = cv2.imread(filename)
        vismap, heatmap = visualize_attention(model, image, steering_angle_gt, 'conv1')
        #cv2.imshow("heatmap", heatmap)
        #cv2.waitKey(1)
        
        cv2.imshow("attention map", vismap)
        cv2.waitKey(1)
