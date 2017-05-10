#
# Created on Apr 23, 2017
#
# @author: dpascualhe
#
# It classifies a given image and shows the activation between layers
# and their weights.
#

import os
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import visualize_util
from keras.models import load_model, Model

def plot_activations(nb_filters, activations):
    ''' Plots the activations obtained between layers '''
    for i in range(nb_filters):
        plt.subplot(math.ceil(nb_filters/7), 7, i + 1)
        plt.imshow(activations[0][:,:,i])
    plt.show()

if __name__ == "__main__":
    model_path = input("Net path: ")
    while not os.path.isfile(model_path):
        model_path = input("Enter a valid path: ")

    im_path = input("Image path: ")
    while not os.path.isfile(im_path):
        im_path = input("Enter a valid path: ")
    
    model = load_model(model_path)

    # We read the image.
    im = cv2.imread(im_path)
    cv2.namedWindow("Sample", cv2.WINDOW_NORMAL)
    cv2.imshow("Sample", im)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    # We adapt the image shape.
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if K.image_dim_ordering() == "th":
        im = im.reshape(1, 1, im.shape[0], im.shape[1])            
    else:      
        im = im.reshape(1, im.shape[0], im.shape[1], 1)
    model.predict(im)
    
    # First convolutional network activation and its plot
    first_conv = Model(input=model.inputs,
                       output=model.get_layer("convolution2d_1").output)
    first_conv_activation = first_conv.predict(im)
    plot_activations(first_conv_activation.shape[3], first_conv_activation)
    first_conv_weights = model.get_layer("convolution2d_1").get_weights()
    print("1st conv. layer weights shape: ", first_conv_weights[0].shape)
    visualize_util.plot(first_conv, "1st_conv.png", show_shapes=True)

    # Second convolutional network activation and its plot
    second_conv = Model(input=model.input,
                        output=model.get_layer("convolution2d_2").output)
    second_conv_activation = second_conv.predict(im)
    plot_activations(second_conv_activation.shape[3], second_conv_activation)
    second_conv_weights = model.get_layer("convolution2d_2").get_weights()
    print("2nd conv. layer weights shape: ", second_conv_weights[0].shape)
    visualize_util.plot(second_conv, "2nd_conv.png", show_shapes=True)
    
    