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
from keras.models import load_model, Model

def plot_activations(nb_filters, activations):
    ''' Plots the activations obtained between layers '''
    nb_filters = activations.shape[3]
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
    st_conv = Model(input=model.inputs,
                    output=model.get_layer("convolution2d_1").output)
    st_output = st_conv.predict(im)
    plot_activations(st_output.shape[3], st_output)    

    # Second convolutional network activation and its plot
    nd_conv = Model(input=model.input,
                    output=model.get_layer("convolution2d_2").output)
    nd_output = nd_conv.predict(im)
    plot_activations(nd_output.shape[3], nd_output)
    
    