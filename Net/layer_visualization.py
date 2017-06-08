#
# Created on Apr 23, 2017
#
# @author: dpascualhe
#
# It classifies a given image and shows the activation between layers
# and their weights.
#

import math
import os

import cv2
from keras import backend as K
from keras.models import load_model, Model
from keras.utils import vis_utils

import matplotlib.pyplot as plt
import numpy as np

class Layer_Visualization:
    def __init__(self, model, im):
        self.model = model        
        # We adapt the image shape.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.im = im.reshape(1, im.shape[0], im.shape[1], 1)
    
    def norm(self, x):
        y = (x - np.amin(x)) / (np.amax(x) - np.amin(x)) * 255
        return y
    
    def plot_data(self, data_name, layer, nb_filters, data, i, j, depth,
                  interp="none", color="jet"):
        plt.figure(data_name + " - " + layer.get_config()["name"])
        size = math.sqrt(nb_filters * depth)
        plt.subplot(math.ceil(size), math.ceil(size), i + 1 + (j*depth))
        plot = plt.imshow(data, interpolation=interp, vmin=0, vmax=255)
        plot.set_cmap(color)
        
    def visualization(self, interp="none", color="jet"):
        pred = np.argmax(self.model.predict(self.im))
        print "\nDigit prediction: ", pred, "\n"
        
        for i, layer in enumerate(self.model.layers):
            if layer.get_config()["name"][:6] == "conv2d":
                # Getting weights
                shape = layer.get_weights()[0].shape
                weights = self.norm(layer.get_weights()[0].reshape(shape[2], shape[0],
                                                              shape[1], shape[3]))
                print "-------------", layer.get_config()["name"], "-------------"
                print "Filters:"
                print "    Width: ", weights.shape[1]
                print "    Height: ", weights.shape[2]
                print "    Depth: ", weights.shape[0]
                print "    Number of filters: ", weights.shape[3], "\n"
        
                # Getting activations
                truncated = Model(inputs=self.model.inputs, outputs=layer.output)
                activations = self.norm(truncated.predict(self.im))
                print "Activations:"
                print "    Width: ", activations.shape[1]
                print "    Height: ", activations.shape[2]
                print "    Depth: ", activations.shape[0]
                print "    Number of activation maps: ", activations.shape[3]
        
                nb_filters = weights.shape[3]
                for j in range(nb_filters):
                    # Weights
                    filter_depth = 1 # weights.shape[0]
                    for k in range(filter_depth):
                        filter = weights[k][:, :, j]
                        self.plot_data("Weights", layer, nb_filters, filter, k, j,
                                  filter_depth, interp, color)
            
                        # Weights gradient
                        sobelx = cv2.Sobel(filter, cv2.CV_32F, 1, 0, ksize=5)
                        sobely = cv2.Sobel(filter, cv2.CV_32F, 0, 1, ksize=5)
                        sobel = self.norm(abs(sobelx + sobely))
                        self.plot_data("Weights gradient", layer, nb_filters, sobel, k,
                                  j, filter_depth, interp, color)
                        
                    # Activation maps
                    activation_map = activations[0][:, :, j]
                    self.plot_data("Activation maps", layer, nb_filters, activation_map,
                              0, j, 1, interp, color)
                plt.show()
                
if __name__ == "__main__":  
    model = load_model("Nets/0-1_tuned/net_1conv.h5")
    im = cv2.imread("Datasets/Samples/0-6.png")
    lv = Layer_Visualization(model, im)
    lv.visualization()
