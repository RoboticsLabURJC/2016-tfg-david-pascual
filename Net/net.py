#
# Created on Mar 12, 2017
#
# @author: dpascualhe
#
# It trains and tests a convolutional neural network with an augmented 
# MNIST dataset.
#

import os
import sys
import math
import datetime

import numpy as np
from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, GaussianNoise
from keras.models import Sequential, load_model
from keras.utils import visualize_util

from netdata import NetData

# Seed for the computer pseudorandom number generator.
np.random.seed(123)

if __name__ == '__main__':    
    batch_size = 128
    nb_classes = 10
    nb_epoch = 12
    nb_train_samples = 384000
    nb_val_samples = 96000
    nb_test_samples = 10000
        
    im_rows, im_cols = 28, 28
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    std_gauss = math.sqrt(0.02)
    
    while True:
        training = input("Do you want to train the model again?(y/n)")
        if training == "y" or training == "n":
            break
    while True:
        verbose = input("Do you want the program to be specially "
                        + "verbose?(y/n)")
        if verbose == "y" or verbose == "n":
            break
    print("\n\n")

    # We load and reshape data in a way that it can work as input of
    # our model.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    
    data = NetData(im_rows, im_cols, nb_classes, 
                   x_train, y_train, x_test, y_test)    
    (x_train, y_train), (x_test, y_test), input_shape = data.adapt(verbose)
    
    if training == "y":
        aug_train = data.augmentation(x_train, y_train, batch_size,"full",
                                      verbose)
        edges_val = data.augmentation(x_train, y_train, batch_size,"edges",
                                      verbose)  

        # We add layers to our model (first layer inserts gaussian
        # noise to make the net more robust).
        model = Sequential()
        model.add(GaussianNoise(std_gauss, input_shape = input_shape))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid', activation='relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))
          
        model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                      metrics=['accuracy'])
            
        # We train the model.
        history = model.fit_generator(aug_train, 
                                      samples_per_epoch=nb_train_samples,
                                      nb_epoch=nb_epoch,
                                      validation_data=edges_val,
                                      nb_val_samples=nb_val_samples, verbose=1)
            
        model.save('net.h5')
    
    model = load_model('net.h5')
    edges_test = data.augmentation(x_test, y_test, batch_size, "edges",
                                   verbose)
    
    # We test the model and plot a diagram.
    visualize_util.plot(model, 'net.png', show_shapes=True)
    score = model.evaluate_generator(edges_test, nb_test_samples)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # We log the results.
    if os.path.isfile("results.txt"):
        file = open("results.txt", "a")
    else: 
        file = open("results.txt", "w")
    file.write("Date: " + str(datetime.datetime.now()) + "\n\n")
    if training == "y":
        file.write("Training the network...\n")
        file.write(str(history.history) + "\n\n")
    else:
        file.write("Just testing the net...\n")
    file.write('Test score:' + str(score[0]) + "\n")
    file.write('Test accuracy:' + str(score[1]) + "\n")  
    file.write('---------------------------------------------------------\n\n')
    file.close()

    