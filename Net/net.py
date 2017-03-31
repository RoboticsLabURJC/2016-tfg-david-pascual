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
from timeit import default_timer as timer

import numpy as np
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.utils import visualize_util

from DataManager.netdata import NetData
from CustomMetrics.learningcurve import LearningCurve
from CustomMetrics.custommetrics import CustomMetrics

start_full = timer()

# Seed for the computer pseudorandom number generator.
np.random.seed(123)

if __name__ == "__main__":  
    nb_epoch = 12
    batch_size = 128
    nb_classes = 10
        
    im_rows, im_cols = 28, 28
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    verbose = 0
    training = 0
    while training != "y" and verbose != "n":
        training = input("Do you want to train the model?(y/n)")
    while verbose != "y" and verbose != "n":
        verbose = input("Do you want the program to be specially "
                        + "verbose?(y/n)")
    print("\n\n")

    data = NetData(im_rows, im_cols, nb_classes)
    
    if training == "y":
        train_ds = input("Train dataset path: ")
        while not os.path.isfile(train_ds):
            train_ds = input("Enter a valid path: ")
        val_ds = input("Validation dataset path: ")
        while not os.path.isfile(val_ds):
            val_ds = input("Enter a valid path: ")
            
    test_ds = input("Test dataset path: ")
    while not os.path.isfile(test_ds):
        test_ds = input("Enter a valid path: ")
        
    if training == "y":    
        # We load and reshape data in a way that it can work as input of
        # our model.
        start_data = timer()
        (X_train, Y_train) = data.load(train_ds)
        (x_train, y_train), input_shape = data.adapt(X_train, Y_train, verbose)
        gen = data.augmentation(x_train, y_train, batch_size, "full", verbose)

        
        (X_val, Y_val) = data.load(val_ds)
        (x_val, y_val), input_shape = data.adapt(X_val, Y_val, verbose)
        end_data = timer()
        
        # We add layers to our model.
        start_train = timer()
        model = Sequential()
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode="valid", activation="relu",
                                input_shape=input_shape))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                activation="relu"))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation="softmax"))
          
        model.compile(loss="categorical_crossentropy", optimizer="adadelta",
                      metrics=["accuracy"])
            
        # We train the model and save data to plot a learning curve.
        learning_curve = LearningCurve()
        validation = model.fit(x_train, y_train, batch_size=batch_size, 
                               nb_epoch=nb_epoch, callbacks=[learning_curve],
                               validation_data=(x_val, y_val))
            
        model.save("net.h5")
        visualize_util.plot(model, "net.png", show_shapes=True)
        end_train = timer()
    
    if training == "n":
        net = input("Net path: ")
        while not os.path.isfile(net):
            net = input("Enter a valid path: ")
        model = load_model(net)

    (X_test, Y_test) = data.load(test_ds)
    (x_test, y_test), input_shape = data.adapt(X_test, Y_test, verbose)
    
    # We test the model.
    start_test = timer()
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    end_test = timer()
    
    # We log the results.
    if training == "n":
        metrics = CustomMetrics(model, x_test, Y_test, batch_size)
    else:
        metrics = CustomMetrics(model, x_test, Y_test, batch_size,
                                learning_curve, validation, training)
    
    metrics_dict = metrics.dictionary()
    metrics.log(metrics_dict)
    end_full = timer()
    
    print("Full time: " + str(end_full-start_full))
    print("Data time: " + str(end_data-start_data))
    print("Train time: " + str(end_train-start_train))
    print("Test time: " + str(end_test-start_test))
    
