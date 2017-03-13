'''
Created on Mar 12, 2017

@author: dpascualhe

'''

from keras import backend
from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, GaussianNoise
from keras.models import Sequential, load_model
from keras.utils import np_utils  # , visualize_util
import cv2
import numpy
import sys
from netdata import NetData
import h5py

# seed for the computer pseudorandom number generator. It will allow us to 
# reproduce the results
numpy.random.seed(123)

if __name__ == '__main__':
    
    '''
    Declaring variables that we'll need
    '''
    batch_size = 128
    nb_classes = 10
    nb_epoch = 12
    nb_train_samples = 384000
    nb_val_samples = 96000
        
    img_rows, img_cols = 28, 28
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    '''
    Loading and shaping data in a way that it can work as input of our model
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    data = NetData(batch_size, nb_classes, img_rows, img_cols, x_train, 
                   y_train, x_test, y_test)
    
    (x_train, y_train), (x_test, y_test), input_shape = data.adapt()
    
    control = 0
    aug_train, aug_val = data.augmentation(x_train, y_train, control)

    '''
    Adding layers to our model
    '''
    model = Sequential()
    
    percent_noise = 0.1
    noise = (1.0/255) * percent_noise
    model.add(GaussianNoise(0.1, input_shape = input_shape))
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
      
    '''
    Compiling the model
    '''
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
        
    '''
    Training the model
    '''
    model.fit_generator(aug_train, samples_per_epoch=nb_train_samples, 
                        nb_epoch=nb_epoch, verbose=1, validation_data=aug_val,
                        nb_val_samples=nb_val_samples)
        
    '''
    Saving the model architecture and weights
    '''
    model.save('net.h5')
    
    '''
    Testing the model
    '''
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    

    