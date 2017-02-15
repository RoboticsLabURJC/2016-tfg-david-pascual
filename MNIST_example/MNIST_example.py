'''
Created on Feb 14, 2017

@author: dpascualhe

It trains and tests a Convolutional Neural Network (CNN) with the MNIST dataset

'''
import numpy
import cv2

# linear stack of neural network layers
from keras.models import Sequential
# core layers
from keras.layers import Dense, Dropout, Activation, Flatten
# CNN layers
from keras.layers import Convolution2D, MaxPooling2D
# utilities
from keras.utils import np_utils
# MNIST dataset
from keras.datasets import mnist
# backend related operations
from keras import backend

# seed for the computer pseudorandom number generator. It will allow us to 
# reproduce the results
numpy.random.seed(123)

if __name__ == '__main__':
    '''
    Declaring variables that we'll need
    '''
    # number of samples that is going to be propagated through the network
    batch_size = 128
    # number of classes
    nb_classes = 10
    # number of complete presentations of the training set to the network during
    # training
    nb_epoch = 12
    
    # image dimensions
    img_rows, img_cols = 28,28
    #number of convolutional filters and its kernel size
    nb_filters = 32
    kernel_size = (3,3)
    #size of pooling area
    pool_size = (2,2)
    
    '''
    Loading and shaping data in a way that it can work as input of our model
    '''
    # MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print ('Original input images data shape: ', x_train.shape)
    cv2.imshow('First sample',x_train[0])
    cv2.waitKey(5000)
    cv2.destroyWindow('First sample')
    
    if backend.image_dim_ordering() == 'th':
        # reshapes 3D data provided (nb_samples, width, height) into 4D
        # (nb_samples, nb_features, width, height) 
        x_train = x_train.reshape (x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape (x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1,img_rows,img_cols)
        print ('Input images data reshaped: ', (x_train.shape))
        print ('-------------------------------------------------------------------')
    else:
        # reshapes 3D data provided (nb_samples, width, height) into 4D
        # (nb_samples, nb_features, width, height) 
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        print ('Input images data reshaped: ', (x_train.shape))
        print ('-------------------------------------------------------------------')
 
    # converts the input data to 32bit floats and normalize it to [0,1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    print ('Original class lable data shape: ', (y_train.shape))
    print ('First 10 class lables: ', (y_train[:10]))
    # converts 1D array into a matrix containing 10 cols (one for each class)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    print ('Class lable data reshaped: ', (y_train.shape))
    print ('-------------------------------------------------------------------')

    # defines the model architecture, in this case, sequential
    model = Sequential()
    
    '''
    Adding layers to our model
    '''
    # convolutional layer
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid', input_shape=input_shape))
    # ReLU layer
    model.add(Activation('relu'))
    # convolutional layer
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    # ReLU layer
    model.add(Activation('relu'))
    # pooling layer
    model.add(MaxPooling2D(pool_size=pool_size))
    # dropout layer
    model.add(Dropout(0.25))
    
    # flattening the weights (making them 1D) to enter fully connected layer
    model.add(Flatten())
    # fully connected layer
    model.add(Dense(128, activation='relu'))
    # dropout layer to prevent overfitting
    model.add(Dropout(0.5))
    # output layer
    model.add(Dense(nb_classes, activation='softmax'))
    
    '''
    Compiling the model
    '''
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', 
                  metrics=['accuracy'])
    
    '''
    Training the model
    '''
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(x_test, y_test))
    
    '''
    Testing the model
    '''
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
