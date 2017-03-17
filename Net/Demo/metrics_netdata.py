#
# Created on Mar 12, 2017
#
# @author: dpascualhe
#

import sys

import cv2
import numpy as np
from keras import backend
from keras.utils import np_utils, visualize_util
from keras.preprocessing import image as imkeras

class NetData:

    def __init__(self, im_rows, im_cols, nb_classes,
                 x_train, y_train, x_test, y_test):
        ''' NetData class deals adapts and augment datasets. '''
        self.im_rows = im_rows
        self.im_cols = im_cols
        self.nb_classes = nb_classes
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.count = 0
    
    def adapt(self, verbose):
        ''' Adapts the dataset shape and format depending on Keras
        backend: TensorFlow or Theano.
        '''
        if backend.image_dim_ordering() == 'th':
            x_train = self.x_train.reshape(self.x_train.shape[0], 1,
                                           self.im_rows, self.im_cols)
            x_test = self.x_test.reshape(self.x_test.shape[0], 1, 
                                         self.im_rows, self.im_cols)
            input_shape = (1, self.im_rows, self.im_cols)
            
        else:
            x_train = self.x_train.reshape(self.x_train.shape[0],
                                           self.im_rows, self.im_cols, 1)
            x_test = self.x_test.reshape(self.x_test.shape[0], self.im_rows,
                                         self.im_cols, 1)
            input_shape = (self.im_rows, self.im_cols, 1)
         
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255 # Normalizes data: [0,255] -> [0,1]
        x_test /= 255
            
        # Converts class vector to class matrix
        y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
        
        if verbose == "y":
            print('Original input images data shape: ', self.x_train.shape)
            print('Input images data reshaped: ', (x_train.shape))
            print('----------------------------------------------------------')
            print('Input images type: ', self.x_train.dtype)
            print('New input images type: ', x_train.dtype)
            print('----------------------------------------------------------')
            print('Original class label data shape: ', (self.y_train.shape))
            print('Class label data reshaped: ', (y_train.shape))
            print('----------------------------------------------------------')
        
        return (x_train, y_train), (x_test, y_test), input_shape

    def sobelEdges(self, sample):
        ''' Apply a sobel filtering in x and y directions in order to
        detect edges. It's used right before data enters the net.
        '''
        im_sobel_x = cv2.Sobel(sample, cv2.CV_32F, 1, 0, ksize=5)
        im_sobel_y = cv2.Sobel(sample, cv2.CV_32F, 0, 1, ksize=5)
        im_edges = cv2.add(abs(im_sobel_x), abs(im_sobel_y))
        im_edges = cv2.normalize(im_edges, None, 0, 1, cv2.NORM_MINMAX)
        im_edges = im_edges[ : , : ,np.newaxis]
        
        return im_edges

    def augmentation(self, x, y, batch_size, mode, verbose):
        ''' Creates a generator that augments data in real time. It can
        apply only a Sobel filtering or a stack of processes that
        randomize the data.
        '''      
        if mode == "full":
            datagen = imkeras.ImageDataGenerator(
                zoom_range=0.2, rotation_range=20, width_shift_range=0.2, 
                height_shift_range=0.2, fill_mode='constant', cval=0,
                preprocessing_function=self.sobelEdges)
        elif mode == "edges":
            datagen = imkeras.ImageDataGenerator(
                preprocessing_function=self.sobelEdges)
  
        generator = datagen.flow(x, y, batch_size=batch_size)
        
        if verbose == "y":
            i = 0
            j = 0
            classes_count = [0,0,0,0,0,0,0,0,0,0]
            for x_batch, y_batch in generator:
                if i == 0:
                    for sample in x_batch:
                        cv2.imshow("Augmented sample", sample)
                        cv2.waitKey(500)
                        j += 1
                        if j > 9:
                            cv2.destroyWindow("Augmented sample")
                            break
                if self.count == 0:
                    for classes in y_batch:
                        if np.where(classes == 1)[0] == [0]:
                            classes_count[0] += 1
                        elif np.where(classes == 1)[0] == [1]:
                            classes_count[1] += 1
                        elif np.where(classes == 1)[0] == [2]:
                            classes_count[2] += 1
                        elif np.where(classes == 1)[0] == [3]:
                            classes_count[3] += 1
                        elif np.where(classes == 1)[0] == [4]:
                            classes_count[4] += 1
                        elif np.where(classes == 1)[0] == [5]:
                            classes_count[5] += 1
                        elif np.where(classes == 1)[0] == [6]:
                            classes_count[6] += 1
                        elif np.where(classes == 1)[0] == [7]:
                            classes_count[7] += 1
                        elif np.where(classes == 1)[0] == [8]:
                            classes_count[8] += 1
                        elif np.where(classes == 1)[0] == [9]:
                            classes_count[9] += 1
                    i += 1
                    if i >= 3000:
                        print("Class distribution: ", classes_count)
                        break
                else:
                    break  
        
        self.count += 1
        
        return generator

    