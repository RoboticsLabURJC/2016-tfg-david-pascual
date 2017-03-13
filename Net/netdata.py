'''
Created on Mar 12, 2017

@author: dpascualhe

NetData class deals adapts and augment datasets

'''
from keras import backend
from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.utils import np_utils  # , visualize_util
import cv2
import numpy
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.preprocessing.image import img_to_array, load_img
import h5py

class NetData:
    
    def __init__(self, batch_size, nb_classes, img_rows, img_cols, x_train, 
                 y_train, x_test, y_test):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def adapt(self):        
        if backend.image_dim_ordering() == 'th':
            # reshapes 3D data provided (nb_samples, width, height) into 4D
            # (nb_samples, nb_features, width, height) 
            x_train = self.x_train.reshape(self.x_train.shape[0], 1,
                                           self.img_rows, self.img_cols)
            x_test = self.x_test.reshape(self.x_test.shape[0], 1, 
                                         self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
            
        else:
            # reshapes 3D data provided (nb_samples, width, height) into 4D
            # (nb_samples, nb_features, width, height) 
            x_train = self.x_train.reshape(self.x_train.shape[0],
                                           self.img_rows, self.img_cols, 1)
            x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows,
                                         self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)
         
        # converts the input data to 32bit floats and normalize it to [0,1]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
            
        # converts class vector (integers from 0 to nb_classes) to class matrix
        # (nb_samples, nb_classes)
        y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
        
        return (x_train, y_train), (x_test, y_test), input_shape

    def augmentation(self, x_train, y_train, control):            
        datagen_train = ImageDataGenerator(zoom_range=0.2,
                                           rotation_range=20, 
                                           width_shift_range=0.2,
                                           height_shift_range=0.2, 
                                           fill_mode='constant', cval=0,
                                           preprocessing_function = 5)  
              
        train_generator = datagen_train.flow(x_train, y_train, batch_size=128)    
        
        if control:
            i = 0
            j = 0
            classes_count = [0,0,0,0,0,0,0,0,0,0]
            for x_batch, y_batch in train_generator:
                if i == 0:
                    for sample in x_batch:
                        cv2.imshow("Augmented sample", sample)
                        cv2.waitKey(500)
                        j += 1
                        if j > 9:
                            cv2.destroyWindow("Augmented sample")
                            break
                for classes in y_batch:
                    if numpy.where(classes == 1)[0] == [0]:
                        classes_count[0] += 1
                    elif numpy.where(classes == 1)[0] == [1]:
                        classes_count[1] += 1
                    elif numpy.where(classes == 1)[0] == [2]:
                        classes_count[2] += 1
                    elif numpy.where(classes == 1)[0] == [3]:
                        classes_count[3] += 1
                    elif numpy.where(classes == 1)[0] == [4]:
                        classes_count[4] += 1
                    elif numpy.where(classes == 1)[0] == [5]:
                        classes_count[5] += 1
                    elif numpy.where(classes == 1)[0] == [6]:
                        classes_count[6] += 1
                    elif numpy.where(classes == 1)[0] == [7]:
                        classes_count[7] += 1
                    elif numpy.where(classes == 1)[0] == [8]:
                        classes_count[8] += 1
                    elif numpy.where(classes == 1)[0] == [9]:
                        classes_count[9] += 1
                i += 1
                if i >= 3000:
                    print(classes_count)
                    break
                
        datagen_val = ImageDataGenerator(preprocessing_function = 5)              
        val_generator = datagen_val.flow(x_train, y_train, batch_size=128)                
            
        return train_generator, val_generator
    