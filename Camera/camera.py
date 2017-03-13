'''
Created on Mar 7, 2017

@author: dpascualhe

Camera class.

Based on @nuriaoyaga code:
https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/master/camera/camera.py

And @Javii91 code:
https://github.com/Javii91/Domotic/blob/master/Others/cameraview.py

'''

import sys, traceback, threading, random
import numpy as np
from PIL import Image
from jderobot import CameraPrx
import easyiceconfig as EasyIce
import cv2
from keras.models import load_model
from keras import backend


class Camera:

    def __init__ (self):
        
        self.model = load_model("/home/dpascualhe/workspace/2016-tfg-david-pascual/Net/net.h5")
        
        status = 0
        ic = None
        
        # Initializing the Ice run time
        ic = EasyIce.initialize(sys.argv)
        
        self.lock = threading.Lock()
    
        try:        
            # Obtaining a proxy for the camera (obj. identity: address)
            obj = ic.propertyToProxy("Digitclassifier.Camera.Proxy")
            
            # We get the first image and print its description
            self.cam = CameraPrx.checkedCast(obj)
            if self.cam:
                self.im = self.cam.getImageData("RGB8")
                self.im_height = self.im.description.height
                self.im_width = self.im.description.width
                print(self.im.description)
            else: 
                print("Interface camera not connected")
                    
        except:
            traceback.print_exc()
            exit()
            status = 1

    # This function gets the image from the webcam and trasformates it for the
    # network
    def getImage(self):        
        if self.cam:            
            self.lock.acquire()
            
            im = np.zeros((self.im_height, self.im_width, 3), np.uint8)
            im = np.frombuffer(self.im.pixelData, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3
            im_trans = self.trasformImage(im)
            # It prints a rectangle over the live image where the ROI is
            cv2.rectangle(im, (258, 178), (382, 302), (0, 0, 255), 2)
            ims = [im, im_trans]
            
            self.lock.release()
            
            return ims
    

    # Updates the camera every time the thread changes
    def update(self):
        if self.cam:
            self.lock.acquire()
            
            self.im = self.cam.getImageData("RGB8")
            self.im_height = self.im.description.height
            self.im_width = self.im.description.width
            
            self.lock.release()

    # Trasformates the image for the network
    def trasformImage(self, im):
        kernel = np.ones((3, 3))
        im_crop = im [180:300, 260:380]
        im_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_ero = cv2.erode(im_gray, kernel)    
        im_res = cv2.resize(im_gray, (28, 28))
        (thr, im_bw) = cv2.threshold(im_res, 128, 255,
                                     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        return im_bw
    
    # A Keras convolutional neural network classifies the image
    def classification(self, im):
        # It adapts the shape of the data before entering the network
        if backend.image_dim_ordering() == 'th':
            im = im.reshape(1, 1, im.shape[0], im.shape[1])            
        else:      
            im = im.reshape(1, im.shape[0], im.shape[1], 1)            
        
        # It predicts the input image class    
        dgt = np.where(self.model.predict(im) == 1)
        print("Keras CNN prediction: ", self.model.predict(im))
        print("Prediction index: ", dgt)
        print("--------------------------------------------------------------")
        if dgt[1].size == 1:
            self.digito = dgt
        else:
            self.digito = (([0]), ([0]))
        return self.digito[1][0]
        
