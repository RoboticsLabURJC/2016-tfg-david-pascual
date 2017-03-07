'''
Created on Mar 7, 2017

@author: dpascualhe

Camera class

Based on @nuriaoyaga code:
https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/master/camera/camera.py

And @Javii91 code:
https://github.com/Javii91/Domotic/blob/master/Others/cameraview.py

'''

import sys, traceback, threading, random
import numpy as np
from PIL import Image
from jderobot import CameraPrx
import Ice
import cv2


class Camera:

    def __init__ (self):
                
        status = 0
        ic = None
        
        self.lock = threading.Lock()
    
        try:        
            # Initializing the Ice run time
            ic = Ice.initialize()
            # Obtaining a proxy for the camera (obj. identity: address)
            obj = ic.stringToProxy('cameraA:default -h localhost -p 9999')
            
            # We get the first image and print its description
            self.cam = CameraPrx.checkedCast(obj)
            if self.cam:
                self.im = self.cam.getImageData("RGB8")
                self.im_height= self.im.description.height
                self.im_width = self.im.description.width
                print(self.im.description)
            else: 
                print("Interface camera not connected")
                    
        except:
            traceback.print_exc()
            status = 1

    # This function gets the image from the webcam and trasformates it for the network
    def getImage(self):        
        if self.cam:            
            self.lock.acquire()
            
            im = np.zeros((self.im_height, self.im_width, 3), np.uint8)
            im = np.frombuffer(self.im.pixelData, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3
            im_trans = self.trasformImage(im)
            ims = [im,im_trans]
            
            self.lock.release()
            
            print("get")
            
            return ims
    

    # Updates the camera every time the thread changes
    def update(self):
        if self.cam:
            self.lock.acquire()
            
            self.im = self.cam.getImageData("RGB8")
            self.im_height= self.im.description.height
            self.im_width = self.im.description.width
            
            self.lock.release()
            
            print("update")

    # Trasformates the image for the network
    def trasformImage(self, im):
        kernel = np.ones((3, 3))
        cv2.rectangle(im, (218, 138), (422, 342), (0, 0, 255), 2)
        im_crop = im [140:340, 220:420]
        
        im_bw = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_dil = cv2.dilate(im_bw, kernel)
        im_erode = cv2.erode(im_bw, kernel)
        im_trans = im_dil - im_erode
        
        print("trans")
        
        return im_trans
    
    # A Keras convolutional neural network classifies the image
    def detection(self, im):
        digito = random.randrange (0, 9, 1)

        print("detect")

        return digito
        
