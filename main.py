'''
Created on Mar 2, 2017

@author: dpascualhe

Pendiente:
    .cfg
    componente android
    insertar red
'''

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from PyQt5 import QtWidgets
import sys

if __name__ == '__main__':
    
    # We define the classes that we're going to need
    cam = Camera()
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.setCamera(cam)
    gui.show()
    
    # Threading camera
    t_cam = ThreadCamera(cam)
    t_cam.start()
    
    # Threading GUI
    t_gui = ThreadGUI(gui)
    t_gui.start()
    
    sys.exit(app.exec_())