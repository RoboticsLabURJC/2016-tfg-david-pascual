#
# Created on Mar 25, 2017
#
# @author: dpascualhe
#
# Figuring out an appropiate way to represent metrics. Data must work 
# as input for Octave. It will replace CustomMetrics.
#

import os

import numpy as np
import scipy.io as sio
from sklearn import metrics
from keras.models import load_model

from DataManager.netdata import NetData

if __name__ == '__main__':
    test_ds = input("Test dataset path: ")
    while not os.path.isfile(test_ds):
        test_ds = input("Enter a valid path: ")
    
    model = load_model("/home/dpascualhe/workspace/2016-tfg-david-pascual" \
                       + "/Net/Nets/net_1-1.h5")
    
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    nb_classes = 10
    batch_size = 128
    im_rows, im_cols = 28, 28
    data = NetData(im_rows, im_cols, nb_classes)
    (X_test, Y_test) = data.load(test_ds)
    (x_test, y_test), input_shape = data.adapt(X_test, Y_test, 0)
    
    ''' New CustomMetrics '''
    
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    Y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)
        
    kloss = np.array(score[0])
    kacc = np.array(score[1])
    conf_mat = metrics.confusion_matrix(Y_test, y_pred)
    loss = metrics.log_loss(Y_test, Y_pred)
    acc = metrics.accuracy_score(Y_test, y_pred)
    pre = metrics.precision_score(Y_test, y_pred, average="macro")    
    rec = metrics.recall_score(Y_test, y_pred, average="macro")
    
    metrics_dict = {'keras loss': kloss, 'keras accuracy': kacc, 
                    'confusions matrix': conf_mat, 'sklearn loss': loss,
                    'sklearn accuracy': acc, 'recall': rec}
    sio.savemat('metrics.mat', {'metrics': metrics_dict})
