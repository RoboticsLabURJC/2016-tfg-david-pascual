#
# Created on Mar 12, 2017
#
# @author: dpascualhe
#

import numpy as np
import scipy.io as sio
from sklearn import metrics

class CustomMetrics():
    def __init__(self, model, y_test, y_pred, batch_size, curve=None,
                 val=None, training=0):
        """ CustomMetrics class outputs a dictionary with a variety of
        metrics to evaluate the neural network performance.
        """
        self.y_test = y_test
        self.y_pred = y_pred
        self.Y_pred = []
        for label in self.y_pred:
            arr = np.zeros(10)
            arr[label] = 1
            self.Y_pred = self.Y_pred.append(arr)
        
        self.curve = curve
        self.val = val
        self.training = training

    def dictionary(self):
        conf_mat = metrics.confusion_matrix(self.y_test, self.y_pred)
        loss = metrics.log_loss(self.y_test, self.Y_pred)
        acc = metrics.accuracy_score(self.y_test, self.y_pred)
        pre = metrics.precision_score(self.y_test, self.y_pred, average=None)    
        rec = metrics.recall_score(self.y_test, self.y_pred, average=None)
    
        metrics_dict = {"confusion matrix": conf_mat, "loss": loss,
                        "accuracy": acc, "precision": pre, "recall": rec}
        
        if self.training == "y":
            metrics_dict["training accuracy"] = self.curve.accuracy
            metrics_dict["training loss"] = self.curve.loss
            metrics_dict["validation loss"] = self.val.history["val_loss"]
            metrics_dict["validation accuracy"] = self.val.history["acc"]

        return metrics_dict

    def log(self, metrics_dict):
        """ Logs the results into a .mat file for Octave. """
        sio.savemat("metrics.mat", {"metrics": metrics_dict})
