#
# Created on Mar 12, 2017
#
# @author: dpascualhe
#

import numpy as np
import scipy.io as sio
from sklearn import metrics

class CustomMetrics():
    def __init__(self, model, x_test, y_test, batch_size, curve=None,
                 val=None, training=0):
        """ CustomMetrics class outputs a dictionary with a variety of
        metrics to evaluate the neural network performance.
        """
        self.y_test = y_test
        self.Y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
        self.y_pred = np.argmax(self.Y_pred, axis=1)
        
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
            metrics_dict["validation accuracy"] = self.val.history["val_loss"]
            metrics_dict["validation loss"] = self.val.history["acc"]

        return metrics_dict

    def log(self, metrics_dict):
        """ Logs the results into a .mat file for Octave. """
        sio.savemat("metrics.mat", {"metrics": metrics_dict})
