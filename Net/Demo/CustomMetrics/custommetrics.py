import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class CustomMetrics():
    def __init__(self, model, x_test, y_test, batch_size, labels):
        self.y_test = y_test        
        predictions = model.predict(x_test, batch_size=batch_size, verbose=0)
        self.y_pred = np.argmax(predictions, axis=1)
        self.labels = labels

    def confusionMatrix(self):
        conf_mat = confusion_matrix(self.y_test, self.y_pred)      
        print("Confusion matrix:\n")
        print(conf_mat)
        print("\n")
        
        return conf_mat
    
    def classReport(self):
        report = classification_report(self.y_test, self.y_pred, self.labels)
        print("Report:\n")
        print(report)
        print("\n")
        
        return report

    def log(self, file, conf_mat, report, hist=None, curve=None):
        # We log the results.
        if os.path.isfile(file):
            f = open(file, "a")
        else: 
            f = open(file, "w")
        
        f.write("Date: " + str(datetime.datetime.now()) + "\n\n")
        
        if (hist and curve) != None:
            f.write("TRAINING (after each batch)\n")
            f.write("    Loss: " + str(curve.loss) + "\n")
            f.write("    Accuracy: " + str(curve.accuracy) + "\n\n")
            f.write("VALIDATION (after each epoch)\n")
            f.write("    Loss: " + str(hist.val_loss) + "\n")
            f.write("    Accuracy: " + str(hist.acc) + "\n\n")
        
        f.write("TESTING\n")
        f.write("    Confusion matrix:\n" + "    ")
        f.write(conf_mat)
        f.write("\n    Classification report:\n")
        f.write(report)
        f.write("\n    Loss:" + str(score[0]) + "\n")
        f.write("    Accuracy:" + str(score[1]) + "\n\n")  
        f.write("--------------------------------------------------------\n\n")
        f.close()