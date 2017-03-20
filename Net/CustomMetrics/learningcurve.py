import keras

class LearningCurve(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))