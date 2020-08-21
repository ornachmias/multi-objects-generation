from tensorflow.keras.callbacks import Callback


class EarlyStopCallback(Callback):
    def __init__(self, max_accuracy=0.959):
        super().__init__()
        self._max_accuracy = max_accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        if logs.get('acc') > self._max_accuracy:
            print("\nReached {}% accuracy, stopping training.".format(self._max_accuracy * 100))
            self.model.stop_training = True
