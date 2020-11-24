import pickle
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os

from models.callbacks.early_stop_callback import EarlyStopCallback


class CallbacksHandler:
    def __init__(self, checkpoints_dir, logs_dir, model_name):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint_path = None
        self.logs_dir = logs_dir
        self.model_name = model_name
        if self.checkpoints_dir is not None and self.model_name is not None:
            self.checkpoint_path = os.path.join(self.checkpoints_dir, self.model_name, 'checkpoint-best.ckpt')
            self.classes_path = os.path.join(self.checkpoints_dir, self.model_name, 'classes.pkl')

        if self.logs_dir is not None and self.model_name is not None:
            self.logs_dir = os.path.join(self.logs_dir, self.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))

    def get_callbacks(self):
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                      save_weights_only=True,
                                      verbose=1, save_best_only=True)
        tensorboard_callback = TensorBoard(log_dir=self.logs_dir, profile_batch=0)
        early_stop_callback = EarlyStopCallback()
        return [cp_callback, tensorboard_callback, early_stop_callback]

    def get_trained_classes(self):
        return pickle.load(open(self.classes_path, 'rb'))

    def set_trained_classes(self, classes):
        pickle.dump(classes, open(self.classes_path, 'wb'))



