from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os

from models.callbacks.early_stop_callback import EarlyStopCallback


class CallbacksHandler:
    def __init__(self, checkpoints_dir, logs_dir):
        self.checkpoints_dir = checkpoints_dir
        self.logs_dir = logs_dir

    def get_callbacks(self, model_name):
        checkpoint_path = os.path.join(self.checkpoints_dir, model_name, 'checkpoint.ckpt')
        logs_dir = os.path.join(self.logs_dir, model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        tensorboard_callback = TensorBoard(log_dir=logs_dir, histogram_freq=1)
        early_stop_callback = EarlyStopCallback()
        return [cp_callback, tensorboard_callback, early_stop_callback]



