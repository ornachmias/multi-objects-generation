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

    def get_callbacks(self):
        self.checkpoint_path = os.path.join(self.checkpoints_dir, self.model_name, 'checkpoint-best.ckpt')
        logs_dir = os.path.join(self.logs_dir, self.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True)
        tensorboard_callback = TensorBoard(log_dir=logs_dir, profile_batch=0)
        early_stop_callback = EarlyStopCallback()
        return [cp_callback, tensorboard_callback, early_stop_callback]



