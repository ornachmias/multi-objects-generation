import argparse
import os

from input_handler import InputHandler
from models.callbacks.callbacks_handler import CallbacksHandler
from models.inception_v3_binary_classifier import InceptionV3BinaryClassifier
from models.inception_v3_dynamic_classifier import InceptionV3DynamicClassifier
import tensorflow as tf

from models.inception_v3_multi_classification import InceptionV3MultiClassification

parser = argparse.ArgumentParser(description='Data Classification Tool')
parser.add_argument('-s', '--image_size', type=int, default=256)
parser.add_argument('-t', '--train_metadata', default='./data/3d_front/generated_classification/train/metadata.csv')
parser.add_argument('-e', '--eval_metadata', default='./data/3d_front/generated_classification/test/metadata.csv')
parser.add_argument('-b', '--batch_size', type=int, default=20)
parser.add_argument('-p', '--epochs', type=int, default=100)
parser.add_argument('-l', '--logs_dir', default='./data/logs')
parser.add_argument('-c', '--checkpoints_dir', default='./data/checkpoints')
parser.add_argument('-m', '--model_name', default='inception_v3_multi')
parser.add_argument('-g', '--gpus', default='0')
parser.add_argument('-a', '--train_all', default='false')
parser.add_argument('-n', '--classes', default=None, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

user_model = InputHandler.Model.parse(args.model_name)
user_train_all = InputHandler.str2bool(args.train_all)
number_of_classes = args.classes
model = None

if user_model == InputHandler.Model.inception_v3:
    if number_of_classes is None:
        model = InceptionV3BinaryClassifier(train_metadata_path=args.train_metadata,
                                            eval_metadata_path=args.eval_metadata,
                                            image_size=args.image_size, batch_size=args.batch_size,
                                            train_all=user_train_all)
    else:
        model = InceptionV3DynamicClassifier(number_of_classes=number_of_classes,
                                             train_metadata_path=args.train_metadata,
                                             eval_metadata_path=args.eval_metadata,
                                             image_size=args.image_size,
                                             batch_size=args.batch_size,
                                             train_all=user_train_all)
elif user_model == InputHandler.Model.inception_v3_multi:
    model = InceptionV3MultiClassification(train_metadata_path=args.train_metadata,
                                           eval_metadata_path=args.eval_metadata,
                                           image_size=args.image_size, batch_size=args.batch_size,
                                           train_all=user_train_all)

callbacks = CallbacksHandler(args.checkpoints_dir, args.logs_dir, model.name)
model.init(callback_handler=callbacks)
model.train(epochs=args.epochs)
