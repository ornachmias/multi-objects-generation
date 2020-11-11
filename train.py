import argparse
import os

from input_handler import InputHandler
from models.callbacks.callbacks_handler import CallbacksHandler
from models.inception_v3_binary_classifier import InceptionV3BinaryClassifier
from models.inception_v3_dynamic_classifier import InceptionV3DynamicClassifier

parser = argparse.ArgumentParser(description='Data Classification Tool')
parser.add_argument('-s', '--image_size', type=int, default=256)
parser.add_argument('-t', '--train_metadata', default='./data/bbox_replace/metadata_train.csv')
parser.add_argument('-e', '--eval_metadata', default='./data/bbox_replace/metadata_eval.csv')
parser.add_argument('-b', '--batch_size', type=int, default=20)
parser.add_argument('-p', '--epochs', type=int, default=100)
parser.add_argument('-l', '--logs_dir', default='./data/logs')
parser.add_argument('-c', '--checkpoints_dir', default='./data/checkpoints')
parser.add_argument('-m', '--model_name', default='inception_v3')
parser.add_argument('-g', '--gpus', default='0')
parser.add_argument('-a', '--train_all', default='false')
parser.add_argument('-c', '--classes', default=None, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

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

model.init()
callbacks = CallbacksHandler(args.checkpoints_dir, args.logs_dir).get_callbacks(model.name)
model.train(callbacks=callbacks, epochs=args.epochs)
