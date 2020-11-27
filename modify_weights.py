import argparse

from tensorflow.keras.optimizers import RMSprop

from input_handler import InputHandler
from models.callbacks.callbacks_handler import CallbacksHandler
from models.inception_v3_multi_classification import InceptionV3MultiClassification
from tensorflow.keras import layers
from tensorflow.keras import Model

parser = argparse.ArgumentParser(description='Modify checkpoint files')
parser.add_argument('-c', '--checkpoints_dir', default='./data/checkpoints')
parser.add_argument('-m', '--model_name', default='inception_v3_multi')
parser.add_argument('-t', '--previous_tag', default='34_classes')
parser.add_argument('-g', '--new_tag', default='2_classes')
parser.add_argument('-s', '--image_size', type=int, default=350)
parser.add_argument('-n', '--new_classes_num', type=int, default=2)

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)

user_model = InputHandler.Model.parse(args.model_name)
model = None

if user_model == InputHandler.Model.inception_v3_multi:
    model = InceptionV3MultiClassification(image_size=args.image_size, is_eval=True, model_tag=args.previous_tag)

callback_handler = CallbacksHandler(args.checkpoints_dir, None, model.name)
model.init(callback_handler)
model.load_model()
x = layers.Dense(args.new_classes_num, activation='sigmoid')(model.model.layers[-2].output)
model.model = Model(model.model.input, x)
model.model.compile(optimizer=RMSprop(lr=model.learning_rate), loss=model.loss, metrics=['acc'])
model.name = model.name.rstrip(args.previous_tag)
model.name += '_' + args.new_tag
callback_handler = CallbacksHandler(args.checkpoints_dir, None, model.name)
model.callback_handler = callback_handler
model.save_model()

