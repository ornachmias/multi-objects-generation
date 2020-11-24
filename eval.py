import argparse
import os

from input_handler import InputHandler
from models.callbacks.callbacks_handler import CallbacksHandler
from models.inception_v3_multi_classification import InceptionV3MultiClassification

parser = argparse.ArgumentParser(description='Data Classification Tool')
parser.add_argument('-p', '--image_path', default='./data/3d_front/3D-FUTURE-scene/test/image/0000022.jpg')
parser.add_argument('-c', '--checkpoints_dir', default='./data/checkpoints')
parser.add_argument('-m', '--model_name', default='inception_v3_multi')
parser.add_argument('-g', '--gpus', default='0')
parser.add_argument('-s', '--image_size', type=int, default=350)
parser.add_argument('-n', '--classes_num', type=int, default=34)

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)

user_model = InputHandler.Model.parse(args.model_name)
image_path = args.image_path
model = None

if not os.path.exists(image_path):
    print('Image path {} was not found'.format(image_path))

if user_model == InputHandler.Model.inception_v3_multi:
    model = InceptionV3MultiClassification(image_size=args.image_size)

callbacks = CallbacksHandler(args.checkpoints_dir, None, model.name)
model.init()
result = model.eval(image_path=image_path)
print(result)

