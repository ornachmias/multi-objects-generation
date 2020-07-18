import argparse

from data_generation.object_outline import ObjectOutline
from datasets.mscoco import Mscoco
from input_handler import InputHandler

parser = argparse.ArgumentParser(description='Multi Object Generation')
parser.add_argument('-d', '--dataset', choices=['mscoco'], default='mscoco')
parser.add_argument('-t', '--generation_type', choices=['outlines'], default='outlines')
parser.add_argument('-p', '--data_path', default='./data')
parser.add_argument('-c', '--count', default=10)

args = parser.parse_args()
user_dataset = InputHandler.Dataset.parse(args.dataset)
user_generation_type = InputHandler.GenerationType.parse(args.generation_type)
user_count = InputHandler.validate_positive_integer(args.count)
user_data_path = args.data_path
print('Received the following parameters: {}'.format(vars(args)))

dataset = None
if user_dataset == InputHandler.Dataset.mscoco:
    dataset = Mscoco(user_data_path)

dataset.initialize()

generator = None
if user_generation_type == InputHandler.GenerationType.outlines:
    generator = ObjectOutline(user_data_path, dataset)

generator.generate(user_count)
