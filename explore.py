import argparse

from datasets.mscoco import Mscoco
from input_handler import InputHandler

parser = argparse.ArgumentParser(description='Exploration Tool')
parser.add_argument('operation', choices=['show'])
parser.add_argument('-d', '--dataset', choices=['mscoco'], default='mscoco')
parser.add_argument('-p', '--data_path', default='./data')
parser.add_argument('-i', '--image_id', default=None, type=int)

args = parser.parse_args()
user_operation = InputHandler.ExploreOperation.parse(args.operation)
user_dataset = InputHandler.Dataset.parse(args.dataset)
user_data_path = args.data_path

user_image_id = args.image_id
InputHandler.validate_none(user_image_id)

InputHandler.print_params(args)

dataset = None
if user_dataset == InputHandler.Dataset.mscoco:
    dataset = Mscoco(user_data_path)

dataset.initialize()

if user_operation == InputHandler.ExploreOperation.show_image:
    dataset.display_image(user_image_id)
