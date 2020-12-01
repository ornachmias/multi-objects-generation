import argparse

from data_generation.bounding_box_replace import BoundingBoxReplace
from data_generation.front_future_3d_render import FrontFuture3DRender
from data_generation.front_model_render import FrontModelRender
from data_generation.future_3d_classification import Future3DClassification
from data_generation.object_net_3d_compose import ObjectNet3DCompose
from data_generation.object_outline import ObjectOutline
from data_generation.scenes_3d_render import Scenes3DRender

from data_generation.segmentation_replace import SegmentationReplace
from data_generation.test_data_generator import TestDataGenerator
from datasets.front_future_3d import FrontFuture3D
from datasets.mscoco import Mscoco
from datasets.object_net_3d import ObjectNet3D
from datasets.scenes_3d import Scenes3D
from datasets.test_dataset import TestDataset
from input_handler import InputHandler

import numpy as np


parser = argparse.ArgumentParser(description='Data Generation Tool')
parser.add_argument('-d', '--dataset',
                    choices=['mscoco', 'object_net_3d', 'test', 'front_future', 'scenes_3d'],
                    default='front_future')
parser.add_argument('-t', '--generation_type',
                    choices=['outlines', 'bboxreplace', 'segreplace', 'compose3d', 'test', 'front_future_render',
                             'scenes_3d_render', 'front_model_render', 'future_classification'],
                    default='front_model_render')
parser.add_argument('-p', '--data_path', default='./data')
parser.add_argument('-c', '--count', default=10000)
parser.add_argument('-m', '--generate_compare', default='true')
parser.add_argument('-b', '--back_object', choices=['none', 'black', 'inpaint'], default='black')

args = parser.parse_args()
user_count = InputHandler.validate_positive_integer(args.count)
user_generate_comparison = InputHandler.str2bool(args.generate_compare)
user_data_path = args.data_path

InputHandler.print_params(args)

dataset = None
if args.dataset == 'mscoco':
    dataset = Mscoco(user_data_path, [1, 2, 3, 4, 5, 6, 7, 8, 9])
elif args.dataset == 'object_net_3d':
    dataset = ObjectNet3D(user_data_path)
elif args.dataset == 'test':
    dataset = TestDataset(user_data_path)
elif args.dataset == 'front_future':
    dataset = FrontFuture3D(user_data_path)
elif args.dataset == 'scenes_3d':
    dataset = Scenes3D(user_data_path)

dataset.initialize()

generator = None
cut_background = args.back_object == 'black'
inpaint = args.back_object == 'inpaint'

if args.generation_type == 'outlines':
    generator = ObjectOutline(user_data_path, dataset)
elif args.generation_type == 'bboxreplace':
    generator = BoundingBoxReplace(user_data_path, dataset, compare_random=user_generate_comparison)
elif args.generation_type == 'segreplace':
    generator = SegmentationReplace(user_data_path, dataset, compare_random=user_generate_comparison,
                                    cut_background=cut_background, inpaint_cut=inpaint)
elif args.generation_type == 'compose3d':
    generator = ObjectNet3DCompose(user_data_path, dataset, compare_random=user_generate_comparison,
                                   cut_background=cut_background, inpaint_cut=inpaint)
elif args.generation_type == 'test':
    generator = TestDataGenerator(dataset)
elif args.generation_type == 'front_future_render':
    mat = np.eye(4)
    mat[1, 3] = 1
    generator = FrontFuture3DRender(dataset, ['chair'], mat, compare_random=user_generate_comparison)
elif args.generation_type == 'scenes_3d_render':
    mat = np.eye(4)
    mat[2, 3] = 50
    generator = Scenes3DRender(dataset, ['chair'], mat, compare_random=user_generate_comparison)
elif args.generation_type == 'front_model_render':
    generator = FrontModelRender(dataset)
elif args.generation_type == 'future_classification':
    generator = Future3DClassification(user_data_path)

generator.generate(user_count)
