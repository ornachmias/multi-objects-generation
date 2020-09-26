import argparse

from data_generation.bounding_box_replace import BoundingBoxReplace
from data_generation.object_net_3d_compose import ObjectNet3DCompose
from data_generation.object_outline import ObjectOutline

from data_generation.segmentation_replace import SegmentationReplace
from data_generation.test_data_generator import TestDataGenerator
from datasets.mscoco import Mscoco
from datasets.object_net_3d import ObjectNet3D
from input_handler import InputHandler

parser = argparse.ArgumentParser(description='Data Generation Tool')
parser.add_argument('-d', '--dataset', choices=['mscoco', 'object_net_3d', 'test'], default='object_net_3d')
parser.add_argument('-t', '--generation_type', choices=['outlines', 'bboxreplace', 'segreplace', 'compose3d', 'test'], default='compose3d')
parser.add_argument('-p', '--data_path', default='./data')
parser.add_argument('-c', '--count', default=5)
parser.add_argument('-m', '--generate_compare', default='true')
parser.add_argument('-b', '--back_object', choices=['none', 'black', 'inpaint'], default='black')

args = parser.parse_args()
user_dataset = InputHandler.Dataset.parse(args.dataset)
user_generation_type = InputHandler.GenerationType.parse(args.generation_type)
user_count = InputHandler.validate_positive_integer(args.count)
user_generate_comparison = InputHandler.str2bool(args.generate_compare)
user_back_object = InputHandler.BackgroundObject.parse(args.back_object)
user_data_path = args.data_path

InputHandler.print_params(args)

dataset = None
if user_dataset == InputHandler.Dataset.mscoco:
    dataset = Mscoco(user_data_path, [1, 2, 3, 4, 5, 6, 7, 8, 9])
elif user_dataset == InputHandler.Dataset.object_net_3d:
    dataset = ObjectNet3D(user_data_path)
elif user_dataset == InputHandler.Dataset.test:
    dataset = ObjectNet3D(user_data_path)

dataset.initialize()

generator = None
cut_background = user_back_object == InputHandler.BackgroundObject.paint_black
inpaint = user_back_object == InputHandler.BackgroundObject.inpaint

if user_generation_type == InputHandler.GenerationType.outlines:
    generator = ObjectOutline(user_data_path, dataset)
elif user_generation_type == InputHandler.GenerationType.bbox_replace:
    generator = BoundingBoxReplace(user_data_path, dataset, compare_random=user_generate_comparison)
elif user_generation_type == InputHandler.GenerationType.seg_replace:
    generator = SegmentationReplace(user_data_path, dataset, compare_random=user_generate_comparison,
                                    cut_background=cut_background, inpaint_cut=inpaint)
elif user_generation_type == InputHandler.GenerationType.compose_3d:
    generator = ObjectNet3DCompose(user_data_path, dataset, compare_random=user_generate_comparison,
                                   cut_background=cut_background, inpaint_cut=inpaint)
elif user_generation_type == InputHandler.GenerationType.test:
    generator = TestDataGenerator(dataset)

generator.generate(user_count)
