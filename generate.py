import argparse

from data_generation.bounding_box_replace import BoundingBoxReplace
from data_generation.object_outline import ObjectOutline

from data_generation.segmentation_replace import SegmentationReplace
from datasets.mscoco import Mscoco
from input_handler import InputHandler

parser = argparse.ArgumentParser(description='Data Generation Tool')
parser.add_argument('-d', '--dataset', choices=['mscoco'], default='mscoco')
parser.add_argument('-t', '--generation_type', choices=['outlines', 'bboxreplace', 'segreplace'], default='segreplace')
parser.add_argument('-p', '--data_path', default='./data')
parser.add_argument('-c', '--count', default=20)
parser.add_argument('-m', '--generate_compare', default='true')
parser.add_argument('-b', '--back_object', choices=['none', 'black', 'inpaint'], default='inpaint')

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

dataset.initialize()

generator = None
if user_generation_type == InputHandler.GenerationType.outlines:
    generator = ObjectOutline(user_data_path, dataset)
elif user_generation_type == InputHandler.GenerationType.bbox_replace:
    generator = BoundingBoxReplace(user_data_path, dataset, compare_random=user_generate_comparison)
elif user_generation_type == InputHandler.GenerationType.seg_replace:
    cut_background = user_back_object == InputHandler.BackgroundObject.paint_black
    inpaint = user_back_object == InputHandler.BackgroundObject.inpaint
    generator = SegmentationReplace(user_data_path, dataset, compare_random=user_generate_comparison,
                                    cut_background=cut_background, inpaint_cut=inpaint)

generator.generate(user_count)
