import argparse

from data_generation.scenes_3d_bbox_render_random import Scenes3DBboxRenderRandom
from data_generation.scenes_3d_bbox_render_transform import Scenes3DBboxRenderTransform

parser = argparse.ArgumentParser(description='Data Generation Tool V2')
parser.add_argument('-d', '--data_dir', default='./data')
parser.add_argument('-t', '--type', default='', choices=['scenes_3d_bbox_render_transform',
                                                         'scenes_3d_bbox_render_random'])
args = parser.parse_args()

generator = None
if args.type == 'scenes_3d_bbox_render_transform':
    generator = Scenes3DBboxRenderTransform(args.data_dir)
elif args.type == 'scenes_3d_bbox_render_random':
    generator = Scenes3DBboxRenderRandom(args.data_dir)

generator.initialize()
generator.generate()
