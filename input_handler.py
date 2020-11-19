import argparse
from enum import Enum


class InputHandler:
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def print_params(args):
        print('Received the following parameters: {}'.format(vars(args)))

    @staticmethod
    def validate_none(x):
        if x is None:
            raise argparse.ArgumentTypeError("{} is None but should be set.".format(x))

    @staticmethod
    def validate_positive_integer(x):
        xt = int(x)
        if xt <= 0:
            raise argparse.ArgumentTypeError("{} is an invalid positive int value".format(x))
        return xt

    @staticmethod
    def to_array(x):
        if not isinstance(x, (list, tuple)):
            return [x]

        return x

    class Dataset(Enum):
        unknown = 0
        mscoco = 1
        object_net_3d = 2
        test = 3
        front_future = 4
        scenes_3d = 5

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'mscoco':
                return InputHandler.Dataset.mscoco
            if i == 'object_net_3d':
                return InputHandler.Dataset.object_net_3d
            if i == 'test':
                return InputHandler.Dataset.test
            if i == 'front_future':
                return InputHandler.Dataset.front_future
            if i == 'scenes_3d':
                return InputHandler.Dataset.scenes_3d
            else:
                argparse.ArgumentTypeError("{} is an invalid dataset.".format(i))

    class GenerationType(Enum):
        unknown = 0
        outlines = 1
        bbox_replace = 2
        seg_replace = 3
        compose_3d = 4
        test = 5
        front_future_render = 6
        scenes_3d_render = 7
        front_model_render = 8
        future_classification = 9

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'outlines':
                return InputHandler.GenerationType.outlines
            elif i == 'bboxreplace':
                return InputHandler.GenerationType.bbox_replace
            elif i == 'segreplace':
                return InputHandler.GenerationType.seg_replace
            elif i == 'compose3d':
                return InputHandler.GenerationType.compose_3d
            elif i == 'test':
                return InputHandler.GenerationType.test
            elif i == 'front_future_render':
                return InputHandler.GenerationType.front_future_render
            elif i == 'scenes_3d_render':
                return InputHandler.GenerationType.scenes_3d_render
            elif i == 'front_model_render':
                return InputHandler.GenerationType.front_model_render
            elif i == 'future_classification':
                return InputHandler.GenerationType.future_classification
            else:
                argparse.ArgumentTypeError("{} is an invalid generation type.".format(i))

    class BackgroundObject(Enum):
        unknown = 0
        none = 1
        paint_black = 2
        inpaint = 3

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'none':
                return InputHandler.BackgroundObject.none
            elif i == 'black':
                return InputHandler.BackgroundObject.paint_black
            elif i == 'inpaint':
                return InputHandler.BackgroundObject.inpaint
            else:
                argparse.ArgumentTypeError("{} is an invalid background object operation.".format(i))

    class ExploreOperation(Enum):
        unknown = 0
        show_image = 1
        count_categories = 2

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'show':
                return InputHandler.ExploreOperation.show_image
            if i == 'count_categories':
                return InputHandler.ExploreOperation.count_categories
            else:
                argparse.ArgumentTypeError("{} is an invalid explore operation.".format(i))

    class Model(Enum):
        unknown = 0
        inception_v3 = 1
        inception_v3_multi = 2

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'inception_v3':
                return InputHandler.Model.inception_v3
            elif i == 'inception_v3_multi':
                return InputHandler.Model.inception_v3_multi
            else:
                argparse.ArgumentTypeError("{} is an invalid model.".format(i))
