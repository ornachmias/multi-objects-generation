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

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'mscoco':
                return InputHandler.Dataset.mscoco
            if i == 'object_net_3d':
                return InputHandler.Dataset.object_net_3d
            if i == 'test':
                return InputHandler.Dataset.test
            else:
                argparse.ArgumentTypeError("{} is an invalid dataset.".format(i))

    class GenerationType(Enum):
        unknown = 0
        outlines = 1
        bbox_replace = 2
        seg_replace = 3
        compose_3d = 4

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

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'show':
                return InputHandler.ExploreOperation.show_image
            else:
                argparse.ArgumentTypeError("{} is an invalid explore operation.".format(i))

    class Model(Enum):
        unknown = 0
        inception_v3 = 1

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'inception_v3':
                return InputHandler.Model.inception_v3
            else:
                argparse.ArgumentTypeError("{} is an invalid model.".format(i))
