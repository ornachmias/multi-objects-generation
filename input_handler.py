import argparse
from enum import Enum


class InputHandler:
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

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'mscoco':
                return InputHandler.Dataset.mscoco
            else:
                argparse.ArgumentTypeError("{} is an invalid dataset.".format(i))

    class GenerationType(Enum):
        unknown = 0
        outlines = 1
        bbox_replace = 2
        seg_replace = 3

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'outlines':
                return InputHandler.GenerationType.outlines
            elif i == 'bboxreplace':
                return InputHandler.GenerationType.bbox_replace
            elif i == 'segreplace':
                return InputHandler.GenerationType.seg_replace
            else:
                argparse.ArgumentTypeError("{} is an invalid generation type.".format(i))

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
