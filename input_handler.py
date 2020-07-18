import argparse
from enum import Enum


class InputHandler:
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

        @staticmethod
        def parse(i):
            i = i.lower()
            if i == 'outlines':
                return InputHandler.GenerationType.outlines
            else:
                argparse.ArgumentTypeError("{} is an invalid generation type.".format(i))
