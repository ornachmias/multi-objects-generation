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
