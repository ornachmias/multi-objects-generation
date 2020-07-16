import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage import io
import math


class ImagesUtils:
    @staticmethod
    def convert_to_numpy(raw_img):
        if isinstance(raw_img, str):
            img = io.imread(raw_img)
        elif isinstance(raw_img, np.ndarray):
            img = raw_img
        elif isinstance(raw_img, list):
            img = np.vstack(raw_img)
        else:
            raise Exception('Could not handle input of type "{}"'.format(type(raw_img)))

        return img.astype(np.uint8)

    @staticmethod
    def show_multiple_images(imgs, titles=None, ncols=3):
        matplotlib.use('TkAgg')
        nrows = math.ceil(len(imgs) / ncols)
        fig = plt.figure()
        spec = GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        if titles is not None:
            assert len(imgs) == len(titles)

        for i in range(len(imgs)):
            row, col = ImagesUtils.get_indices(i, ncols)
            ax = fig.add_subplot(spec[row, col])
            ax.axis('off')
            if titles is not None:
                ax.set_title(titles[i])
            ax.imshow(ImagesUtils.convert_to_numpy(imgs[i]), interpolation='nearest')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def get_indices(i, ncols):
        col = int(i % ncols)
        row = math.floor(i / ncols)
        return row, col







