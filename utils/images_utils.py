import os
import random

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
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
            row, col = ImagesUtils.__get_indices(i, ncols)
            ax = fig.add_subplot(spec[row, col])
            ax.axis('off')
            if titles is not None:
                ax.set_title(titles[i])
            ax.imshow(ImagesUtils.convert_to_numpy(imgs[i]), interpolation='nearest')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def save_image(img, dirpath, filename):
        os.makedirs(dirpath, exist_ok=True)
        path = os.path.join(dirpath, filename + '.png')
        if os.path.exists(path):
            return None
        Image.fromarray(img).save(path)
        return path

    @staticmethod
    def concat_images(images):
        images = [Image.fromarray(img) for img in images]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        return np.array(combined)

    @staticmethod
    def __get_indices(i, ncols):
        col = int(i % ncols)
        row = math.floor(i / ncols)
        return row, col

    @staticmethod
    def get_random_position(crop_width, crop_height, image_weight, image_height):
        x_range = image_weight - crop_width
        y_range = image_height - crop_height
        return random.randint(0, x_range), random.randint(0, y_range)







