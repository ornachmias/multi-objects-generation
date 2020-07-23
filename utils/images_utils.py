import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from matplotlib.gridspec import GridSpec
from matplotlib import cm
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
    def segmentation_mask_to_map(seg):
        seg_map = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)
        channels = seg.shape[2]
        for i in range(channels):
            seg_map[seg[:, :, i] == 1] = seg[:, :, i][seg[:, :, i] == 1] * i

        return seg_map

    @staticmethod
    def segmentation_map_to_color_map(seg_map):
        color_map = cm.get_cmap('jet', 3)
        new_map = color_map(seg_map)
        new_map = (new_map[:, :, :3] * 255).astype(np.uint8)
        return new_map

    @staticmethod
    def save_image(img, dirpath, filename):
        os.makedirs(dirpath, exist_ok=True)
        Image.fromarray(img).save(os.path.join(dirpath, filename + '.png'))

    @staticmethod
    def bbox_to_mask(bbox_annotation, img_shape):
        x_min = int(bbox_annotation[0])
        x_max = int(bbox_annotation[2]) + x_min
        y_min = int(bbox_annotation[1])
        y_max = int(bbox_annotation[3]) + y_min
        bbox_mask = np.zeros((img_shape[0], img_shape[1]))
        bbox_mask[x_min:x_max, y_min:y_max] = 1
        return bbox_mask

    @staticmethod
    def __get_indices(i, ncols):
        col = int(i % ncols)
        row = math.floor(i / ncols)
        return row, col







