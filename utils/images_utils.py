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
    def replace_content_bbox(img1, bbox1, img2, bbox2):
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        bbox1 = (bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3])
        bbox2 = (bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
        region_image_1 = img1.crop(bbox1)
        region_size_1 = region_image_1.size
        region_image_2 = img2.crop(bbox2)
        region_size_2 = region_image_2.size
        region_image_1 = region_image_1.resize(region_size_2)
        region_image_2 = region_image_2.resize(region_size_1)
        img1.paste(region_image_2, bbox1)
        img2.paste(region_image_1, bbox2)
        return np.array(img1), np.array(img2)

    @staticmethod
    def replace_content_segmentation(img1, seg1, img2, seg2):
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        bbox1 = ImagesUtils.__get_bbox(seg1)
        bbox2 = ImagesUtils.__get_bbox(seg2)

        seg1 = Image.fromarray(seg1 * 255)
        seg2 = Image.fromarray(seg2 * 255)

        region_image_1 = img1.crop(bbox1)
        region_size_1 = region_image_1.size
        region_seg_1 = seg1.crop(bbox1)

        region_image_2 = img2.crop(bbox2)
        region_size_2 = region_image_2.size
        region_seg_2 = seg2.crop(bbox2)

        region_image_1 = region_image_1.resize(region_size_2)
        region_seg_1 = region_seg_1.resize(region_size_2)

        region_image_2 = region_image_2.resize(region_size_1)
        region_seg_2 = region_seg_2.resize(region_size_1)

        img1.paste(region_image_2, box=bbox1, mask=region_seg_2)
        img2.paste(region_image_1, box=bbox2, mask=region_seg_1)
        return np.array(img1), np.array(img2)

    @staticmethod
    def __get_bbox(seg):
        seg_loc = np.where(seg == 1)
        bbox = (np.min(seg_loc[1]), np.min(seg_loc[0]), np.max(seg_loc[1]), np.max(seg_loc[0]))
        return bbox


    @staticmethod
    def __get_indices(i, ncols):
        col = int(i % ncols)
        row = math.floor(i / ncols)
        return row, col







