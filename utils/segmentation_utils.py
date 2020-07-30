import random

import numpy as np
from matplotlib import cm
from PIL import Image

from utils.images_utils import ImagesUtils


class SegmentationUtils:
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
    def random_place_segmentation(img1, img2, seg2):
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        bbox2 = SegmentationUtils.get_bbox(seg2)
        seg2 = Image.fromarray(seg2 * 255)

        bbox_width = bbox2[2]
        bbox_height = bbox2[3]
        bbox2 = (bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
        region_image_2 = img2.crop(bbox2)
        region_size_2 = region_image_2.size
        region_seg_2 = seg2.crop(bbox2)

        if (bbox_width / img1.size[0]) > (bbox_height / img1.size[1]):
            max_resize_factor = img1.size[0] / bbox_width
        else:
            max_resize_factor = img1.size[1] / bbox_height

        max_resize_factor *= 100
        resize_factor = random.randint(1, int(max_resize_factor))
        resize_factor /= 100

        region_image_2 = region_image_2.resize((int(region_size_2[0] * resize_factor),
                                                int(region_size_2[1] * resize_factor)))
        region_seg_2 = region_seg_2.resize((int(region_size_2[0] * resize_factor),
                                            int(region_size_2[1] * resize_factor)))

        x, y = ImagesUtils.get_random_position(region_image_2.size[0], region_image_2.size[1],
                                               img1.size[0], img1.size[1])
        img1.paste(region_image_2, (x, y), mask=region_seg_2)
        return np.array(img1)

    @staticmethod
    def get_bbox(seg):
        seg_loc = np.where(seg == 1)
        bbox = (np.min(seg_loc[1]), np.min(seg_loc[0]), np.max(seg_loc[1]), np.max(seg_loc[0]))
        return bbox
