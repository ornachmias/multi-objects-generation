import random

import numpy as np
from PIL import Image

from utils.images_utils import ImagesUtils


class BboxUtils:
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
    def random_place_bbox(img1, img2, bbox2):
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        bbox_width = bbox2[2]
        bbox_height = bbox2[3]
        bbox2 = (bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
        region_image_2 = img2.crop(bbox2)
        region_size_2 = region_image_2.size

        if (bbox_width / img1.size[0]) > (bbox_height / img1.size[1]):
            max_resize_factor = img1.size[0] / bbox_width
        else:
            max_resize_factor = img1.size[1] / bbox_height

        max_resize_factor *= 100
        resize_factor = random.randint(1, int(max_resize_factor))
        resize_factor /= 100

        region_image_2 = region_image_2.resize((int(region_size_2[0] * resize_factor),
                                                int(region_size_2[1] * resize_factor)))
        x, y = ImagesUtils.get_random_position(region_image_2.size[0], region_image_2.size[1],
                                               img1.size[0], img1.size[1])
        img1.paste(region_image_2, (x, y))
        return np.array(img1)

