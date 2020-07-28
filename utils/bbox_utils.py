import numpy as np
from PIL import Image


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