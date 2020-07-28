import numpy as np
from matplotlib import cm
from PIL import Image


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
    def replace_content_segmentation(img1, seg1, img2, seg2):
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        bbox1 = SegmentationUtils.__get_bbox(seg1)
        bbox2 = SegmentationUtils.__get_bbox(seg2)

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
