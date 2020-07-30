import os
import random

from PIL import Image
import numpy as np

from api.generative_inpainting_api import GenerativeInpaintingApi
from data_generation.bounding_box_replace import BoundingBoxReplace
from utils.images_utils import ImagesUtils
from utils.segmentation_utils import SegmentationUtils


class SegmentationReplace(BoundingBoxReplace):
    def __init__(self, root_path, dataset, compare_random=True, cut_background=False, inpaint_cut=False):
        super().__init__(root_path, dataset, 'seg_replace', compare_random)
        self._cut_background = cut_background
        self._inpaint_cut = inpaint_cut
        if inpaint_cut:
            self._inpaint_api = GenerativeInpaintingApi(root_path)

    def _generate_images(self, category_dir, category_id, image_id_1, image_id_2):
        image_1, seg_1, _ = self._dataset.get_image(image_id_1, [category_id])
        image_2, seg_2, _ = self._dataset.get_image(image_id_2, [category_id])
        edited_image_1, edited_image_2 = \
            self.replace_content_segmentation(image_1, seg_1[:, :, 0], image_2, seg_2[:, :, 0])
        ImagesUtils.save_image(edited_image_1, category_dir, str(image_id_1))
        ImagesUtils.save_image(edited_image_2, category_dir, str(image_id_2))

        if self._compare_random:
            compare_output_dir = os.path.join(self._compare_dir, self._categories[category_id])
            self._generate_comparison(image_id_1, image_id_2, image_1, edited_image_1,
                                      image_2, seg_2[:, :, 0], compare_output_dir)
            self._generate_comparison(image_id_2, image_id_1, image_2, edited_image_2,
                                      image_1, seg_1[:, :, 0], compare_output_dir)

    def _generate_comparison(self, image_id_1, image_id_2, image_1, edited_image_1, image_2, seg_2, category_dir):
        random_edit_1 = SegmentationUtils.random_place_segmentation(image_1, image_2, seg_2)
        correct_image_index = random.randint(0, 1)
        if correct_image_index == 0:
            images = [edited_image_1, random_edit_1]
        else:
            images = [random_edit_1, edited_image_1]

        couple = ImagesUtils.concat_images(images)
        path = ImagesUtils.save_image(couple, category_dir, str(image_id_1))
        self._log_comparison(image_id_1, image_id_2, path, correct_image_index)

    def replace_content_segmentation(self, img1, seg1, img2, seg2):
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        bbox1 = SegmentationUtils.get_bbox(seg1)
        bbox2 = SegmentationUtils.get_bbox(seg2)

        seg1 = Image.fromarray(seg1 * 255)
        seg2 = Image.fromarray(seg2 * 255)

        region_image_1 = img1.crop(bbox1)
        region_size_1 = region_image_1.size
        region_seg_1 = seg1.crop(bbox1)

        region_image_2 = img2.crop(bbox2)
        region_size_2 = region_image_2.size
        region_seg_2 = seg2.crop(bbox2)

        if self._cut_background or self._inpaint_cut:
            img1.paste((0, 0, 0), mask=seg1)
            img2.paste((0, 0, 0), mask=seg2)

        if self._inpaint_cut:
            api_result = self._inpaint_api.inpaint(np.array(img1), np.array(seg1))
            img1 = Image.fromarray(api_result)
            api_result = self._inpaint_api.inpaint(np.array(img2), np.array(seg2))
            img2 = Image.fromarray(api_result)

        region_image_1 = region_image_1.resize(region_size_2)
        region_seg_1 = region_seg_1.resize(region_size_2)

        region_image_2 = region_image_2.resize(region_size_1)
        region_seg_2 = region_seg_2.resize(region_size_1)

        img1.paste(region_image_2, box=bbox1, mask=region_seg_2)
        img2.paste(region_image_1, box=bbox2, mask=region_seg_1)
        return np.array(img1), np.array(img2)
