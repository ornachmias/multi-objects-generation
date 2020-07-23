import os
from PIL import Image
import numpy as np

from data_generation.base_generator import BaseGenerator
from utils.images_utils import ImagesUtils


class BoundingBoxReplace(BaseGenerator):
    def __init__(self, root_path, dataset):
        super().__init__(dataset)

        self._bbox_replace_dir = os.path.join(root_path, 'bbox_replace')
        os.makedirs(self._bbox_replace_dir, exist_ok=True)

    def generate(self, count):
        category_ids = self._dataset.category_ids
        if not category_ids:
            category_ids = [c[0] for c in self._dataset.get_categories()]

        category_names = {}
        for c in self._dataset.get_categories():
            category_names[c[0]] = c[1]

        for category_id in category_ids:
            category_dir = os.path.join(self._bbox_replace_dir, category_names[category_id])
            image_ids = self._dataset.get_image_ids([category_id])[:count]
            ratios = self._get_images_bbox_ratio(image_ids, category_id)
            sorted_ratios = sorted(ratios, key=ratios.get)
            for i in range(len(sorted_ratios) - 1):
                image_id_1 = sorted_ratios[i]
                image_id_2 = sorted_ratios[i + 1]
                image_1, _, bboxes_1 = self._dataset.get_image(image_id_1, [category_id])
                image_2, _, bboxes_2 = self._dataset.get_image(image_id_2, [category_id])
                edited_image_1, edited_image_2 = self._replace_content(image_1, bboxes_1[0], image_2, bboxes_2[0])
                ImagesUtils.save_image(edited_image_1, category_dir, str(image_id_1))
                ImagesUtils.save_image(edited_image_2, category_dir, str(image_id_2))

    def _get_images_bbox_ratio(self, image_ids, category_id):
        result = {}
        for image_id in image_ids:
            _, _, bboxes = self._dataset.get_image(image_id, [category_id])
            bbox = bboxes[0]
            w, h = bbox[2], bbox[3]
            result[image_id] = w / h

        return result

    def _replace_content(self, img1, bbox1, img2, bbox2):
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


