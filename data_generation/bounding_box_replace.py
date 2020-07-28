import math
import os
import numpy as np

from data_generation.base_generator import BaseGenerator
from utils.images_utils import ImagesUtils


class BoundingBoxReplace(BaseGenerator):
    def __init__(self, root_path, dataset, output_dir_name='bbox_replace'):
        super().__init__(dataset)

        self._output_dir = os.path.join(root_path, output_dir_name)
        os.makedirs(self._output_dir, exist_ok=True)

        self._ratio_groups = 5
        self._batch_size = 20
        self._categories = self._get_dataset_categories()

    def generate(self, count):
        for category_id in self._categories:
            images_count = 0
            used_images = {}

            while images_count < count:
                category_dir = os.path.join(self._output_dir, self._categories[category_id])
                image_ids = self._dataset.get_image_ids([category_id])[images_count:self._batch_size + images_count]
                images_categorization = self._categorize_images(image_ids, category_id)

                for ratio_category in images_categorization:
                    if len(images_categorization[ratio_category]) <= 1:
                        continue

                    sorted_resolutions = sorted(images_categorization[ratio_category],
                                                key=images_categorization[ratio_category].get)
                    for i in range(0, len(sorted_resolutions) - 1, 2):
                        if images_count >= count:
                            break

                        image_id_1 = sorted_resolutions[i]
                        image_id_2 = sorted_resolutions[i + 1]
                        if image_id_1 in used_images or image_id_2 in used_images:
                            continue

                        self._generate_images(category_dir, category_id, image_id_1, image_id_2)
                        used_images[image_id_1] = 1
                        used_images[image_id_2] = 1
                        images_count += 2

    def _generate_images(self, category_dir, category_id, image_id_1, image_id_2):
        image_1, _, bboxes_1 = self._dataset.get_image(image_id_1, [category_id])
        image_2, _, bboxes_2 = self._dataset.get_image(image_id_2, [category_id])
        edited_image_1, edited_image_2 = \
            ImagesUtils.replace_content_bbox(image_1, bboxes_1[0], image_2, bboxes_2[0])
        ImagesUtils.save_image(edited_image_1, category_dir, str(image_id_1))
        ImagesUtils.save_image(edited_image_2, category_dir, str(image_id_2))

    def _categorize_images(self, image_ids, category_id):
        images_metadata = {}
        ratios = []
        for image_id in image_ids:
            _, _, bboxes = self._dataset.get_image(image_id, [category_id])
            bbox = bboxes[0]
            w, h = bbox[2], bbox[3]
            images_metadata[image_id] = (w / h, w * h)
            ratios.append(w / h)

        min_ratio = np.min(ratios)
        max_ratio = np.max(ratios)

        result = {i: {} for i in range(self._ratio_groups)}
        for image_id in images_metadata:
            current_ratio = images_metadata[image_id][0]
            current_res = images_metadata[image_id][1]
            ratio_category = self._get_ratio_category(min_ratio, max_ratio, current_ratio)
            result[ratio_category][image_id] = current_res

        return result

    def _get_ratio_category(self, min_ratio, max_ratio, ratio):
        ratio_range = max_ratio - min_ratio
        range_size = ratio_range / (self._ratio_groups - 1)
        return math.floor((ratio - min_ratio) / range_size)

    def _get_dataset_categories(self):
        categories = {}
        category_ids = self._dataset.category_ids
        if not category_ids:
            category_ids = [c[0] for c in self._dataset.get_categories()]

        for c in self._dataset.get_categories():
            if c[0] in category_ids:
                categories[c[0]] = c[1]

        return categories


