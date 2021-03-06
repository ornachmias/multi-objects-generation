import math
import os
import random

import numpy as np
from PIL import Image

from data_generation.base_generator import BaseGenerator
from datasets.mscoco import Mscoco
from utils.images_utils import ImagesUtils
from utils.bbox_utils import BboxUtils


class BoundingBoxReplace(BaseGenerator):
    def __init__(self, root_path, dataset, output_dir_name='bbox_replace', compare_random=False):
        super().__init__(dataset)
        assert isinstance(dataset, Mscoco), "Generator support only " + Mscoco.__name__ + " dataset"

        self._output_dir = os.path.join(root_path, output_dir_name)
        os.makedirs(self._output_dir, exist_ok=True)
        print('Setting {} output directory as {}'.format(self.__class__.__name__, self._output_dir))

        self._metadata = os.path.join(self._output_dir, 'metadata.csv')

        self._compare_random = compare_random
        if self._compare_random:
            self._compare_dir = os.path.join(self._output_dir, 'compare')
            os.makedirs(self._compare_dir, exist_ok=True)
            self._compare_metadata = os.path.join(self._compare_dir, 'metadata.csv')

        self._ratio_groups = 5
        self._batch_size = 20
        self._categories = self._get_dataset_categories()
        print('Generating data for categories={}'.format(self._categories))

    def generate(self, count):
        for category_id in self._categories:
            print('Generating data for category {}'.format(category_id))
            images_count = 0
            index = 0
            used_images = {}

            category_dir = os.path.join(self._output_dir, self._categories[category_id])

            while images_count < count:
                print('Current images in category: {}'.format(images_count))

                image_ids = self._dataset.get_image_ids([category_id])[images_count:self._batch_size + index]
                index += self._batch_size
                if image_ids is None or len(image_ids) == 0:
                    break

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

                        try:
                            self._generate_images(category_dir, category_id, image_id_1, image_id_2)
                            used_images[image_id_1] = 1
                            used_images[image_id_2] = 1
                            images_count += 2
                        except:
                            print('Failed to generate images for ids: {}, {}'.format(image_id_1, image_id_2))

            print('Generated {} images in category {}'.format(images_count, category_id))

        self._split_data()



    def _generate_images(self, category_dir, category_id, image_id_1, image_id_2):
        image_1, _, bboxes_1 = self._dataset.get_image(image_id_1, [category_id])
        image_2, _, bboxes_2 = self._dataset.get_image(image_id_2, [category_id])
        edited_image_1, edited_image_2 = \
            self.replace_content_bbox(image_1, bboxes_1[0], image_2, bboxes_2[0])

        path = ImagesUtils.save_image(edited_image_1, category_dir, '{}_edited'.format(str(image_id_1)))
        if path is not None:
            self._log(image_id_1, path, is_correct=1)

        path = ImagesUtils.save_image(edited_image_2, category_dir, '{}_edited'.format(str(image_id_2)))
        if path is not None:
            self._log(image_id_2, path, is_correct=1)

        if self._compare_random:
            compare_output_dir = os.path.join(self._compare_dir, self._categories[category_id])
            self._generate_comparison(image_id_1, image_id_2, image_1, edited_image_1,
                                      image_2, bboxes_2[0], compare_output_dir)
            self._generate_comparison(image_id_2, image_id_1, image_2, edited_image_2,
                                      image_1, bboxes_1[0], compare_output_dir)

    def _generate_comparison(self, image_id_1, image_id_2, image_1, edited_image_1, image_2, bbox_2, category_dir):
        random_edit_1 = BboxUtils.random_place_bbox(image_1, image_2, bbox_2)

        path = ImagesUtils.save_image(random_edit_1, category_dir, '{}_random'.format(str(image_id_1)))
        if path is not None:
            self._log(image_id_1, path, is_correct=0)

        correct_image_index = random.randint(0, 1)
        if correct_image_index == 0:
            images = [edited_image_1, random_edit_1]
        else:
            images = [random_edit_1, edited_image_1]

        couple = ImagesUtils.concat_images(images)
        path = ImagesUtils.save_image(couple, category_dir, '{}_{}'.format(str(image_id_1), str(image_id_2)))
        if path is not None:
            self._log_comparison(image_id_1, image_id_2, path, correct_image_index)

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

    def replace_content_bbox(self, img1, bbox1, img2, bbox2):
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


