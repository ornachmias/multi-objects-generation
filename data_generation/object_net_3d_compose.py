import os
import random
import traceback

import numpy as np
from PIL import Image

from api.generative_inpainting_api import GenerativeInpaintingApi
from data_generation.base_generator import BaseGenerator
from datasets.object_net_3d import ObjectNet3D
from utils.images_utils import ImagesUtils


class ObjectNet3DCompose(BaseGenerator):
    def __init__(self, root_path, dataset, compare_random=False, cut_background=False, inpaint_cut=False):
        super().__init__(dataset)
        assert isinstance(dataset, ObjectNet3D), "Generator support only " + ObjectNet3D.__name__ + " dataset"

        self._output_dir = os.path.join(root_path, 'object_3d_net', 'generated')
        os.makedirs(self._output_dir, exist_ok=True)
        print('Setting {} output directory as {}'.format(self.__class__.__name__, self._output_dir))

        self._metadata = os.path.join(self._output_dir, 'metadata.csv')
        self._compare_random = compare_random
        if self._compare_random:
            self._compare_dir = os.path.join(self._output_dir, 'compare')
            os.makedirs(self._compare_dir, exist_ok=True)
            self._compare_metadata = os.path.join(self._compare_dir, 'metadata.csv')

        self._cut_background = cut_background
        self._inpaint_cut = inpaint_cut
        if inpaint_cut:
            self._inpaint_api = GenerativeInpaintingApi(root_path)

    def generate(self, count):
        train_ids, _, _ = self._dataset.get_image_ids()
        random.shuffle(train_ids)
        images_count = 0
        i = 0

        while images_count < count:
            image_id = train_ids[i]
            renders = self._dataset.get_renders(image_id)
            if not renders:
                i += 1
                continue

            generated_image_index = 0
            for record, render in renders:
                try:
                    render = render.crop()
                    x1, y1 = record['bbox'][0], record['bbox'][1]
                    x2, y2 = record['bbox'][2], record['bbox'][3]
                    background_image = self._dataset.get_background_image(image_id)
                    background_image = self.color_background(background_image, (x1, y1, x2, y2))
                    correct_image = self.construct_image(background_image, render, x1, y1, x2, y2)
                    path = ImagesUtils.save_image(correct_image, self._output_dir,
                                                  '{}_{}_edited'.format(image_id, str(generated_image_index)))
                    if path is not None:
                        self._log(image_id, path, is_correct=1)

                    background_image = self._dataset.get_background_image(image_id)
                    background_image = self.color_background(background_image, (x1, y1, x2, y2))
                    random_image = self.random_place(background_image, render, (x1, y1, x2, y2))
                    path = ImagesUtils.save_image(random_image, self._output_dir,
                                                  '{}_{}_random'.format(image_id, str(generated_image_index)))
                    if path is not None:
                        self._log(image_id, path, is_correct=0)

                    if self._compare_random:
                        temp_id = '{}_{}'.format(image_id, str(generated_image_index))
                        self._generate_comparison(correct_image, random_image, temp_id + '_compare', temp_id)

                    images_count += 1
                except Exception as e:
                    traceback.print_exc()

            i += 1

    def color_background(self, img, bbox):
        if self._cut_background or self._inpaint_cut:
            img.paste((0, 0, 0), box=bbox)

        if self._inpaint_cut:
            mask = np.zeros((img.size[0], img.size[1]))
            mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
            api_result = self._inpaint_api.inpaint(np.array(img), np.array(mask).T)
            img = Image.fromarray(api_result)

        return img

    def _generate_comparison(self, correct_image, random_image, file_name, image_id):
        correct_image_index = random.randint(0, 1)
        if correct_image_index == 0:
            images = [correct_image, random_image]
        else:
            images = [random_image, correct_image]

        couple = ImagesUtils.concat_images(images)
        path = ImagesUtils.save_image(couple, self._compare_dir, file_name)
        if path is not None:
            self._log_comparison(image_id, image_id, path, correct_image_index)

    def construct_image(self, background_image, object_image, x1, y1, x2, y2):
        resized = object_image.resize((x2 - x1, y2 - y1))
        background_image.paste(resized, (x1, y1, x2, y2), resized)
        return np.asarray(background_image)

    def random_place(self, img, render, correct_bbox):
        bbox_width = correct_bbox[2] - correct_bbox[0]
        bbox_height = correct_bbox[3] - correct_bbox[1]

        if (bbox_width / img.size[0]) > (bbox_height / img.size[1]):
            max_resize_factor = img.size[0] / render.size[0]
        else:
            max_resize_factor = img.size[1] / render.size[1]

        max_resize_factor *= 100
        resize_factor = random.randint(int(max_resize_factor / 2), int(max_resize_factor))
        resize_factor /= 100

        region_render = render.resize((int(render.size[0] * resize_factor),
                                       int(render.size[1] * resize_factor)))
        x, y = ImagesUtils.get_random_position(region_render.size[0], region_render.size[1],
                                               img.size[0], img.size[1])
        img.paste(region_render, (x, y), region_render)
        return np.array(img)



