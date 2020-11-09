import os
import random
import traceback

from data_generation.base_generator import BaseGenerator
from datasets.front_future_3d import FrontFuture3D
from datasets.scenes_3d import Scenes3D
from utils.images_utils import ImagesUtils


class Scenes3DRender(BaseGenerator):
    def __init__(self, dataset, transform_categories, transform_matrix, compare_random=False):
        super().__init__(dataset)
        assert isinstance(dataset, Scenes3D), "Generator support only " + Scenes3D.__name__ + " dataset"

        self._output_dir = os.path.join(dataset.scenes_3d_dir, 'generated')
        os.makedirs(self._output_dir, exist_ok=True)
        print('Setting {} output directory as {}'.format(self.__class__.__name__, self._output_dir))

        self._transform_categories = transform_categories
        self._transform_matrix = transform_matrix
        self._metadata = os.path.join(self._output_dir, 'metadata.csv')
        self._compare_random = compare_random
        if self._compare_random:
            self._compare_dir = os.path.join(self._output_dir, 'compare')
            os.makedirs(self._compare_dir, exist_ok=True)
            self._compare_metadata = os.path.join(self._compare_dir, 'metadata.csv')

    def generate(self, count):
        scene_ids = self._dataset.get_scene_ids()
        random.shuffle(scene_ids)
        images_count = 0
        i = 0

        while images_count < count:
            scene_id = scene_ids[i]
            path1 = os.path.join(self._output_dir, '{}_0_edited.png'.format(scene_id))
            path2 = os.path.join(self._output_dir, '{}_0_random.png'.format(scene_id))
            if os.path.exists(path1) or os.path.exists(path2):
                i += 1
                continue

            correct_renders = None
            incorrect_renders = None
            try:
                correct_renders, incorrect_renders = \
                    self._dataset.compose_layout(scene_id, self._transform_categories, self._transform_matrix)
            except Exception as e:
                traceback.print_exc()

            if not correct_renders or not incorrect_renders:
                i += 1
                continue

            generated_image_index = 0
            for correct_render, incorrect_render in zip(correct_renders, incorrect_renders):
                try:
                    self.generate_renders(scene_id, generated_image_index, correct_render, incorrect_render)
                    generated_image_index += 1
                    images_count += 1

                    if images_count % 5 == 0:
                        print(images_count)

                except Exception as e:
                    traceback.print_exc()

            i += 1

    def generate_renders(self, scene_id, generated_image_index, correct_render, incorrect_render):
        path = ImagesUtils.save_image(correct_render, self._output_dir,
                                      '{}_{}_edited'.format(scene_id, str(generated_image_index)))

        if path is not None:
            self._log(scene_id, path, is_correct=1)

        path = ImagesUtils.save_image(incorrect_render, self._output_dir,
                                      '{}_{}_random'.format(scene_id, str(generated_image_index)))

        if path is not None:
            self._log(scene_id, path, is_correct=0)

        if self._compare_random:
            temp_id = '{}_{}'.format(scene_id, str(generated_image_index))
            self._generate_comparison(correct_render, incorrect_render, temp_id + '_compare', temp_id)
