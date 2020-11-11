import os
import random
import traceback

from data_generation.base_generator import BaseGenerator
from datasets.front_future_3d import FrontFuture3D
from utils.images_utils import ImagesUtils


class FrontModelRender(BaseGenerator):
    def __init__(self, dataset):
        super().__init__(dataset)
        assert isinstance(dataset, FrontFuture3D), "Generator support only " + FrontFuture3D.__name__ + " dataset"

        self._output_dir = os.path.join(dataset.front_future_dir, 'generated_model')
        os.makedirs(self._output_dir, exist_ok=True)
        print('Setting {} output directory as {}'.format(self.__class__.__name__, self._output_dir))

        self._metadata = os.path.join(self._output_dir, 'metadata.csv')

    def generate(self, count):
        models_paths = self._dataset.get_model_paths()
        random.shuffle(models_paths)
        images_count = 0
        i = 0

        while i < len(models_paths):
            model_path = models_paths[i]
            path = os.path.join(self._output_dir, '{}_0.png'.format(i))
            if os.path.exists(path):
                i += 1
                continue

            renders = None
            category = None
            try:
                renders, category = self._dataset.render_model(model_path)
            except Exception as e:
                traceback.print_exc()

            if not category or not renders:
                i += 1
                continue

            generated_image_index = 0
            for render in renders:
                try:
                    self.save_render(render, category, i, generated_image_index)
                    generated_image_index += 1
                    images_count += 1

                    if images_count % 5 == 0:
                        print(images_count)

                except Exception as e:
                    traceback.print_exc()

            i += 1

    def save_render(self, render, category, id, generated_image_index):
        path = None
        try:
            path = ImagesUtils.save_image(render, self._output_dir,
                                          '{}_{}'.format(id, str(generated_image_index)))
        except Exception as e:
            print('Error saving file, skipping')

        if path is not None:
            self._log(path, path, is_correct=category)
