import os
import random

from data_generation.base_generator import BaseGenerator
from datasets.test_dataset import TestDataset


class TestDataGenerator(BaseGenerator):
    def __init__(self, dataset):
        super().__init__(dataset)
        assert isinstance(dataset, TestDataset), "Generator support only " + TestDataset.__name__ + " dataset"

        self._metadata = os.path.join(dataset.dataset_dir, 'metadata.csv')
        self._output_dir = dataset.dataset_dir

    def generate(self, count):
        image_ids = self._dataset.get_image_ids()
        random.shuffle(image_ids)
        image_ids = image_ids[:count]

        for image_id in image_ids:
            _, is_correct = self._dataset.get_image(image_id)
            self._log(image_id, self._dataset.get_full_path(image_id), int(is_correct))

        self._split_data()
