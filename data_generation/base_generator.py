import csv
import os
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    def __init__(self, dataset):
        self._dataset = dataset
        self._metadata = None
        self._compare_metadata = None

    @abstractmethod
    def generate(self, count):
        pass

    def _log(self, image_id, path, is_correct=1):
        if not os.path.exists(self._metadata):
            with open(self._metadata, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_id', 'path', 'is_correct'])

        with open(self._metadata, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([image_id, path, str(is_correct)])

    def _log_comparison(self, image_id_1, image_id_2, path, correct_image_index):
        if not os.path.exists(self._compare_metadata):
            with open(self._compare_metadata, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_id_1', 'image_id_2', 'path', 'correct_image_index'])

        with open(self._compare_metadata, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([image_id_1, image_id_2, path, correct_image_index])
