import csv
import os
import random
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    def __init__(self, dataset):
        self._dataset = dataset
        self._metadata = None
        self._compare_metadata = None
        self._output_dir = None

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

    def _split_data(self, train=0.7, eval=0.25, test=0.05):
        assert train + eval + test == 1.0, 'Splits does not sum to 1'
        with open(self._metadata, 'r') as metadata_file:
            content = metadata_file.readlines()
            headers = content[0]
            content = content[1:]
            random.shuffle(content)
            total_size = len(content)
            index = 0
            curr_path = os.path.join(self._output_dir, 'metadata_train.csv')
            with open(curr_path, 'w') as file:
                size = int(total_size * train)
                file.write(headers)
                file.writelines(content[:size])
                index += size

            curr_path = os.path.join(self._output_dir, 'metadata_eval.csv')
            with open(curr_path, 'w') as file:
                size = int(total_size * eval)
                file.write(headers)
                file.writelines(content[index:index+size])
                index += size

            curr_path = os.path.join(self._output_dir, 'metadata_test.csv')
            with open(curr_path, 'w') as file:
                file.write(headers)
                file.writelines(content[index:])
