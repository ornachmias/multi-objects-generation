import os

from utils.files_utils import FilesUtils
from utils.images_utils import ImagesUtils


class TestDataset:
    def __init__(self, data_path):
        self._download_url = 'Dummy url, real one requires authentication token.'
        self.dataset_dir = os.path.join(data_path, 'dogs_vs_cats')
        self._train_dir = os.path.join(self.dataset_dir, 'train')
        self._test_dir = os.path.join(self.dataset_dir, 'test1')

    def initialize(self, force_init=False):
        os.makedirs(self.dataset_dir, exist_ok=True)

        downloaded_target_path = os.path.join(self.dataset_dir, 'dogs-vs-cats.zip')
        if not os.path.exists(downloaded_target_path) or force_init:
            FilesUtils.download(self._download_url, downloaded_target_path)

        if not os.path.exists(self.dataset_dir) or force_init:
            FilesUtils.extract(downloaded_target_path)

        if not os.path.exists(self._train_dir):
            FilesUtils.extract(os.path.join(self.dataset_dir, 'train.zip'))

        if not os.path.exists(self._test_dir):
            FilesUtils.extract(os.path.join(self.dataset_dir, 'test1.zip'))

    def get_image(self, image_id):
        image_path = os.path.join(self._train_dir, image_id)
        img = ImagesUtils.convert_to_numpy(image_path)
        return img, 'dog' in image_id

    def get_image_ids(self):
        return os.listdir(self._train_dir)

    def get_full_path(self, image_id):
        return os.path.join(self._train_dir, image_id)
