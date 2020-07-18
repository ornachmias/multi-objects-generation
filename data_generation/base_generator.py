from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    def __init__(self, dataset):
        self._dataset = dataset

    @abstractmethod
    def process_image(self, img_id, img, mask):
        pass

    def generate(self, count):
        image_ids = self._dataset.get_image_ids()[:count]
        for image_id in image_ids:
            img, mask = self._dataset.get_image(image_id)
            self.process_image(image_id, img, mask)

