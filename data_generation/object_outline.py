import os

from data_generation.base_generator import BaseGenerator
from utils.images_utils import ImagesUtils
from utils.segmentation_utils import SegmentationUtils


class ObjectOutline(BaseGenerator):
    def __init__(self, root_path, dataset):
        super().__init__(dataset)

        self._outlines_dir = os.path.join(root_path, 'outlines')
        os.makedirs(self._outlines_dir, exist_ok=True)

    def process_image(self, img_id, mask):
        color_map = SegmentationUtils.segmentation_map_to_color_map(
            SegmentationUtils.segmentation_mask_to_map(mask))
        ImagesUtils.save_image(color_map, self._outlines_dir, str(img_id))

    def generate(self, count):
        image_ids = self._dataset.get_image_ids()[:count]
        for image_id in image_ids:
            img, mask, _ = self._dataset.get_image(image_id)
            self.process_image(image_id, mask)




