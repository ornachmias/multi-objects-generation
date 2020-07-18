import os

from data_generation.base_generator import BaseGenerator
from utils.images_utils import ImagesUtils


class ObjectOutline(BaseGenerator):
    def __init__(self, root_path, dataset):
        super().__init__(dataset)

        self._outlines_dir = os.path.join(root_path, 'outlines')
        os.makedirs(self._outlines_dir, exist_ok=True)

    def process_image(self, img_id, img, mask):
        color_map = ImagesUtils.segmentation_map_to_color_map(
            ImagesUtils.segmentation_mask_to_map(mask))
        ImagesUtils.save_image(color_map, self._outlines_dir, str(img_id))




