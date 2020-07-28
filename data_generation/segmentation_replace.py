from data_generation.bounding_box_replace import BoundingBoxReplace
from utils.images_utils import ImagesUtils
from utils.segmentation_utils import SegmentationUtils


class SegmentationReplace(BoundingBoxReplace):
    def __init__(self, root_path, dataset):
        super().__init__(root_path, dataset, 'seg_replace')

    def _generate_images(self, category_dir, category_id, image_id_1, image_id_2):
        image_1, seg_1, _ = self._dataset.get_image(image_id_1, [category_id])
        image_2, seg_2, _ = self._dataset.get_image(image_id_2, [category_id])
        edited_image_1, edited_image_2 = \
            SegmentationUtils.replace_content_segmentation(image_1, seg_1[:, :, 0], image_2, seg_2[:, :, 0])
        ImagesUtils.save_image(edited_image_1, category_dir, str(image_id_1))
        ImagesUtils.save_image(edited_image_2, category_dir, str(image_id_2))

