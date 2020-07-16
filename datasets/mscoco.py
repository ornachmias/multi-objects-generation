import os
import pickle

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import matplotlib

from utils.files_utils import FilesUtils


class Mscoco:
    def __init__(self, root_path):
        self._mscoco_file = 'annotations_trainval2017.zip'
        self._download_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        self._mscoco_dir = os.path.join(root_path, 'mscoco')
        self._annotation_dir = os.path.join(self._mscoco_dir, 'annotations')
        self._annotation_file = os.path.join(self._annotation_dir, 'instances_train2017.json')
        self._mscoco_api_path = os.path.join(self._mscoco_dir, 'mscoco_api.pkl')
        self._mscoco_api = None

    def initialize(self, force_init=False):
        os.makedirs(self._mscoco_dir, exist_ok=True)
        downloaded_target_path = os.path.join(self._mscoco_dir, self._mscoco_file)

        if not os.path.exists(downloaded_target_path) or force_init:
            FilesUtils.download(self._download_url, downloaded_target_path)

        if not os.path.exists(self._annotation_dir) or force_init:
            FilesUtils.validate_path(downloaded_target_path)
            FilesUtils.extract(downloaded_target_path)

        FilesUtils.validate_path(self._annotation_dir)
        FilesUtils.validate_path(self._annotation_file)
        self._mscoco_api = self.__load_mscoco_api()

    def get_image(self, image_id, category_id, get_segmentation_mask=True):
        img_object = self._mscoco_api.loadImgs(image_id)[0]
        img = io.imread(img_object['coco_url'])

        if get_segmentation_mask:
            annotation_id = self._mscoco_api.getAnnIds(imgIds=img_object['id'], catIds=[category_id], iscrowd=None)
            annotation = self._mscoco_api.loadAnns(annotation_id)
            mask = self._mscoco_api.annToMask(annotation[0])
            return img, mask

        return img

    def get_categories(self):
        categories = self._mscoco_api.loadCats(self._mscoco_api.getCatIds())
        return [(c['id'], c['name']) for c in categories]

    def get_random_image_id(self):
        img_ids = self._mscoco_api.getImgIds()
        return img_ids[np.random.randint(0, len(img_ids))]

    def get_image_ids(self, category_id):
        img_ids = self._mscoco_api.getImgIds(catIds=[category_id])
        return img_ids

    def display_image(self, image_id, show_annotation=False):
        # Allow interactive mode
        matplotlib.use('TkAgg')

        img = self._mscoco_api.loadImgs(image_id)[0]
        i = io.imread(img['coco_url'])
        plt.axis('off')
        plt.imshow(i)

        if show_annotation:
            annotation_id = self._mscoco_api.getAnnIds(imgIds=img['id'], iscrowd=None)
            annotation = self._mscoco_api.loadAnns(annotation_id)
            self._mscoco_api.showAnns(annotation)

        plt.show()

    def __load_mscoco_api(self):
        if os.path.exists(self._mscoco_api_path):
            return pickle.load(open(self._mscoco_api_path, 'rb'))
        else:
            coco = COCO(self._annotation_file)
            pickle.dump(coco, open(self._mscoco_api_path, 'wb'))
            return coco


