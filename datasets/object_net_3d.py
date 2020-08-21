import os

from PIL import Image
import numpy as np
import scipy.io as sio

from utils.files_utils import FilesUtils


class ObjectNet3D:
    def __init__(self, data_path, categories=[]):
        self._annotation_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip'
        self._images_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip'
        self._cad_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip'
        self._metadata_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip'
        self._categories = categories
        self._object_3d_net_dir = os.path.join(data_path, 'object_3d_net')
        self._annotation_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'Annotations')
        self._cad_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'CAD')
        self._images_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'Images')
        self._metadata_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'Image_sets')

    def initialize(self, force_init=False):
        os.makedirs(self._object_3d_net_dir, exist_ok=True)

        downloaded_target_path = os.path.join(self._object_3d_net_dir, 'ObjectNet3D_annotations.zip')
        ObjectNet3D._download_and_extract(self._annotation_url, downloaded_target_path, self._annotation_dir,
                                          force_init)

        downloaded_target_path = os.path.join(self._object_3d_net_dir, 'ObjectNet3D_images.zip')
        ObjectNet3D._download_and_extract(self._images_url, downloaded_target_path, self._images_dir,
                                          force_init)

        downloaded_target_path = os.path.join(self._object_3d_net_dir, 'ObjectNet3D_cads.zip')
        ObjectNet3D._download_and_extract(self._cad_url, downloaded_target_path, self._cad_dir,
                                          force_init)

        downloaded_target_path = os.path.join(self._object_3d_net_dir, 'ObjectNet3D_cads.zip')
        ObjectNet3D._download_and_extract(self._cad_url, downloaded_target_path, self._cad_dir,
                                          force_init)

        downloaded_target_path = os.path.join(self._object_3d_net_dir, 'ObjectNet3D_image_sets.zip')
        ObjectNet3D._download_and_extract(self._metadata_url, downloaded_target_path, self._metadata_dir,
                                          force_init)

    def get_categories(self):
        categories_path = os.path.join(self._metadata_dir, 'classes.txt')
        with open(categories_path) as f:
            return f.read().splitlines()

    def get_image(self, image_id, category_ids=None):
        annotation_path = os.path.join(self._annotation_dir, image_id + '.mat')
        record = ObjectNet3D._parse_matrix(sio.loadmat(annotation_path))

        image_path = os.path.join(self._images_dir, image_id + '.JPEG')
        img = Image.open(image_path)
        data = np.asarray(img, dtype=np.uint8)

        return data, record

    def show_image(self, image_id):
        pass


    @staticmethod
    def _download_and_extract(url, download_target_path, extracted_dir, force_init):
        if not os.path.exists(download_target_path) or force_init:
            FilesUtils.download(url, download_target_path)

        if not os.path.exists(extracted_dir) or force_init:
            FilesUtils.validate_path(download_target_path)
            FilesUtils.extract(download_target_path)

    @staticmethod
    def _parse_matrix(mat):
        result = []
        record = mat['record'][0][0]
        objects = record['objects'][0]
        for i in range(objects.shape[0]):
            curr_dic = {}
            curr_dic['object_cls'] = objects[i]['class'].item()
            curr_dic['truncated'] = str(objects[i]['truncated'].item())
            curr_dic['occluded'] = str(objects[i]['occluded'].item())
            curr_dic['difficult'] = str(objects[i]['difficult'].item())
            curr_dic['cad_index'] = objects[i]['cad_index'].item()
            curr_dic['viewpoint'] = objects[i]['viewpoint'][0][0]
            try:
                curr_dic['azimuth'] = str(curr_dic['viewpoint']['azimuth'].item())
                curr_dic['elevation'] = str(curr_dic['viewpoint']['elevation'].item())
            except:
                curr_dic['azimuth'] = str(curr_dic['viewpoint']['azimuth_coarse'].item())
                curr_dic['elevation'] = str(curr_dic['viewpoint']['elevation_coarse'].item())
            curr_dic['distance'] = str(curr_dic['viewpoint']['distance'].item())
            curr_dic['inplane_rotation'] = str(curr_dic['viewpoint']['theta'].item())
            curr_dic['bbox'] = tuple((objects[i]['bbox'][0]).astype('int'))
            result.append(curr_dic)

        return result


dataset = ObjectNet3D('../data')
dataset.initialize()
dataset.show_image('n00000001_947')
