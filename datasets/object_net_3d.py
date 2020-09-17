import os

from PIL import Image
import scipy.io as sio

from utils.files_utils import FilesUtils
from utils.obj_render import ObjRender


class ObjectNet3D:
    def __init__(self, data_path):
        self._annotation_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip'
        self._images_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip'
        self._cad_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip'
        self._metadata_url = 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip'
        self._object_3d_net_dir = os.path.join(data_path, 'object_3d_net')
        self._annotation_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'Annotations')
        self._cad_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'CAD')
        self._images_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'Images')
        self._metadata_dir = os.path.join(self._object_3d_net_dir, 'ObjectNet3D', 'Image_sets')
        self._shapenet_dir = os.path.join(data_path, 'shape_net', 'ShapeNetCore.v1')

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

    def get_image_ids(self):
        train_path = os.path.join(self._metadata_dir, 'train.txt')
        eval_path = os.path.join(self._metadata_dir, 'val.txt')
        test_path = os.path.join(self._metadata_dir, 'test.txt')
        return FilesUtils.read_file(train_path), FilesUtils.read_file(eval_path), FilesUtils.read_file(test_path)

    def get_image(self, image_id):
        annotation_path = os.path.join(self._annotation_dir, image_id + '.mat')
        records, img_file_name, img_size = ObjectNet3D._parse_matrix(sio.loadmat(annotation_path))

        image_path = os.path.join(self._images_dir, img_file_name)
        img = Image.open(image_path)

        return img, records, img_size

    def get_background_image(self, image_id):
        annotation_path = os.path.join(self._annotation_dir, image_id + '.mat')
        records, img_file_name, img_size = ObjectNet3D._parse_matrix(sio.loadmat(annotation_path))

        image_path = os.path.join(self._images_dir, img_file_name)
        return Image.open(image_path)

    def get_renders(self, image_id):
        result = []
        _, records, _ = self.get_image(image_id)
        for record in records:
            if 'shapenet_dir' not in record:
                continue
            else:
                path = os.path.join(self._shapenet_dir, record['shapenet_dir'], record['shapenet_sub_dir'], 'model.obj')

            obj_render = ObjRender(path, record)
            rendered = obj_render.render()
            result.append((record, rendered))

        return result

    def show_image(self, image_id):
        img, records, img_size = self.get_image(image_id)
        for record in records:
            if 'shapenet_dir' not in record:
                continue
            else:
                path = os.path.join(self._shapenet_dir, record['shapenet_dir'], record['shapenet_sub_dir'], 'model.obj')

            obj_render = ObjRender(path, record)
            rendered = obj_render.render()
            self.construct_image(img, rendered, record)

        img.show()

    def construct_image(self, background_image, object_image, record):
        x1 = record['bbox'][0]
        y1 = record['bbox'][1]
        x2 = record['bbox'][2]
        y2 = record['bbox'][3]
        resized = object_image.resize((x2-x1, y2-y1))
        background_image.paste(resized, (x1, y1, x2, y2), resized)
        return background_image

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
        img_size = record['imgsize'][0]
        img_file_name = mat['record']['filename'][0][0][0]

        objects = record['objects'][0]
        for i in range(objects.shape[0]):
            curr_dic = {}
            if objects['shapenet'][i]:
                curr_dic['shapenet_dir'] = objects['shapenet'][i]['synset'][0][0][0]
                curr_dic['shapenet_sub_dir'] = objects['shapenet'][i]['selected'][0][0][0][0][0]
            curr_dic['object_cls'] = objects[i]['class'].item()
            curr_dic['truncated'] = str(objects[i]['truncated'].item())
            curr_dic['occluded'] = str(objects[i]['occluded'].item())
            curr_dic['difficult'] = str(objects[i]['difficult'].item())
            curr_dic['cad_index'] = objects[i]['cad_index'].item()
            curr_dic['viewpoint'] = objects[i]['viewpoint'][0][0]
            curr_dic['px'] = curr_dic['viewpoint']['px'][0][0]
            curr_dic['py'] = curr_dic['viewpoint']['py'][0][0]
            curr_dic['img_size'] = img_size
            try:
                curr_dic['azimuth'] = str(curr_dic['viewpoint']['azimuth'].item())
                curr_dic['elevation'] = str(curr_dic['viewpoint']['elevation'].item())
            except:
                curr_dic['azimuth'] = str(curr_dic['viewpoint']['azimuth_coarse'].item())
                curr_dic['elevation'] = str(curr_dic['viewpoint']['elevation_coarse'].item())
            curr_dic['azimuth'] = int(float(curr_dic['azimuth']))
            curr_dic['elevation'] = int(float(curr_dic['elevation']))
            curr_dic['distance'] = float(str(curr_dic['viewpoint']['distance'].item()))
            curr_dic['inplane_rotation'] = float(str(curr_dic['viewpoint']['theta'].item()))
            curr_dic['bbox'] = tuple((objects[i]['bbox'][0]).astype('int'))
            curr_dic['viewport'] = curr_dic['viewpoint']['viewport'][0][0]
            curr_dic['focal'] = curr_dic['viewpoint']['focal'][0][0]
            result.append(curr_dic)

        return result, img_file_name, img_size
