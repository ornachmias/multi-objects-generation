import os

import numpy as np
import trimesh
from trimesh import Scene
from PIL import Image

from utils.files_utils import FilesUtils


class FrontFuture3D:
    def __init__(self, data_path):
        self._3d_future_model_url = \
            'https://tianchi-media.oss-accelerate.aliyuncs.com/65347_3D-future/3D-FUTURE-model.zip'
        self._3d_front_url = 'https://tianchi-media.oss-accelerate.aliyuncs.com/65347_3D-future/3D-FRONT.zip'

        self.front_future_dir = os.path.join(data_path, '3d_front')
        self._3d_front_dir = os.path.join(self.front_future_dir, '3D-FRONT')
        self._3d_future_model_dir = os.path.join(self.front_future_dir, '3D-FUTURE-model')
        self.obj_custom_dir = os.path.join(self.front_future_dir, 'obj_output')

    def initialize(self, force_init=False):
        os.makedirs(self.front_future_dir, exist_ok=True)

        downloaded_target_path = os.path.join(self.front_future_dir, '3D-FUTURE-model.zip')
        FrontFuture3D._download_and_extract(self._3d_future_model_url, downloaded_target_path,
                                            self._3d_future_model_dir, force_init)

        downloaded_target_path = os.path.join(self.front_future_dir, '3D-FRONT.zip')
        FrontFuture3D._download_and_extract(self._3d_future_model_url, downloaded_target_path,
                                            self._3d_future_model_dir, force_init)

    def get_scene_ids(self):
        return os.listdir(self.obj_custom_dir)

    @staticmethod
    def _download_and_extract(url, download_target_path, extracted_dir, force_init):
        if not os.path.exists(download_target_path) or force_init:
            FilesUtils.download(url, download_target_path)

        if not os.path.exists(extracted_dir) or force_init:
            FilesUtils.validate_path(download_target_path)
            FilesUtils.extract(download_target_path)

    def render_scene(self, scene_id, transform_category, transform_matrix):
        scene_dir = os.path.join(self.obj_custom_dir, scene_id)
        rooms_ids = os.listdir(scene_dir)
        correct_imgs = []
        incorrect_imgs = []
        for rooms_id in rooms_ids:
            room_dir = os.path.join(self.obj_custom_dir, scene_id, rooms_id)
            room_files = [os.path.join(room_dir, f) for f in os.listdir(room_dir) if f.endswith('.obj')]
            if len(room_files) > 1:
                print('Rendering {}'.format(room_dir))
                correct_img, incorrect_img = self.render_room(scene_id, rooms_id, transform_category, transform_matrix)
                correct_imgs.append(np.array(correct_img))
                incorrect_imgs.append(np.array(incorrect_img))

        return correct_imgs, incorrect_imgs

    def render_room(self, scene_id, room_id, transform_category, transform_matrix):
        room_dir = os.path.join(self.obj_custom_dir, scene_id, room_id)
        room_files = [os.path.join(room_dir, f) for f in os.listdir(room_dir) if f.endswith('.obj')]

        scene_trimesh = Scene()
        categories = {}
        for obj_file in room_files:
            with open(obj_file) as f:
                first_line = f.readline()
                if first_line.startswith('# category='):
                    category = first_line.replace('# category=', '')
                    k = os.path.basename(obj_file)
                    categories[k] = category.lower()

            obj_trimesh = trimesh.load(obj_file)
            scene_trimesh.add_geometry(obj_trimesh, node_name=obj_file)

        correct_img = Image.open(trimesh.util.wrap_as_stream(
            scene_trimesh.save_image(resolution=(1000, 1000), background=[255, 255, 255, 0], flags={'cull': True})))

        for geo_identifier in scene_trimesh.geometry:
            if geo_identifier != 'mesh.obj':
                if transform_category in categories[geo_identifier]:
                    scene_trimesh.geometry[geo_identifier].apply_transform(transform_matrix)
                    break

        incorrect_image = Image.open(trimesh.util.wrap_as_stream(
            scene_trimesh.save_image(resolution=(1000, 1000), background=[255, 255, 255, 0], flags={'cull': True})))

        return correct_img, incorrect_image
