import json
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
        self._camera_json = os.path.join(self.front_future_dir, 'camera_json')

    def initialize(self, force_init=False):
        os.makedirs(self.front_future_dir, exist_ok=True)

        downloaded_target_path = os.path.join(self.front_future_dir, '3D-FUTURE-model.zip')
        FrontFuture3D._download_and_extract(self._3d_future_model_url, downloaded_target_path,
                                            self._3d_future_model_dir, force_init)

        downloaded_target_path = os.path.join(self.front_future_dir, '3D-FRONT.zip')
        FrontFuture3D._download_and_extract(self._3d_future_model_url, downloaded_target_path,
                                            self._3d_future_model_dir, force_init)

        if not os.path.exists(self._camera_json):
            raise Exception('Path {} does not exists, please run camera parameters extraction.'.
                            format(self._camera_json))

    def get_scene_ids(self):
        return os.listdir(self.obj_custom_dir)

    def get_model_paths(self):
        model_ids = {}
        scene_paths = [os.path.join(self.obj_custom_dir, d) for d in os.listdir(self.obj_custom_dir)
                       if os.path.isdir(os.path.join(self.obj_custom_dir, d))]

        for scene_path in scene_paths:
            room_dirs = [os.path.join(scene_path, d) for d in os.listdir(scene_path)
                         if os.path.isdir(os.path.join(scene_path, d))]

            for room_dir in room_dirs:
                model_paths = [os.path.join(room_dir, f) for f in os.listdir(room_dir)
                               if f != 'mesh.obj' and f.endswith('.obj')]

                for model_path in model_paths:
                    model_ids[model_path] = 1

        return list(model_ids.keys())

    def render_model(self, model_path):
        category = None
        with open(model_path) as f:
            first_line = f.readline()
            if first_line.startswith('# category='):
                category = first_line.replace('# category=', '').strip()

        result = []
        for x in range(0, 360, 120):
            for y in range(0, 360, 120):
                for z in range(0, 360, 120):
                    scene = Scene()
                    obj = trimesh.load(model_path)
                    obj.apply_transform((self.x_rotation(np.deg2rad(x))))
                    obj.apply_transform((self.y_rotation(np.deg2rad(y))))
                    obj.apply_transform((self.z_rotation(np.deg2rad(z))))
                    scene.add_geometry(obj)
                    result.append(np.array(Image.open(trimesh.util.wrap_as_stream(
                        scene.save_image(background=[255, 255, 255, 0], flags={'cull': True})))))

        return result, category

    def z_rotation(self, angle):
        rot = np.eye(4)
        rot[0, 0] = np.cos(angle)
        rot[1, 0] = np.sin(angle)
        rot[0, 1] = -np.sin(angle)
        rot[1, 1] = np.cos(angle)
        return rot

    def y_rotation(self, angle):
        rot = np.eye(4)
        rot[0, 0] = np.cos(angle)
        rot[0, 2] = np.sin(angle)
        rot[2, 0] = -np.sin(angle)
        rot[2, 2] = np.cos(angle)
        return rot

    def x_rotation(self, angle):
        rot = np.eye(4)
        rot[1, 1] = np.cos(angle)
        rot[1, 2] = -np.sin(angle)
        rot[2, 1] = np.sin(angle)
        rot[2, 2] = np.cos(angle)
        return rot

    @staticmethod
    def _download_and_extract(url, download_target_path, extracted_dir, force_init):
        if not os.path.exists(download_target_path) or force_init:
            FilesUtils.download(url, download_target_path)

        if not os.path.exists(extracted_dir) or force_init:
            FilesUtils.validate_path(download_target_path)
            FilesUtils.extract(download_target_path)

    def compose_layout(self, layout_id, transform_categories, transform_matrix):
        correct_imgs = []
        incorrect_imgs = []
        layout_dir = os.path.join(self.obj_custom_dir, layout_id)
        rooms_ids = os.listdir(layout_dir)
        camera_json_path = os.path.join(self._camera_json, layout_id + '.camera.json')

        layout_files = []
        for rooms_id in rooms_ids:
            room_dir = os.path.join(self.obj_custom_dir, layout_id, rooms_id)
            room_files = [os.path.join(room_dir, f) for f in os.listdir(room_dir) if f.endswith('.obj')]
            if len(room_files) > 1:
                layout_files.extend(room_files)

        scene_trimesh = Scene()
        categories = {}
        for obj_file in layout_files:
            with open(obj_file) as f:
                first_line = f.readline()
                if first_line.startswith('# category='):
                    category = first_line.replace('# category=', '')
                    k = os.path.basename(obj_file)
                    categories[k] = category.lower()

            obj_trimesh = trimesh.load(obj_file)
            scene_trimesh.add_geometry(obj_trimesh, node_name=obj_file)

        with open(camera_json_path) as camera_json_file:
            camera_json = json.load(camera_json_file)
            for camera_params in camera_json:
                correct_img, incorrect_img = self.render_view(scene_trimesh, transform_categories, transform_matrix,
                                                              camera_params, categories)
                correct_imgs.append(correct_img)
                incorrect_imgs.append(incorrect_img)

        return correct_imgs, incorrect_imgs

    def render_view(self, scene_trimesh, transform_categories, transform_matrix, camera_params, categories):
        temp_scene = scene_trimesh.copy()
        pos = np.array(camera_params['pos'])
        target = np.array(camera_params['target'])
        fov = camera_params['fov']
        temp_scene.camera.fov = [fov, fov]
        squared_dist = np.sum((pos - target) ** 2, axis=0)
        dist = np.sqrt(squared_dist)
        trans = temp_scene.camera.look_at(points=[target], center=pos, distance=dist * 1.2)
        temp_scene.camera_transform = trans
        correct_img = Image.open(trimesh.util.wrap_as_stream(
            temp_scene.save_image(resolution=(1000, 1000), background=[255, 255, 255, 0], flags={'cull': True})))

        for name in categories:
            for geo_identifier in temp_scene.geometry:
                category = categories[name].lower()
                for transform_category in transform_categories:
                    if transform_category in category and name in geo_identifier:
                        temp_scene.geometry[geo_identifier].apply_transform(transform_matrix)

        incorrect_image = Image.open(trimesh.util.wrap_as_stream(
            temp_scene.save_image(resolution=(1000, 1000), background=[255, 255, 255, 0], flags={'cull': True})))

        return np.array(correct_img), np.array(incorrect_image)

    def look_at(self, center, target):
        up = np.array([0.0, 1.0, 0.0])
        f = (target - center)
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        u = u / np.linalg.norm(u)

        m = np.zeros((4, 4))
        m[0, :-1] = s
        m[1, :-1] = u
        m[2, :-1] = -f
        m[-1, -1] = 1.0
        r = m[:3, :3]
        t = np.matmul(r, center)
        m[:3, 3] = t.reshape((3,))
        return m
