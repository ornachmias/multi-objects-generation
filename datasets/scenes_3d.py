import os
import json
import trimesh
import numpy as np
from PIL import Image

from trimesh import Scene

from utils.files_utils import FilesUtils


class Scenes3D:
    def __init__(self, data_dir):
        self.scenes_3d_dir = os.path.join(data_dir, 'scenes_3d')

        self._models_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/wss.models.zip'
        self._textures_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/wss.texture.zip'
        self._categories_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/model_categories.tsv'
        self._scenes_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/scenes.csv'
        self._models_zip_path = os.path.join(self.scenes_3d_dir, 'wss.models.zip')
        self._textures_zip_path = os.path.join(self.scenes_3d_dir, 'wss.texture.zip')
        self._models_dir = os.path.join(self.scenes_3d_dir, 'models')
        self._textures_dir = os.path.join(self.scenes_3d_dir, 'textures')
        self._model_scale_path = os.path.join(self.scenes_3d_dir, 'model_scales.csv')
        self._scenes_path = os.path.join(self.scenes_3d_dir, 'scenes.csv')
        self._categories_path = os.path.join(self.scenes_3d_dir, 'model_categories.tsv')
        self._scenes = {}
        self._categories = {}

    def initialize(self):
        with open(self._scenes_path) as f:
            scenes_lines = f.readlines()

        for line in scenes_lines[1:]:
            separator_index = line.find(',')
            scene_name = line[:separator_index]
            scene_data = line[separator_index + 1:]
            self._scenes[scene_name] = json.loads(scene_data[1:-2])

        with open(self._categories_path) as f:
            categories_lines = f.readlines()

        for line in categories_lines:
            split_line = line.split()
            model_id = split_line[0].replace('wss.', '')
            category = split_line[1].lower()
            self._categories[model_id] = category

    def download_and_extract(self, force_init):
        if not os.path.exists(self._models_zip_path) or force_init:
            FilesUtils.download(self._models_url, self._models_zip_path)

        if not os.path.exists(self._models_dir) or force_init:
            FilesUtils.validate_path(self._models_zip_path)
            FilesUtils.extract(self._models_zip_path)

        if not os.path.exists(self._textures_zip_path) or force_init:
            FilesUtils.download(self._textures_url, self._textures_zip_path)

        if not os.path.exists(self._textures_dir) or force_init:
            FilesUtils.validate_path(self._textures_zip_path)
            FilesUtils.extract(self._textures_zip_path)

        if not os.path.exists(self._categories_path) or force_init:
            FilesUtils.download(self._categories_url, self._categories_path)

        if not os.path.exists(self._scenes_path) or force_init:
            FilesUtils.download(self._scenes_url, self._scenes_path)

    def get_scene_ids(self):
        return list(self._scenes.keys())

    def compose_layout(self, scene_id, transform_categories, transform_matrix):
        correct_scenes, incorrect_scenes = self.compose_scene(scene_id, transform_categories, transform_matrix)
        correct_images = []
        incorrect_images = []
        for correct_scene in correct_scenes:
            correct_image = Image.open(trimesh.util.wrap_as_stream(
                correct_scene.save_image(background=[255, 255, 255, 0], flags={'cull': True})))
            correct_images.append(np.array(correct_image))

        for incorrect_scene in incorrect_scenes:
            incorrect_image = Image.open(trimesh.util.wrap_as_stream(
                incorrect_scene.save_image(background=[255, 255, 255, 0], flags={'cull': True})))
            incorrect_images.append(np.array(incorrect_image))

        return correct_images, incorrect_images

    def compose_scene(self, scene_id, transform_categories, transform_matrix):
        correct_scenes = []
        incorrect_scenes = []
        scene_json = self._scenes[scene_id]
        metadata_jsons = scene_json['objects']
        camera_transforms = self.get_camera_transforms()

        correct_scene = Scene()

        for metadata_json in metadata_jsons:
            correct_scene = self.set_mesh(correct_scene, metadata_json)

        for camera_transform in camera_transforms:
            tmp_scene = correct_scene.copy()
            tmp_scene.camera_transform = camera_transform
            correct_scenes.append(tmp_scene)

        incorrect_scene = Scene()

        for metadata_json in metadata_jsons:
            incorrect_scene = self.set_mesh(incorrect_scene, metadata_json, transform_categories, transform_matrix)

        for camera_transform in camera_transforms:
            tmp_scene = incorrect_scene.copy()
            tmp_scene.camera_transform = camera_transform
            incorrect_scenes.append(tmp_scene)

        return correct_scenes, incorrect_scenes

    def set_mesh(self, scene, metadata_json, transform_categories=None, transform_matrix=None):
        id = metadata_json['modelID']
        if id is None or id == '':
            return None

        obj_path = os.path.join(self._models_dir, id + '.obj')
        if not os.path.exists(obj_path):
            return None

        obj_trimesh = trimesh.load(obj_path)
        transformation_matrix = np.reshape(metadata_json['transform'], (4, 4)).T
        obj_trimesh.apply_transform(transformation_matrix)
        if transform_categories is not None \
                and transform_matrix is not None \
                and self._categories[id] in transform_categories:
            obj_trimesh.apply_transform(transform_matrix)

        scene.add_geometry(obj_trimesh)
        return scene

    def set_camera_transform(self, scene):
        camera_transform = np.zeros((4, 4))
        camera_transform[0, 0] = 1.
        camera_transform[0, 3] = 123.09577675
        camera_transform[1, 1] = 0.46565674
        camera_transform[1, 2] = -0.88496542
        camera_transform[1, 3] = -121.93568384
        camera_transform[2, 1] = 0.88496542
        camera_transform[2, 2] = 0.46565674
        camera_transform[2, 3] = 134.79709113
        camera_transform[3, 3] = 1.
        scene.camera_transform = camera_transform
        return scene

    def get_camera_transforms(self):
        return [np.array([[1., 0., 0., 123.09577675],
                          [0., 0.46565674, -0.88496542, -121.93568384],
                          [0., 0.88496542, 0.46565674, 134.79709113],
                          [0., 0., 0., 1.]]),
                np.array([[1., 0., 0., 123.09577675],
                          [0., 0.16266292, -0.9866817, -130.73452539],
                          [0., 0.9866817, 0.16266292, 54.55492985],
                          [0., 0., 0., 1.]]),
                np.array([[7.54404832e-01, -2.61137239e-18, 6.56409435e-01, 2.27399143e+02],
                          [6.47667179e-01, 1.62662921e-01, -7.44357439e-01, -4.91768017e+00],
                          [-1.06773477e-01, 9.86681696e-01, 1.22713694e-01, 3.38129463e+01],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[9.32193350e-01, -2.59279804e-18, -3.61960713e-01, 7.68570236e+01],
                          [-3.57140012e-01, 1.62662921e-01, -9.19778115e-01, -5.05534274e+00],
                          [5.88775871e-02, 9.86681696e-01, 1.51633293e-01, 3.38356411e+01],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[9.24967757e-01, 1.20992630e-01, -3.60271329e-01, 3.64320925e+01],
                          [-3.69435302e-01, 5.08689110e-01, -7.77658628e-01, -7.92278879e+01],
                          [8.91751397e-02, 8.52406104e-01, 5.15219974e-01, 1.36236310e+02],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[9.34644506e-01, 6.94584791e-02, 3.48733661e-01, 2.10643890e+02],
                          [1.93729161e-01, 7.22931082e-01, -6.63204085e-01, -6.29575772e+01],
                          [-2.98175551e-01, 6.87419934e-01, 6.62227431e-01, 1.79234636e+02],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[9.38109647e-01, -9.10961088e-03, 3.46218578e-01, 2.14951660e+02],
                          [1.88904348e-01, 8.51322952e-01, -4.89453137e-01, -2.00656108e+00],
                          [-2.90285096e-01, 5.24562905e-01, 8.00355119e-01, 2.36814343e+02],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[9.93263427e-01, 4.74376262e-02, -1.05723397e-01, 8.03862230e+01],
                          [-1.04402391e-01, 7.62213889e-01, -6.38850626e-01, -1.00964112e+02],
                          [5.02782846e-02, 6.45584738e-01, 7.62031781e-01, 2.10377818e+02],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[9.99850636e-01, -2.01283961e-19, -1.72830900e-02, 1.18830589e+02],
                          [-1.36415702e-03, 9.96880145e-01, -7.89183677e-02, 4.83890764e+01],
                          [1.72291694e-02, 7.89301570e-02, 9.96731248e-01, 3.09244860e+02],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[9.76281662e-01, -3.05604492e-03, -2.16482739e-01, 6.63181894e+01],
                          [-1.12491254e-01, 8.47176711e-01, -5.19266148e-01, -6.66881498e+01],
                          [1.84986036e-01, 5.31302433e-01, 8.26739309e-01, 2.35591590e+02],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])]
