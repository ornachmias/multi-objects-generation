import os
import json
import trimesh
import numpy as np
from PIL import Image

from trimesh import Scene


class Scenes3D:
    def __init__(self, data_dir):
        self.scenes_3d_dir = os.path.join(data_dir, 'scenes_3d')
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

    def get_scene_ids(self):
        return list(self._scenes.keys())

    def compose_layout(self, scene_id, transform_categories, transform_matrix):
        correct_scene, incorrect_scene = self.compose_scene(scene_id, transform_categories, transform_matrix)
        correct_image = Image.open(trimesh.util.wrap_as_stream(
            correct_scene.save_image(resolution=(1000, 1000), background=[255, 255, 255, 0], flags={'cull': True})))
        incorrect_image = Image.open(trimesh.util.wrap_as_stream(
            incorrect_scene.save_image(resolution=(1000, 1000), background=[255, 255, 255, 0], flags={'cull': True})))

        return [np.array(correct_image)], [np.array(incorrect_image)]

    def compose_scene(self, scene_id, transform_categories, transform_matrix):
        scene_json = self._scenes[scene_id]
        metadata_jsons = scene_json['objects']

        correct_scene = Scene()

        for metadata_json in metadata_jsons:
            correct_scene = self.set_mesh(correct_scene, metadata_json)

        self.set_camera_transform(correct_scene)

        incorrect_scene = Scene()

        for metadata_json in metadata_jsons:
            incorrect_scene = self.set_mesh(incorrect_scene, metadata_json, transform_categories, transform_matrix)

        self.set_camera_transform(incorrect_scene)

        return correct_scene, incorrect_scene

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
