import json
import os
import pickle

import numpy as np
import trimesh
from trimesh import Scene
from tqdm import tqdm
import trimesh.visual
from PIL import Image, ImageDraw
from trimesh.visual.material import SimpleMaterial
from random import randrange

from datasets.scenes_3d import Scenes3D
from utils.images_utils import ImagesUtils


class Scenes3DBboxRenderRandom:
    def __init__(self, data_dir):
        self.output_dir = os.path.join(data_dir, 'generated', 'scenes_3d_bbox_render_random')
        self.metadata_output_dir = os.path.join(self.output_dir, 'metadata')
        self.images_output_dir = os.path.join(self.output_dir, 'images')
        self.dataset = Scenes3D(data_dir)
        self.transform_categories = ['chair']
        self.output_resolution = (640, 480)

    def initialize(self):
        self.dataset.initialize()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_output_dir, exist_ok=True)
        os.makedirs(self.images_output_dir, exist_ok=True)

    def generate(self):
        camera_transforms = self.get_camera_transforms()
        camera_transforms_path = os.path.join(self.metadata_output_dir, 'camera_transforms.pkl')
        pickle.dump(camera_transforms, open(camera_transforms_path, 'wb'))

        scene_ids = self.dataset.get_scene_ids()
        for scene_id in tqdm(scene_ids):
            output_metadata_path = os.path.join(self.metadata_output_dir, scene_id + '.json')
            if os.path.exists(output_metadata_path):
                continue

            model_ids = self.get_scene_model_ids(scene_id)
            output_metadata = {
                'scene_id': scene_id,
                'model_ids': model_ids,
                'generated_scenes': []
            }

            if not self.is_scene_valid(model_ids):
                continue

            for camera_transform_index, camera_transform in enumerate(camera_transforms):
                transform_matrices = self.get_random_transform_matrices(model_ids)

                scene_metadata = {
                    'camera_transform_index': camera_transform_index,
                    'plausible':
                        self.generate_single_view_scene(scene_id, camera_transform, camera_transform_index,
                                                        False, transform_matrices),
                    'implausible':
                        self.generate_single_view_scene(scene_id, camera_transform, camera_transform_index,
                                                        True, transform_matrices)
                }

                output_metadata['generated_scenes'].append(scene_metadata)

            with open(output_metadata_path, 'w') as fp:
                json.dump(output_metadata, fp)

    def get_random_transform_matrices(self, model_ids):
        num_objects = len([i for i in model_ids if self.dataset.get_object_category(i) in self.transform_categories])
        matrices = []
        for i in range(num_objects):
            transform_matrix = np.eye(4)
            transform_matrix[0, 3] = randrange(-50, 50) #(x-> going right)
            transform_matrix[1, 3] = randrange(-50, 50) #(y-> go backward)
            transform_matrix[2, 3] = randrange(50) #(z-> going up)
            matrices.append((transform_matrix, trimesh.transformations.random_rotation_matrix()))

        print(matrices)
        return matrices

    def is_scene_valid(self, model_ids):
        is_valid = False
        for model_id in model_ids:
            if self.dataset.get_object_category(model_id) in self.transform_categories:
                is_valid = True
                break

        return is_valid

    def generate_single_view_scene(self, scene_id, camera_transform, camera_transform_index,
                                   apply_transform, transform_matrices):
        scene_metadata = {
            'objects': [],
            'render_path': None
        }

        scene, model_ids = self.build_scene(scene_id, transform_matrices, apply_transform)
        scene.camera_transform = camera_transform
        scene.camera.resolution = self.output_resolution
        scene.show()
        bboxes = self.get_bounding_boxes(scene_id, camera_transform, apply_transform)

        for model_id in model_ids:
            if 'room' in model_id:
                continue

            category = self.dataset.get_object_category(model_id)
            bbox = bboxes[model_id]

            model_metadata = {
                'model_id': model_id,
                'bbox': bbox,
                'category': category
            }

            scene_metadata['objects'].append(model_metadata)

        scene_metadata['render_path'] = self.save_render(scene, scene_id, apply_transform, camera_transform_index)
        return scene_metadata

    def save_render(self, scene, scene_id, apply_transform, camera_transform_index):
        img = Image.open(trimesh.util.wrap_as_stream(
            scene.save_image(background=[255, 255, 255, 0], flags={'cull': True}, resolution=None)))

        filename = scene_id + '_' + str(camera_transform_index).zfill(3)
        if apply_transform:
            filename += '_implausible'
        else:
            filename += '_plausible'

        filename += '.png'
        image_path = os.path.join(self.images_output_dir, filename)
        img.save(image_path)
        return os.path.basename(self.images_output_dir) + '/' + filename

    def get_scene_model_ids(self, scene_id):
        objects_metadata = self.dataset.get_scene_metadata(scene_id)['objects']
        model_ids = []

        for object_metadata in objects_metadata:
            model_id = object_metadata['modelID']
            if model_id is None or model_id == '':
                continue

            model_path = self.dataset.get_model_path(model_id)
            if not os.path.exists(model_path):
                continue

            model_ids.append(model_id)

        return model_ids

    def get_bounding_boxes(self, scene_id, camera_transform, transform_matrices, apply_transformation=False):
        objects_metadata = self.dataset.get_scene_metadata(scene_id)['objects']
        bounding_boxes = {}

        transform_index = 0
        for object_metadata in objects_metadata:
            scene = Scene()
            scene.camera_transform = camera_transform
            scene.camera.resolution = self.output_resolution

            model_id = object_metadata['modelID']
            if model_id is None or model_id == '':
                continue

            model_path = self.dataset.get_model_path(model_id)
            if not os.path.exists(model_path) or 'room' in model_path:
                continue

            model = trimesh.load(model_path)
            if isinstance(model, Scene):
                for g in model.geometry:
                    model.geometry[g] = self.color_model(model.geometry[g], (0, 0, 0))
            else:
                model = self.color_model(model, (0, 0, 0))

            if apply_transformation and self.dataset.get_object_category(model_id) in self.transform_categories:
                model.apply_transform(transform_matrices[transform_index][1])
                model.apply_transform(transform_matrices[transform_index][0])
                transform_index += 1
            else:
                transformation_matrix = np.reshape(object_metadata['transform'], (4, 4)).T
                model.apply_transform(transformation_matrix)

            scene.add_geometry(model)

            img = Image.open(trimesh.util.wrap_as_stream(
                scene.save_image(background=[255, 255, 255, 255], flags={'cull': True}, resolution=None)))
            bbox = self.detect_bounding_box(img, [0, 0, 0, 255])
            bounding_boxes[model_id] = bbox

        return bounding_boxes

    def color_model(self, trimesh_object, color):
        trimesh_object.visual.material = SimpleMaterial(ambient=color, diffuse=color, specular=color)
        return trimesh_object

    def detect_bounding_box(self, img, foreground_color):
        img_matrix = ImagesUtils.convert_to_numpy(img)
        m = np.all(img_matrix == foreground_color, axis=-1)
        if len(np.unique(m)) == 1:
            return None

        indices = np.where(m)
        x1 = int(np.min(indices[1]))
        x2 = int(np.max(indices[1]))
        y1 = int(np.min(indices[0]))
        y2 = int(np.max(indices[0]))
        return (x1, y1), (x2, y2)

    def build_scene(self, scene_id, transform_matrices, apply_transformation=False):
        objects_metadata = self.dataset.get_scene_metadata(scene_id)['objects']
        model_ids = []
        scene = Scene()

        transform_index = 0
        for object_metadata in objects_metadata:
            model_id = object_metadata['modelID']
            if model_id is None or model_id == '':
                continue

            model_path = self.dataset.get_model_path(model_id)
            if not os.path.exists(model_path):
                continue

            model_ids.append(model_id)
            model = trimesh.load(model_path)

            transformation_matrix = np.reshape(object_metadata['transform'], (4, 4)).T
            if apply_transformation and self.dataset.get_object_category(model_id) in self.transform_categories:
                original_transform = np.eye(4)
                original_transform[:, 3] = transformation_matrix[:, 3]
                model.apply_transform(transform_matrices[transform_index][1])
                model.apply_transform(transform_matrices[transform_index][0])
                model.apply_transform(original_transform)
                transform_index += 1
            else:
                model.apply_transform(transformation_matrix)

            scene.add_geometry(model)

        return scene, model_ids

    def draw_bounding_box(self, img, bbox):
        draw = ImageDraw.Draw(img)
        draw.rectangle(tuple(map(tuple, bbox)), outline='#ff8888')
        return img

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


generator = Scenes3DBboxRenderRandom('../data')
generator.initialize()
generator.generate()