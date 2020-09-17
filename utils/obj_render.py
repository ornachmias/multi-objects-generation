import os

import numpy as np
import trimesh
from PIL import Image

# Override trimesh internal implementation of euler_matrix to change euler angles structure
old_euler_matrix = trimesh.transformations.euler_matrix


def new_euler_matrix(ai, aj, ak, axes='sxyz'):
    return old_euler_matrix(ai, aj, ak, 'r' + 'yxz')


trimesh.transformations.euler_matrix = new_euler_matrix


class ObjRender:
    def __init__(self, obj_path, record):
        if not os.path.exists(obj_path):
            raise Exception('Path {} does not exists'.format(obj_path))

        self.record = record
        self.meshes = trimesh.load(obj_path)
        self.load_scene_params()

    def render(self):
        img = Image.open(trimesh.util.wrap_as_stream(
            self.meshes.save_image(resolution=(1000, 1000), background=[255, 255, 255, 0], flags={'cull': True})))
        return self.crop_background(img)

    def show(self):
        self.meshes.show(smooth=True, background=[255, 255, 255, 0])

    def load_scene_params(self):
        azimuth, elevation, theta = self.get_angles()
        azimuth, elevation = self.adjust_shapenet(azimuth, elevation, self.record['object_cls'])
        distance = self.record['distance']
        self.meshes = trimesh.Scene(geometry=self.meshes)
        self.meshes.set_camera(angles=(azimuth, elevation, theta), distance=distance)
        self.meshes.camera.K = self.intrinsics_params()
        self.meshes.camera.z_far *= 100
        self.meshes.camera.z_near = 0.001

    def adjust_shapenet(self, azimuth, elevation, class_name):
        if class_name in ['knife', 'skateboard']:
            return azimuth + (np.pi / 2), elevation

        if class_name == 'pillow':
            return azimuth, elevation + (np.pi / 2)

        if class_name == 'telephone':
            return azimuth, -elevation

        return azimuth, elevation

    def get_angles(self):
        azimuth = np.deg2rad(self.record['azimuth'])
        elevation = np.deg2rad(self.record['elevation'])
        theta = np.deg2rad(self.record['inplane_rotation'])
        return azimuth + (np.pi / 2), -elevation, -theta

    def crop_background(self, img):
        np_array = np.array(img)
        blank_px = [255, 255, 255, 0]
        mask = np_array != blank_px
        cords = np.argwhere(mask)
        x0, y0, z0 = cords.min(axis=0)
        x1, y1, z1 = cords.max(axis=0) + 1
        cropped_box = np_array[x0:x1, y0:y1, z0:z1]
        return Image.fromarray(cropped_box, 'RGBA')

    def intrinsics_params(self):
        intrinsics = np.eye(3)
        intrinsics[0, 0] = self.record['focal'] * self.record['viewport']
        intrinsics[1, 1] = self.record['focal'] * self.record['viewport']
        intrinsics[0, 2] = self.record['px']
        intrinsics[1, 2] = self.record['py']
        return intrinsics
