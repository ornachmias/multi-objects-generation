import os

import trimesh
import numpy as np
from PIL import Image
from trimesh.scene.lighting import Light, DirectionalLight


class ObjRender:
    def __init__(self, obj_path, record):
        if not os.path.exists(obj_path):
            raise Exception('Path {} does not exists'.format(obj_path))

        self.record = record
        self.meshes = trimesh.load(obj_path)
        self.load_scene_params()

    def render(self):
        img = Image.open(trimesh.util.wrap_as_stream(
            self.meshes.save_image(resolution=None, background=[255, 255, 255, 0])))
        return self._crop_background(img)

    def show(self):
        self.meshes.show(smooth=True, background=[255, 255, 255, 0])

    def load_scene_params(self):
        elevation = np.deg2rad(self.record['elevation'])
        azimuth = np.deg2rad(self.record['azimuth'])
        theta = np.deg2rad(self.record['inplane_rotation'])
        distance = self.record['distance']
        focal = self.record['focal']

        c = self.camera_center(azimuth, elevation, distance)
        azimuth, elevation = self.rotate_cord_system(azimuth, elevation)
        r = self.rotation_matrix(azimuth, elevation)
        p = self.perspective_project_matrix(focal, r, c)
        p = np.vstack([p, [0, 0, 0, 1]])

        self.meshes = trimesh.Scene(geometry=self.meshes)
        self.meshes.set_camera(angles=(0, 0, -theta), distance=distance*2)
        self.meshes.camera.z_far *= 100
        self.meshes.apply_transform(p)

    def camera_center(self, azimuth, elevation, distance):
        sin_a, cos_a = np.sin(azimuth), np.cos(azimuth)
        sin_e, cos_e = np.sin(elevation), np.cos(elevation)

        c = np.zeros((3, 1))
        c[0, 0] = distance * cos_e * sin_a
        c[1, 0] = -distance * cos_e * cos_a
        c[2, 0] = distance * sin_e

        return c

    def rotate_cord_system(self, azimuth, elevation):
        return -azimuth, -((np.pi / 2) - elevation)

    def rotation_matrix(self, azimuth, elevation):
        sin_a, cos_a = np.sin(azimuth), np.cos(azimuth)
        sin_e, cos_e = np.sin(elevation), np.cos(elevation)

        r_z = np.eye(3)
        r_z[0, 0] = cos_a
        r_z[0, 1] = -sin_a
        r_z[1, 0] = sin_a
        r_z[1, 1] = cos_a

        r_x = np.eye(3)
        r_x[1, 1] = cos_e
        r_x[1, 2] = -sin_e
        r_x[2, 1] = sin_e
        r_x[2, 2] = cos_e

        return np.matmul(r_x, r_z)

    def perspective_project_matrix(self, f, r, c):
        m = 1
        p_1 = np.zeros((3, 3))
        p_1[0, 0] = m * f
        p_1[1, 1] = m * f
        p_1[2, 2] = -1.

        p_2 = np.zeros((3, 4))
        p_2[:3, :3] = r
        p_2[:3, 3] = -np.matmul(r, c).reshape(3)

        return np.matmul(p_1, p_2)

    def _crop_background(self, img):
        np_array = np.array(img)
        blank_px = [255, 255, 255, 0]
        mask = np_array != blank_px
        cords = np.argwhere(mask)
        x0, y0, z0 = cords.min(axis=0)
        x1, y1, z1 = cords.max(axis=0) + 1
        cropped_box = np_array[x0:x1, y0:y1, z0:z1]
        return Image.fromarray(cropped_box, 'RGBA')
