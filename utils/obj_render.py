import os

import trimesh
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


class ObjRender:
    def __init__(self, obj_path, record):
        if not os.path.exists(obj_path):
            raise Exception('Path {} does not exists'.format(obj_path))

        self.record = record
        self.meshes = trimesh.load(obj_path)
        self.load_scene_params()

    def render(self):
        img = Image.open(trimesh.util.wrap_as_stream(
            self.meshes.save_image(resolution=self.meshes.camera.resolution, background=[255, 255, 255, 0])))
        return self._crop_background(img)

    def show(self):
        # self.meshes.camera.resolution=(self.record['img_size'][0], self.record['img_size'][1])
        self.meshes.show(smooth=True, background=[255, 255, 255, 0])

    def load_scene_params(self):
        self.adjust_shapenet()
        theta = np.deg2rad(self.record['inplane_rotation'])
        elevation = np.deg2rad(self.record['elevation'])
        azimuth = np.deg2rad(self.record['azimuth'])
        distance = self.record['distance']
        print('a={}, e={}, t={}, px={}, py={}'.format(int(self.record['azimuth']), int(self.record['elevation']),
                                                      int(self.record['inplane_rotation']),
                                                      int(self.record['px']), int(self.record['py'])))

        azimuth *= -1.
        elevation += (np.pi/2)
        self.meshes.set_camera(distance=distance)
        self.meshes.camera.K = self.intrinsics()
        self.meshes.apply_transform(self.z_rotation(azimuth))
        self.meshes.apply_transform(self.y_rotation(elevation))
        self.meshes.apply_transform(self.z_rotation(theta))
        # r = self.inner_plane_rotation(azimuth, elevation, theta, distance)
        # self.meshes.apply_transform(r)

    def intrinsics(self):
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = self.record['focal'] * self.record['viewport']
        intrinsics[1, 1] = self.record['focal'] * self.record['viewport']
        intrinsics[0, 2] = self.record['px']
        intrinsics[1, 2] = self.record['py']
        intrinsics[2, 2] = 1
        return intrinsics

    def _crop_background(self, img):
        np_array = np.array(img)
        blank_px = [255, 255, 255, 0]
        mask = np_array != blank_px
        cords = np.argwhere(mask)
        x0, y0, z0 = cords.min(axis=0)
        x1, y1, z1 = cords.max(axis=0) + 1
        cropped_box = np_array[x0:x1, y0:y1, z0:z1]
        return Image.fromarray(cropped_box, 'RGBA')

    def adjust_shapenet(self):
        if self.record['object_cls'] == 'knife' or self.record['object_cls'] == 'skateboard':
            self.record['azimuth'] += 180
        elif self.record['object_cls'] == 'pillow':
            self.record['elevation'] += 90
        elif self.record['object_cls'] == 'telephone':
            self.record['azimuth'] += 90
            self.record['elevation'] *= -1
        else:
            self.record['azimuth'] += 90

    def adjust_3d(self):
        self.record['azimuth'] *= -1
        self.record['elevation'] -= 90
        self.record['inplane_rotation'] *= -1

    def x_rotation(self, theta):
        r = np.eye(4)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        r[1, 1] = cos_theta
        r[1, 2] = -1. * sin_theta
        r[2, 1] = sin_theta
        r[2, 2] = cos_theta
        return r

    def y_rotation(self, theta):
        r = np.eye(4)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        r[0, 0] = cos_theta
        r[0, 2] = sin_theta
        r[2, 0] = -1. * sin_theta
        r[2, 2] = cos_theta
        return r

    def z_rotation(self, theta):
        r = np.eye(4)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        r[0, 0] = cos_theta
        r[0, 1] = -1. * sin_theta
        r[1, 0] = sin_theta
        r[1, 1] = cos_theta
        return r

    def inner_plane_rotation(self, a, e, theta, distance):
        sin_a, sin_e, sin_theta = np.sin(a), np.sin(e), np.sin(theta)
        cos_a, cos_e, cos_theta = np.cos(a), np.cos(e), np.cos(theta)

        c = np.zeros((3, 1))
        c[0] = distance * cos_e * sin_a
        c[1] = -1. * distance * cos_e * cos_a
        c[2] = distance * sin_e

        r_z = np.eye(3)
        r_z[0, 0] = cos_a
        r_z[0, 1] = -1. * sin_a
        r_z[1, 0] = sin_a
        r_z[1, 1] = cos_a

        r_x = np.eye(3)
        r_x[1, 1] = cos_e
        r_x[1, 2] = -1. * sin_e
        r_x[2, 1] = sin_e
        r_x[2, 2] = cos_e

        r_z_2 = np.eye(3)
        r_z_2[0, 0] = cos_theta
        r_z_2[0, 1] = -1. * sin_theta
        r_z_2[1, 0] = sin_theta
        r_z_2[1, 1] = cos_theta

        r = np.matmul(np.matmul(r_z_2, r_x), r_z)

        intrinsics = np.eye(3)
        intrinsics[0, 0] = self.record['focal'] * self.record['viewport']
        intrinsics[1, 1] = self.record['focal'] * self.record['viewport']

        p = np.zeros((3, 4))
        p[:3, :3] = r
        p[:3, 3] = -1. * np.matmul(r, c).reshape(3)

        result = np.eye(4)
        result[:3, :4] = np.matmul(intrinsics, p)
        return result
