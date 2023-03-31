# Codes are taken from BPNet, CVPR'21
# https://github.com/wbhu/BPNet/blob/main/dataset/voxelizer.py

import collections
import numpy as np
from dataset.voxelization_utils import sparse_quantize
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

    def __init__(self,
                 voxel_size=1,
                 clip_bound=None,
                 use_augmentation=False,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):
        '''
        Args:
          voxel_size: side length of a voxel
          clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
          scale_augmentation_bound: None or (0.9, 1.1)
          rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
            Use random order of x, y, z to prevent bias.
          translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
          ignore_label: label assigned for ignore (not a training label).
        '''
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    def get_transformation_matrix(self):
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1 / self.voxel_size
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None, trans_aug_ratio=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) &
                     (coords[:, 0] < (lim[0][1] + center[0])) &
                     (coords[:, 1] >= (lim[1][0] + center[1])) &
                     (coords[:, 1] < (lim[1][1] + center[1])) &
                     (coords[:, 2] >= (lim[2][0] + center[2])) &
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None, link=None, return_ind=False):
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        if self.clip_bound is not None:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            clip_inds = self.clip(coords, center, trans_aug_ratio)
            if clip_inds.sum():
                coords, feats = coords[clip_inds], feats[clip_inds]
                if labels is not None:
                    labels = labels[clip_inds]

        # Get rotation and scale
        M_v, M_r = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_v
        if self.use_augmentation:
            rigid_transformation = M_r @ rigid_transformation

        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

        # Align all coordinates to the origin.
        min_coords = coords_aug.min(0)
        M_t = np.eye(4)
        M_t[:3, -1] = -min_coords
        rigid_transformation = M_t @ rigid_transformation
        coords_aug = np.floor(coords_aug - min_coords)

        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)
        coords_aug, feats, labels = coords_aug[inds], feats[inds], labels[inds]

        # Normal rotation
        if feats.shape[1] > 6:
            feats[:, 3:6] = feats[:, 3:6] @ (M_r[:3, :3].T)

        if return_ind:
            return coords_aug, feats, labels, np.array(inds_reconstruct), inds
        if link is not None:
            return coords_aug, feats, labels, np.array(inds_reconstruct), link[inds]

        return coords_aug, feats, labels, np.array(inds_reconstruct)
