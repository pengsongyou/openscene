import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class ChromaticTranslation(object):
    '''Add random color to the image, input must be an array in [0,255] or a PIL image'''

    def __init__(self, trans_range_ratio=1e-1):
        '''
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        '''
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels


class ChromaticAutoContrast(object):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels):
        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = np.min(feats, 0, keepdims=True)
            hi = np.max(feats, 0, keepdims=True)

            scale = 255 / (hi - lo)

            contrast_feats = (feats - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats, labels


class ChromaticJitter(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return coords, feats, labels


class HueSaturationTranslation(object):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return coords, feats, labels


##############################
# Coordinate transformations
##############################
class RandomHorizontalFlip(object):

    def __init__(self, upright_axis, is_temporal):
        '''
        upright_axis: axis index among x,y,z, i.e. 2 for z
        '''
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels


class ElasticDistortion:

    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, granularity, magnitude):
        '''Apply elastic distortion on sparse coordinate space.

          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        '''
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min +
                                       granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords = coords + interp(coords) * magnitude
        return coords

    def __call__(self, pointcloud):
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    pointcloud = self.elastic_distortion(pointcloud, granularity, magnitude)
        return pointcloud


class Compose(object):
    '''Composes several transforms together.'''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class cfl_collate_fn_factory:
    '''Generates collate function for coords, feats, labels.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    '''

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch = [], [], []

        batch_id = 0
        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                    f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
                )
                break
            coords_batch.append(
                torch.cat((torch.from_numpy(coords[batch_id]).int(),
                           torch.ones(num_points, 1).int() * batch_id), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())

            batch_id += 1

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).int()
        return coords_batch, feats_batch, labels_batch


class cflt_collate_fn_factory:
    '''Generates collate function for coords, feats, labels, point_clouds, transformations.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    '''

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels, pointclouds, transformations = list(zip(*list_data))
        cfl_collate_fn = cfl_collate_fn_factory(limit_numpoints=self.limit_numpoints)
        coords_batch, feats_batch, labels_batch = cfl_collate_fn(list(zip(coords, feats, labels)))
        num_truncated_batch = coords_batch[:, -1].max().item() + 1

        batch_id = 0
        pointclouds_batch, transformations_batch = [], []
        for pointcloud, transformation in zip(pointclouds, transformations):
            if batch_id >= num_truncated_batch:
                break
            pointclouds_batch.append(
                torch.cat((torch.from_numpy(pointcloud), torch.ones(pointcloud.shape[0], 1) * batch_id),
                          1))
            transformations_batch.append(
                torch.cat(
                    (torch.from_numpy(transformation), torch.ones(transformation.shape[0], 1) * batch_id),
                    1))
            batch_id += 1

        pointclouds_batch = torch.cat(pointclouds_batch, 0).float()
        transformations_batch = torch.cat(transformations_batch, 0).float()
        return coords_batch, feats_batch, labels_batch, pointclouds_batch, transformations_batch
