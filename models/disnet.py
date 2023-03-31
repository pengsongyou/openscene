'''3D model for distillation.'''

from collections import OrderedDict
from models.mink_unet import mink_unet as model3D
from torch import nn


def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    model = model3D(**kwargs)
    return model


class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, cfg=None):
        super(DisNet, self).__init__()
        if not hasattr(cfg, 'feature_2d_extractor'):
            cfg.feature_2d_extractor = 'openseg'
        if 'lseg' in cfg.feature_2d_extractor:
            last_dim = 512
        elif 'openseg' in cfg.feature_2d_extractor:
            last_dim = 768
        else:
            raise NotImplementedError

        # MinkowskiNet for 3D point clouds
        net3d = constructor3d(in_channels=3, out_channels=last_dim, D=3, arch=cfg.arch_3d)
        self.net3d = net3d

    def forward(self, sparse_3d):
        '''Forward method.'''
        return self.net3d(sparse_3d)
