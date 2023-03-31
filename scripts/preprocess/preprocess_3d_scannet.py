import glob, os
import multiprocessing as mp
import numpy as np
import plyfile
import torch

# Map relevant classes to {0,1,...,19}, and ignored classes to 255
remapper = np.ones(150) * (255)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def process_one_scene(fn):
    '''process one scene.'''

    fn2 = fn[:-3] + 'labels.ply'
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, 3:6]) / 127.5 - 1
    a = plyfile.PlyData().read(fn2)
    w = remapper[np.array(a.elements[0]['label'])]

    torch.save((coords, colors, w),
            os.path.join(out_dir, fn[:-4].split('/')[-1] + '.pth'))
    print(fn, fn2)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'train' # choose between 'train' | 'val'
out_dir = '../../data/scannet_3d/{}'.format(split)
in_path = '/PATH_TO/scannet/scans' # downloaded original scannet data
scene_list = process_txt('../../dataset/scannet/scannetv2_{}.txt'.format(split))
#####################################

os.makedirs(out_dir, exist_ok=True)
files = []
files2 = []
for scene in scene_list:

    files.append(glob.glob(os.path.join(in_path,
                    scene, '*_vh_clean_2.ply'))[0])
    files2.append(glob.glob(os.path.join(in_path,
                    scene,'*_vh_clean_2.labels.ply'))[0])
    assert len(files) == len(files2)

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_scene, files)
p.close()
p.join()
