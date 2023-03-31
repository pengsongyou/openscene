import os
import multiprocessing as mp
import numpy as np
import plyfile
import torch



NUSCENES_FULL_CLASSES = ( # 32 classes
    'noise',
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck',
    'flat.driveable_surface',
    'flat.other',
    'flat.sidewalk',
    'flat.terrain',
    'static.manmade',
    'static.other',
    'static.vegetation',
    'vehicle.ego',
    'unlabeled',
)

VALID_NUSCENES_CLASS_IDS = ()

NUSCENES_CLASS_REMAP = 256*np.ones(32) # map from 32 classes to 16 classes
NUSCENES_CLASS_REMAP[2] = 7 # person
NUSCENES_CLASS_REMAP[3] = 7
NUSCENES_CLASS_REMAP[4] = 7
NUSCENES_CLASS_REMAP[6] = 7
NUSCENES_CLASS_REMAP[9] = 1 # barrier
NUSCENES_CLASS_REMAP[12] = 8 # traffic cone
NUSCENES_CLASS_REMAP[14] = 2 # bicycle
NUSCENES_CLASS_REMAP[15] = 3 # bus
NUSCENES_CLASS_REMAP[16] = 3
NUSCENES_CLASS_REMAP[17] = 4 # car
NUSCENES_CLASS_REMAP[18] = 5 # construction vehicle
NUSCENES_CLASS_REMAP[21] = 6 # motorcycle
NUSCENES_CLASS_REMAP[22] = 9 # trailer ???
NUSCENES_CLASS_REMAP[23] = 10 # truck
NUSCENES_CLASS_REMAP[24] = 11 # drivable surface
NUSCENES_CLASS_REMAP[25] = 12 # other flat??
NUSCENES_CLASS_REMAP[26] = 13 # sidewalk
NUSCENES_CLASS_REMAP[27] = 14 # terrain
NUSCENES_CLASS_REMAP[28] = 15 # manmade
NUSCENES_CLASS_REMAP[30] = 16 # vegetation


def process_one_sequence(fn):
    '''process one sequence.'''

    scene_name = fn.split('/')[-2]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    category_id = np.ascontiguousarray(v[:, -1]).astype(int)

    if not export_all_points: # we only consider points with annotations
        dir_timestamp = fn[:-9] + 'scene-timestamps.npy'
        timestamp = np.load(dir_timestamp)
        mask = (timestamp==timestamp.max())[:, 0] # mask for points with annotations
        coords = coords[mask]
        category_id = category_id[mask]

    category_id[category_id==-1] = 0
    remapped_labels = NUSCENES_CLASS_REMAP[category_id]
    remapped_labels -= 1

    torch.save((coords, 0, remapped_labels),
            os.path.join(out_dir,  scene_name + '.pth'))
    print(fn)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'val' # 'train' | 'val'
out_dir = 'data/nuscenes_3d/{}'.format(split)
in_path = '/PATH_TO/nuscenes/{}'.format(split) # downloaded original nuscenes data
export_all_points = True # default we export all points within 0.5 sec
scene_list = os.listdir(in_path)
################

os.makedirs(out_dir, exist_ok=True)
files = []
for scene in scene_list:
    files.append(os.path.join(in_path, scene, 'scene.ply'))

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_sequence, files)
p.close()
p.join()
