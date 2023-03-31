import glob, os
import multiprocessing as mp
import numpy as np
import imageio
import cv2
import shutil
from tqdm import tqdm
from preprocess_util import adjust_intrinsic


def remove_items(test_list, item):
    return [i for i in test_list if i != item]

def obtain_intr_extr_matterport(file):
    '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''

    lines = file.readlines()
    
    intrinsics = []
    extrinsics = []
    img_names = []
    for i, line in enumerate(lines):
        line = line.strip()
        if 'intrinsics_matrix' in line:
            line = line.replace('intrinsics_matrix ', '')
            line = line.split(' ')
            line = remove_items(line, '')
            if len(line) !=9:
                print('something wrong at {}'.format(i))
            intrinsic = np.asarray(line).astype(float).reshape(3, 3)
            intrinsics.extend([intrinsic, intrinsic, intrinsic, intrinsic, intrinsic, intrinsic])
        elif 'scan' in line:
            line = line.split(' ')
            img_names.append(line[2])
            
            line = remove_items(line, '')[3:]
            if len(line) != 16:
                print('something wrong at {}'.format(i))
            extrinsic = np.asarray(line).astype(float).reshape(4, 4)
            extrinsics.append(extrinsic)

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics

def process_one_scene(fn):
    '''process one scene.'''

    # process RGB images
    img_name = fn.split('/')[-1]
    img_id = np.where(img_names==img_name)[0].item()

    img = imageio.v3.imread(fn)
    img = cv2.resize(img, img_dim, interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(os.path.join(out_dir_color, img_name), img)

    # process depth images
    pano_d, img_type, yaw_id = fn.split('/')[-1].split('_')
    fn_depth = fn.replace('color', 'depth')
    fn_depth = fn_depth[:-8] + 'd'+img_type[1] + '_' + yaw_id[0] + '.png'
    depth_name = fn_depth.split('/')[-1]
    depth = imageio.v3.imread(fn_depth).astype(np.uint16)
    depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(os.path.join(out_dir_depth, depth_name), depth)

    #process poses
    file_name = img_name.split('.jpg')[0]
    pose = pose_list[img_id]
    pose[:3, 1] *= -1.0
    pose[:3, 2] *= -1.0
    np.savetxt(os.path.join(out_dir_pose, file_name+'.txt'), pose)

    #process intrinsic parameters
    intrinsic = adjust_intrinsic(intr_list[img_id], original_img_dim, img_dim)
    np.savetxt(os.path.join(out_dir_intrinsic, file_name+'.txt'), intrinsic)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'train' # 'train' | 'val' | 'test'
out_dir = '../../data/matterport_2d/'
in_path = '../../data/matterport/scans' # downloaded original matterport data
scene_list = process_txt('../../dataset/matterport/scenes_{}.txt'.format(split))
#####################################

os.makedirs(out_dir, exist_ok=True)


####### Meta Data #######
img_dim = (640, 512)
original_img_dim = (1280, 1024)


for scene in tqdm(scene_list):
    out_dir_color = os.path.join(out_dir, scene, 'color')
    out_dir_depth = os.path.join(out_dir, scene, 'depth')
    out_dir_pose = os.path.join(out_dir, scene, 'pose')
    out_dir_intrinsic = os.path.join(out_dir, scene, 'intrinsic')
    if not os.path.exists(out_dir_color):
        os.makedirs(out_dir_color)
    if not os.path.exists(out_dir_depth):
        os.makedirs(out_dir_depth)
    if not os.path.exists(out_dir_pose):
        os.makedirs(out_dir_pose)
    if not os.path.exists(out_dir_intrinsic):
        os.makedirs(out_dir_intrinsic)

    # save the camera parameters to the folder
    camera_dir = os.path.join(in_path,
            scene, 'undistorted_camera_parameters', '{}.conf'.format(scene))
    img_names, intr_list, pose_list = obtain_intr_extr_matterport(open(camera_dir))
    # out_dir_camera = os.path.join(out_dir, scene, 'camera.conf')
    # shutil.copyfile(camera_dir, out_dir_camera)

    files = glob.glob(os.path.join(in_path, scene, 'undistorted_color_images', '*.jpg'))
    p = mp.Pool(processes=mp.cpu_count())
    p.map(process_one_scene, files)
    p.close()
