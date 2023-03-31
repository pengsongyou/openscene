import os
import math
import multiprocessing as mp
import numpy as np
import imageio
import cv2

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(
        image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def process_one_sequence(scene):
    '''process one sequence.'''

    out_dir_color = os.path.join(out_dir, scene, 'color')
    out_dir_pose = os.path.join(out_dir, scene, 'pose')
    out_dir_K = os.path.join(out_dir, scene, 'K')
    os.makedirs(out_dir_color, exist_ok=True)
    os.makedirs(out_dir_pose, exist_ok=True)
    os.makedirs(out_dir_K, exist_ok=True)

    timestamp = sorted(os.listdir(os.path.join(data_path, scene, 'frames')))[-1] # take only the last timestamp
    for cam in cam_locs:
        img_name = os.path.join(data_path, scene, 'frames', timestamp, cam, 'color_image.jpg')
        img = imageio.v3.imread(img_name)
        img = cv2.resize(img, img_size)
        imageio.imwrite(os.path.join(out_dir_color, cam + '.jpg'), img)
        # copy the camera parameters to the folder
        pose_dir = os.path.join(data_path, scene, 'frames', timestamp, cam, 'cam2scene.txt')
        pose = np.asarray([[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                    (x.split(" ") for x in open(pose_dir).read().splitlines())])
        np.save(os.path.join(out_dir_pose, cam+'.npy'), pose)
        # shutil.copyfile(pose_dir, os.path.join(out_dir_pose, cam+'.txt'))
        K_dir = os.path.join(data_path, scene, 'frames', timestamp, cam, 'K.txt')
        K = np.asarray([[float(x[0]), float(x[1]), float(x[2])] for x in
                    (x.split(" ") for x in open(K_dir).read().splitlines())])
        K = adjust_intrinsic(K, intrinsic_image_dim=(1600, 900), image_dim=img_size)
        np.save(os.path.join(out_dir_K, cam+'.npy'), K)

        # shutil.copyfile(pose_dir, os.path.join(out_dir_K, cam+'.txt'))
    print(scene, ' done')


#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'train' # 'train' | 'val'
out_dir = 'data/nuscenes_2d/{}'.format(split)
data_path = '/PATH_TO/nuscenes/{}'.format(split) # downloaded original nuscenes data
scene_list = os.listdir(data_path)
#####################################

os.makedirs(out_dir, exist_ok=True)

cam_locs = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']
img_size = (800, 450)

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_sequence, scene_list)
p.close()
p.join()
