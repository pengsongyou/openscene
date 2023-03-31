import glob, os
import multiprocessing as mp
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from preprocess_util import make_intrinsic, adjust_intrinsic

def process_one_scene(fn):
    '''process one scene.'''

    # process RGB images
    img_name = fn.split('/')[-1]
    img_id = int(int(img_name.split('frame')[-1].split('.')[0])/sample_freq)
    img = imageio.v3.imread(fn)
    img = cv2.resize(img, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_color, str(img_id)+'.jpg'), img)

    # process depth images
    depth_name = img_name.replace('.jpg', '.png').replace('frame', 'depth')
    fn_depth = os.path.join(fn.split('frame')[0], depth_name)
    depth = imageio.v3.imread(fn_depth).astype(np.uint16)
    depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_depth, str(img_id)+'.png'), depth)

    #process poses
    np.savetxt(os.path.join(out_dir_pose, str(img_id)+'.txt'), pose_list[img_id])
    
def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
out_dir = '../../data/replica_processed/replica_2d/'
in_path = '../../data/Replica/' # downloaded original replica data
sample_freq = 10
#####################################

os.makedirs(out_dir, exist_ok=True)

####### Meta Data #######
scene_list = ['office0', 'office1', 'office2', 'office3',
              'office4', 'room0', 'room1', 'room2']
img_dim = (640, 360)
original_img_dim = (1200, 680)
# intrinsic parameters on the original image size
intrinsics = make_intrinsic(fx=600.0, fy=600.0, mx=599.5, my=339.5)

# save the intrinsic parameters of resized images
intrinsics = adjust_intrinsic(intrinsics, original_img_dim, img_dim)
np.savetxt(os.path.join(out_dir, 'intrinsics.txt'), intrinsics)


for scene in tqdm(scene_list):
    out_dir_color = os.path.join(out_dir, scene, 'color')
    out_dir_depth = os.path.join(out_dir, scene, 'depth')
    out_dir_pose = os.path.join(out_dir, scene, 'pose')
    if not os.path.exists(out_dir_color):
        os.makedirs(out_dir_color)
    if not os.path.exists(out_dir_depth):
        os.makedirs(out_dir_depth)
    if not os.path.exists(out_dir_pose):
        os.makedirs(out_dir_pose)

    # save the camera parameters to the folder
    camera_dir = os.path.join(in_path,
            scene, 'traj.txt')
    poses = np.loadtxt(camera_dir).reshape(-1, 4, 4)
    pose_list = poses[::sample_freq]

    files = glob.glob(os.path.join(in_path, scene, 'results', '*.jpg'))
    files = sorted(files)
    files = files[::sample_freq] # every 10 frames

    process_one_scene(files[0])

    p = mp.Pool(processes=mp.cpu_count())
    p.map(process_one_scene, files)
    p.close()