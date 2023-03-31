import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature, get_matterport_camera_data


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Matterport3D.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" | "test" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    keep_features_in_memory = args.keep_features_in_memory

    # load 3D data (point cloud, color and the corresponding labels)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    # obtain all camera views related information (specificially for Matterport)
    intrinsics, poses, img_dirs, scene_id, num_img = \
            get_matterport_camera_data(data_path, locs_in, args)
    if num_img == 0:
        print('no views inside {}'.format(scene_id))
        return 1

    n_interval = num_rand_file_per_scene
    n_finished = 0
    for n in range(n_interval):

        if exists(join(out_dir, scene_id +'_%d.pt'%(n))):
            n_finished += 1
            print(scene_id +'_%d.pt'%(n) + ' already done!')
            continue
    if n_finished == n_interval:
        return 1

    # short hand for processing 2D features
    device = torch.device('cpu')

    # extract image features and keep them in the memory
    # default: False (extract image on the fly)
    if keep_features_in_memory and openseg_model is not None:
        img_features = []
        for img_dir in tqdm(img_dirs):
            img_features.append(extract_openseg_img_feature(img_dir, openseg_model, text_emb))

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):
        # load pose
        pose = poses[img_id]

        # load per-image intrinsic
        intr = intrinsics[img_id]

        # load depth and convert to meter
        depth_dir = img_dir.replace('color', 'depth')
        _, img_type, yaw_id = img_dir.split('/')[-1].split('_')
        depth_dir = depth_dir[:-8] + 'd'+img_type[1] + '_' + yaw_id[0] + '.png'
        depth = imageio.v2.imread(depth_dir) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth, intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask
        if keep_features_in_memory:
            feat_2d = img_features[img_id].to(device)
        else:
            feat_2d = extract_openseg_img_feature(img_dir, openseg_model, text_emb).to(device)

        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

        counter[mask!=0]+= 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]

    counter[counter==0] = 1e-5
    feat_bank = sum_features/counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args)

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #### Dataset specific parameters #####
    img_dim = (640, 512)
    depth_scale = 4000.0
    #######################################
    visibility_threshold = 0.02 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension
    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'matterport_3d')
    data_root_2d = join(data_dir,'matterport_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range


    if split== 'train': # for training set, export a chunk of point cloud
        args.n_split_points = 20000
        args.num_rand_file_per_scene = 5
    else: # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000
        args.num_rand_file_per_scene = 1

    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if args.openseg_model != '':
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        args.text_emb = tf.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    main(args)
## python matterport_openseg.py --data_dir ../../data --output_dir ../../data/matterport_multiview_openseg --openseg_model ~/workspace/openseg_exported_clip --split train