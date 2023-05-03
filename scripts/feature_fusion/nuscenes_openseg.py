import os
import torch
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper


def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on nuScenes.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" ')
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
    split = args.split
    img_size = args.img_dim
    data_root_2d = args.data_root_2d
    point2img_mapper = args.point2img_mapper
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    cam_locs = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']

    # load 3D data (point cloud, color and the corresponding labels)
    # Only process points with GT label annotation
    locs_in = torch.load(data_path)[0]
    labels_in = torch.load(data_path)[2]
    mask_entire = labels_in!=255

    locs_in = locs_in[mask_entire]
    n_points = locs_in.shape[0]

    scene_id = data_path.split('/')[-1].split('.')[0]
    if exists(join(out_dir, scene_id +'.pt')):
        print(scene_id +'.pt' + ' already done!')
        return 1

    # process 2D features
    scene = join(data_root_2d, split, scene_id)
    img_dir_base = join(scene, 'color')
    pose_dir_base = join(scene, 'pose')
    K_dir_base = join(scene, 'K')
    num_img = len(cam_locs)

    device = torch.device('cpu')

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, args.feat_dim), device=device)


    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, cam in enumerate(tqdm(cam_locs)):
        intr = np.load(join(K_dir_base, cam+'.npy'))
        pose = np.load(join(pose_dir_base, cam+'.npy'))
        img_dir = join(img_dir_base, cam+'.jpg')

        # calculate the 3d-2d mapping
        mapping = np.ones([n_points_cur, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth=None, intrinsic=intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask
        feat_2d = extract_openseg_img_feature(
            img_dir, openseg_model, text_emb, img_size=[img_size[1], img_size[0]]).to(device)

        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

        counter[mask!=0]+= 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]

    counter[counter==0] = 1e-5
    feat_bank = sum_features/counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[point_ids] = True
    mask_entire[mask_entire==True] = mask
    torch.save({"feat": feat_bank[mask].half().cpu(),
                "mask_full": mask_entire},
            join(out_dir, scene_id +'.pt'))

    print(join(out_dir, scene_id +'.pt') + ' is saved!')


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #### Dataset specific parameters #####
    img_dim = (800, 450)
    ######################################

    args.cut_num_pixel_boundary = 5 # do not use the features on the image boundary
    args.feat_dim = 768 # CLIP feature dimension
    split = args.split
    data_dir = args.data_dir
    args.img_dim = img_dim

    data_root = join(data_dir, 'nuscenes_3d')
    data_root_2d = join(data_dir,'nuscenes_2d')

    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

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