import os
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from os.path import join, exists
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Replica.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    scene_id = args.scene

    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    keep_features_in_memory = args.keep_features_in_memory

    # load 3D data
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

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
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
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
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        # calculate the 3d-2d mapping based on the depth
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

    ##### Dataset specific parameters #####
    img_dim = (640, 360)
    depth_scale = 6553.5

    scenes = ['room0', 'room1', 'room2', 'office0',
              'office1', 'office2', 'office3', 'office4']
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension

    data_dir = args.data_dir

    data_root = join(data_dir, 'replica_3d')
    data_root_2d = join(data_dir,'replica_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

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

    # load intrinsic parameter
    intrinsics=np.loadtxt(os.path.join(args.data_root_2d, 'intrinsics.txt'))

    # calculate image pixel-3D points correspondances   
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsics,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    for scene in scenes:
        data_path=os.path.join(data_root, f'{scene}.pth')
        args.scene=scene
        process_one_scene(data_path, out_dir, args)



if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)
## python replica_openseg.py --data_dir ../../../../datasets/replica_processed/ --output_dir tmp --openseg_model /cluster/project/cvg/songyou/workspace/openseg_exported_clip/