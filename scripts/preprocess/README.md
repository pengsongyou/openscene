# 2D & 3D Data Preprocessing

## Overview

This document provides instructions for pre-processing both 3D and 2D data for different datasets, including 
- ScanNet
- Matterport3D
- nuScenes
- Replica

One can also use the provided code as a reference for pre-processing your customized dataset.

## Prerequisites

### Environment
Before you begin, simply activate the `openscene` conda environment.

Alternatively, make sure the following package installed:
- `torch`
- `numpy`
- `plyfile`
- `opencv-python`
- `imageio`
- `pandas`
- `tqdm`

### Download the original dataset
- **ScanNet**: Download ScanNet v2 data from the [official ScanNet website](https://github.com/ScanNet/ScanNet).

- **Matterport**: Download Matterport3D dataset from the [official website](https://niessner.github.io/Matterport/).

- **nuScenes**: Download nuScenes from the [nuScenes official website](https://www.nuscenes.org/nuscenes), and [nuImages](https://www.nuscenes.org/nuimages).

- **Replica**: You can download through [the script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) provided by NICE-SLAM.
## Run the code

For preprocessing 3D point clouds with GT labels (except for Replica, no semantic labels), one can simply run:
```bash
python preprocess_3d_{$DATASET_NAME}.py
```

For preprocessing 2D RGB-D images, one can also simply run:
```bash
python preprocess_2d_{$DATASET_NAME}.py
```

**Note**: In the code, you might need to modify the following:
- `in_path`: path to the original downloaded dataset
- `out_dir`: output directory to save your processed data
- `scene_list`: path to the list containing all scenes
- `split`: choose from `train`/`val`/`test` data to process

The only exception is [`preprocess_2d_scannet.py`](./preprocess_2d_scannet.py). The usage can be found in the code.


## Folder structure
Once running the pre-processing code above, you should have a data structure like below. Here we choose the processed ScanNet as an example:

```
data/
│
├── scannet_2d
│   │
│   ├── scene0000_00
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │   └── intrinsic (only for Matterport3D)
│   │
│   ├── scene0000_01
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │   └── intrinsic
│   │
│   └── ...
|   |
|   └── intrinsics.txt (fixed intrinsic parameters for all scenes)
│
└── scannet_3d
    │
    ├── train
    │   ├── scene0000_00.pth
    │   ├── scene0000_01.pth
    │   ├── ...
    │   ├── scene0706_00.pth
    │
    └── val
        ├── scene0011_00.pth
        └── ...
    
```

**Customized dataset**: Make sure that you have the same structure after preprocessing.
## Suggestions & Troubleshooting

If you encounter any issues, refer to the following troubleshooting tips:

- We downscale 2D images for all dataset, so make sure you set `img_dim` and `original_img_dim` correctly.
- To reproduce the results shown in Table 3 and 5 in the paper, you need to change `num_classes` in [`preprocess_3d_matterport_K_num_classes.py`](./preprocess_3d_matterport_K_num_classes.py), which corresponds to the top K number of object classes in NYU Class annotation.
- Matterport3D has per-image intrinsic parameters, so we have a separate folder for it. All other datasets have the global intrinsic parameters for all images and all scenes.
- No train/val/test split for Replica.
- Under [`dataset`](../../dataset), we provide for ScanNet and matterport the `scene_list` (the list for all scene names) and official `.tsv` files for label mapping.

For additional help, please refer to the code documentation or contact the author.