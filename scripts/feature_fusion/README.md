# Multi-view Feature Fusion

## Overview

Here we provide instructions for multi-view feature fusion on different dataset, including ScanNet, Matterport3D, nuScenes, and Replica. This corresponds to **Section 3.1** in our [paper](https://arxiv.org/abs/2211.15654).

**Note**: For now we provide only codes for multiview fusion with the **[OpenSeg](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/openseg)** model. However, you can easily adapt our code for other per-pixel feature extractor like [LSeg](https://github.com/isl-org/lang-seg) or [OVSeg](https://github.com/facebookresearch/ov-seg).


## Prerequisites

### Data preprocessing
Follow [this instruction](../preprocess/README.md) to obtain the processed 2D and 3D data.
- **3D**: Point clouds in the pytorch format `.pth`
- **2D**: RGB-D images with their intrinsic and extrinsic parameters

### Envinroment
You can simply activate the `openscene` conda environment, or alternatively, make sure the following package installed:
- `torch`
- `tensorflow v2` (for OpenSeg feature extraction)
- `numpy`
- `imageio`
- `tqdm`

To use **OpenSeg** as the feature extractor, you can either take the demo model inside the jupyter notebook from their [official repo](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/openseg), or download from [here](https://drive.google.com/file/d/1DgyH-1124Mo8p6IUJ-ikAiwVZDDfteak/view?usp=sharing).

## Run the code

Take ScanNet as an example, to perform multi-view feature fusion your can run:
```bash
python scannet_openseg.py \
            --data_dir PATH/TO/scannet_processed \
            --output_dir PATH/TO/OUTPUT_DIR \
            --openseg_model PATH/TO/OPENSEG_MODEL \
            --process_id_range 0,100\
            --split train
```

where:
- `data_dir`: path to the pre-processed 2D&3D data from [here](../../openscene#datasets)
- `output_dir`: output directory to save your fused features
- `openseg_model`: path to the OpenSeg model
- `process_id_range`: only process scenes within the range
- `split`: choose from `train`/`val`/`test` data to process

[`replica_openseg.py`](./replica_openseg.py) does not have `process_id_range` and `split`.

## Customized Dataset
For your own customized dataset, one can use the provided codes as a reference. We recommend you to modify [`replica_openseg.py`](./replica_openseg.py), especially if you only care about inferecing your own point clouds without 3D distillation. If you want to do 3D distillation, please check and modify [`scannet_openseg.py`](./scannet_openseg.py).


## Suggestions & Troubleshooting

- For using the OpenSeg model, you need to make sure you have an NVIDIA GPU with **>30G memory**.

For additional help, please refer to the code documentation or contact the author.