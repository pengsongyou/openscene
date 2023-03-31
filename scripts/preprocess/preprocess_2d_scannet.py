# pre-process ScanNet 2D data
# code adapted from https://github.com/angeladai/3DMV/blob/master/prepare_data/prepare_2d_data.py
#
# note: depends on the sens file reader from ScanNet:
#       https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# which is renamed to scannet_sensordata.py under this directory

# Example usage:
#    python prepare_2d_scannet.py --scannet_path /PATH_TO/scannet/scans \
#                             --output_path ../../data/scannet_2d \
#                             --export_label_images \
#                             --label_map_file /PATH_TO_TSV_FILE/scannetv2-labels.combined.tsv

import argparse
import os
import sys
import csv
import numpy as np
import skimage.transform as sktf
import imageio
from scannet_sensordata import SensorData
from preprocess_util import make_intrinsic, adjust_intrinsic

# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--export_label_images', dest='export_label_images', action='store_true')
parser.add_argument('--label_type', default='label-filt', help='which labels (label or label-filt)')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--label_map_file', default='',
                    help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument('--output_image_width', type=int, default=320, help='export image width')
parser.add_argument('--output_image_height', type=int, default=240, help='export image height')

parser.set_defaults(export_label_images=False)
opt = parser.parse_args()
if opt.export_label_images:
    assert opt.label_map_file != ''
print(opt)


def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k, v in label_mapping.items():
        mapped[image == k] = v
    return mapped.astype(np.uint8)

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    label_mapping = None
    if opt.export_label_images:
        label_map = read_label_mapping(opt.label_map_file, label_from='id', label_to='nyu40id')

    # save the global intrinsic parameters of resized images
    img_dim = (320, 240)
    original_img_dim = (640, 480)
    intrinsics = make_intrinsic(fx=577.870605, fy=577.870605, mx=319.5, my=239.5)
    intrinsics = adjust_intrinsic(intrinsics, original_img_dim, img_dim)
    np.savetxt(os.path.join(opt.output_path, 'intrinsics.txt'), intrinsics)

    scenes = sorted([d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))])
    print('Found %d scenes' % len(scenes))
    for i in range(0,len(scenes)):
        sens_file = os.path.join(opt.scannet_path, scenes[i], scenes[i] + '.sens')
        label_path = os.path.join(opt.scannet_path, scenes[i], opt.label_type)
        if opt.export_label_images and not os.path.isdir(label_path):
            print_error('Error: using export_label_images option but label path %s does not exist' % label_path)
        output_color_path = os.path.join(opt.output_path, scenes[i], 'color')
        
        if os.path.exists(os.path.join(opt.output_path, scenes[i])) and os.path.exists(output_color_path + '/0.jpg'):
            print(scenes[i] + ' already extracted!')
            continue

        if not os.path.isdir(output_color_path):
            os.makedirs(output_color_path)
        output_depth_path = os.path.join(opt.output_path, scenes[i], 'depth')
        if not os.path.isdir(output_depth_path):
            os.makedirs(output_depth_path)
        output_pose_path = os.path.join(opt.output_path, scenes[i], 'pose')
        if not os.path.isdir(output_pose_path):
            os.makedirs(output_pose_path)
        output_label_path = os.path.join(opt.output_path, scenes[i], 'label')
        if opt.export_label_images and not os.path.isdir(output_label_path):
            os.makedirs(output_label_path)

        # read and export
        sys.stdout.write('\r[ %d | %d ] %s\tloading...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        if os.path.exists(sens_file):
            sd = SensorData(sens_file)
        else:
            print(scenes[i] + " does not exist!")
            os.system('python ~/disk2/download-scannet.py -o ~/disk2/scannet --type .sens --id '+ scenes[i])
            sd = SensorData(sens_file)
        sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        sd.export_color_images(output_color_path, image_size=[opt.output_image_height, opt.output_image_width],
                               frame_skip=opt.frame_skip)
        sd.export_depth_images(output_depth_path, image_size=[opt.output_image_height, opt.output_image_width],
                               frame_skip=opt.frame_skip)
        sd.export_poses(output_pose_path, frame_skip=opt.frame_skip)

        if opt.export_label_images:

            for f in range(0, len(sd.frames), opt.frame_skip):
                label_file = os.path.join(label_path, str(f) + '.png')
                image = np.array(imageio.imread(label_file))
                image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0,
                                    preserve_range=True)
                mapped_image = map_label_image(image, label_map)
                imageio.imwrite(os.path.join(output_label_path, str(f) + '.png'), mapped_image)
    print('')


if __name__ == '__main__':
    main()