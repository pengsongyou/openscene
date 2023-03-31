import os
import random
import numpy as np
import logging
import argparse

import open3d
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from os.path import join
from util import metric
from util import config

from MinkowskiEngine import SparseTensor
from dataset.point_loader import Point3DLoader, collation_fn_eval_all
from tqdm import tqdm
from run.train_mink import get_model


def get_parser():
    '''Parse the config file.'''
    parser = argparse.ArgumentParser(description='MinkowskiNet evaluation.')
    parser.add_argument('--config', type=str,
                    default='config/scannet/eval_mink.yaml',
                    help='config file')
    parser.add_argument('opts',
                    default=None,
                    help='see config/scannet/train_mink.yaml for all options',
                    nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    '''Main function.'''

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if not hasattr(args, 'use_shm'):
        args.use_shm = True

    if args.use_shm:
        # Following code is for caching dataset into memory
        _ = Point3DLoader(datapath_prefix=args.data_root,
                        voxel_size=args.voxel_size,
                        split='val',
                        aug=False,
                        memcache_init=True,
                        eval_all=True,
                        identifier=6797)
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if os.path.isfile(args.model_path):
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        args.input_color = False

    val_data = Point3DLoader(datapath_prefix=args.data_root, voxel_size=args.voxel_size,
                            split=args.split, aug=False,memcache_init=args.use_shm,
                            eval_all=True, identifier=6797, input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                            shuffle=False, num_workers=args.test_workers,
                                            pin_memory=True, drop_last=False,
                                            collate_fn=collation_fn_eval_all, sampler=val_sampler)

    # ####################### Test ####################### #
    evaluate(model, val_loader)

def evaluate(model, val_loader):
    '''Evaluation MinkowskiNet.'''

    torch.backends.cudnn.enabled = False
    dataset_name = args.data_root.split('/')[-1]
    model.eval()

    with torch.no_grad():
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds = []
            gts = []

            # repeat the evaluation process
            # to account for the randomness in MinkowskiNet voxelization
            seed = np.random.randint(10000)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for i, (coords, feat, label, inds_reverse) in enumerate(tqdm(val_loader)):
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                predictions = model(sinput)
                predictions_enlarge = predictions[inds_reverse, :]
                if 'nuscenes_3d' in dataset_name:
                    label_mask = label!=255
                    label = label[label_mask]
                    predictions_enlarge = predictions_enlarge[label_mask]

                if args.multiprocessing_distributed:
                    dist.all_reduce(predictions_enlarge)
                if args.test_repeats==1:
                    preds.append(predictions_enlarge.detach_().cpu().max(1)[1])
                else:
                    preds.append(predictions_enlarge.detach_().cpu())
                gts.append(label.cpu())
            gt = torch.cat(gts)
            pred = torch.cat(preds)
            if args.test_repeats==1:
                current_iou = metric.evaluate(pred.numpy(),
                                              gt.numpy(),
                                              dataset=dataset_name,
                                              stdout=True)
            else:
                current_iou = metric.evaluate(pred.max(1)[1].numpy(),
                                              gt.numpy(),
                                              dataset=dataset_name)
                if rep_i == 0 and main_process():
                    np.save(join(args.save_folder, 'gt.npy'), gt.numpy())
                store = pred + store
                accumu_iou = metric.evaluate(store.max(1)[1].numpy(),
                                             gt.numpy(),
                                             stdout=True,
                                             dataset=dataset_name)
                if main_process():
                    np.save(join(args.save_folder, 'pred.npy'), store.max(1)[1].numpy())
                print(current_iou, accumu_iou)

if __name__ == '__main__':
    main()
