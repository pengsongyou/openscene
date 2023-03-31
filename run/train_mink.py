# code modified from https://github.com/wbhu/BPNet/blob/main/tool/train.py
import os
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from MinkowskiEngine import SparseTensor
from util import config
from dataset.point_loader import Point3DLoader, collation_fn, collation_fn_eval_all
from util.util import AverageMeter, intersectionAndUnionGPU, poly_learning_rate, save_checkpoint
from models.mink_unet import MinkUNet18A as Model

best_iou = 0.0


def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='MinkowskiNet training.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/train_mink.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/train_mink.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
    return cfg


def get_logger():
    '''Get logger.'''
    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    '''Main function.'''

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    if not hasattr(args, 'use_shm'):
        args.use_shm = True
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    _ = Point3DLoader(datapath_prefix=args.data_root,
                    voxel_size=args.voxel_size,
                    split='train',
                    aug=args.aug,
                    memcache_init=args.use_shm,
                    loop=5)
    if args.evaluate:
        _ = Point3DLoader(datapath_prefix=args.data_root,
                        voxel_size=args.voxel_size,
                        split='val',
                        aug=args.aug,
                        memcache_init=args.use_shm)
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    train_data = Point3DLoader(datapath_prefix=args.data_root, voxel_size=args.voxel_size,
                                split='train', aug=args.aug, memcache_init=args.use_shm,
                                loop=args.loop, input_color=args.input_color)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                shuffle=(train_sampler is None),
                                                num_workers=args.workers, pin_memory=True,
                                                drop_last=True, collate_fn=collation_fn,
                                                worker_init_fn=worker_init_fn,
                                                sampler=train_sampler)
    if args.evaluate:
        val_data = Point3DLoader(datapath_prefix=args.data_root, voxel_size=args.voxel_size,
                                    split='val', aug=False, memcache_init=args.use_shm,
                                    eval_all=True, input_color=args.input_color)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                    shuffle=False, num_workers=args.workers, pin_memory=True,
                                    drop_last=False, collate_fn=collation_fn_eval_all,
                                    sampler=val_sampler)

    # ####################### Train ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        loss_train, miou_train, macc_train, allacc_train = train(
            train_loader, model, criterion, optimizer, epoch)
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', miou_train, epoch_log)
            writer.add_scalar('mAcc_train', macc_train, epoch_log)
            writer.add_scalar('allAcc_train', allacc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, miou_val, macc_val, allacc_val = validate(
                val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', miou_val, epoch_log)
                writer.add_scalar('mAcc_val', macc_val, epoch_log)
                writer.add_scalar('allAcc_val', allacc_val, epoch_log)
                # remember best iou and save checkpoint
                is_best = miou_val > best_iou
                best_iou = max(best_iou, miou_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: {:.3f}'.format(best_iou))


def get_model(cfg):
    '''Get the 3D model.'''

    model = Model(in_channels=3, out_channels=cfg.classes, D=3)
    return model


def train(train_loader, model, criterion, optimizer, epoch):
    '''Training pipeline.'''

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        (coords, feat, label) = batch_data
        coords[:, :3] += (torch.rand(3) * 100).type_as(coords)

        sinput = SparseTensor(feat.cuda(non_blocking=True),
                              coords.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        output = model(sinput)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(
            output, label.detach(), args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu(
        ).numpy(), union.cpu().numpy(), target.cpu().numpy()

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / \
            (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(
            args.base_lr, current_iter, max_iter, power=args.power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(
                            epoch+1, args.epochs, i+1, len(train_loader),
                            batch_time=batch_time, data_time=data_time,
                            remain_time=remain_time,
                            loss_meter=loss_meter,
                            accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(
                intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(
                intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    miou = np.mean(iou_class)
    macc = np.mean(accuracy_class)
    allacc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
                epoch + 1, args.epochs, miou, macc, allacc))
    return loss_meter.avg, miou, macc, allacc


def validate(val_loader, model, criterion):
    '''Validation.'''

    torch.backends.cudnn.enabled = False
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            (coords, feat, label, inds_reverse) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label.cuda(non_blocking=True)
            output = model(sinput)
            # pdb.set_trace()
            output = output[inds_reverse, :]
            loss = criterion(output, label)

            output = output.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, label.detach(),
                                                                  args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)
            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    miou = np.mean(iou_class)
    macc = np.mean(accuracy_class)
    allacc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(miou, macc, allacc))
    return loss_meter.avg, miou, macc, allacc


if __name__ == '__main__':
    main()
