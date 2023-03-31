'''IoU'''
import numpy as np
from dataset.label_constants import *

UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        # return confusion[:num_classes, :num_classes]
        return confusion

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'scannet_3d' in dataset:
        CLASS_LABELS = SCANNET_LABELS_20
    elif 'matterport_3d_40' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_40
    elif 'matterport_3d_80' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_80
    elif 'matterport_3d_160' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_160
    elif 'matterport_3d' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_21
    elif 'nuscenes_3d' in dataset:
        CLASS_LABELS = NUSCENES_LABELS_16
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
    return mean_iou
