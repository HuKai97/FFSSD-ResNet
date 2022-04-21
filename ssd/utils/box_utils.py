import torch
import math


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
                  回归预测结果  [bs,all_anchors,xywh]=[bs, 24564, 4]
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
               [all_anchor, 4]
        center_variance: a float used to change the scale of center.    0.1
        size_variance: a float used to change of scale of size.         0.2
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)  # [num_anchors, 4] -> [1, num_anchors, 4]
    return torch.cat([
        # xy  利用default box将xy回归参数进行解码
        # 解码后xy坐标=预测的xy参数*default box的wh坐标*default box的xy坐标方差+default box的xy坐标
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        # wh  利用default box将wh回归参数进行解码
        # 解码后的wh坐标=e^(预测的wh参数*default box的wh坐标方差) * default box的wh坐标
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors, iou_threshold):
    """把每个prior box进行正负样本匹配
    Assign ground truth boxes and targets to priors.
    Args:
        gt_boxes (num_targets, 4): ground truth boxes. 图片的所有类别
        gt_labels (num_targets): labels of targets. gt图片的所有类别
        priors (num_priors, 4): corner form priors  先验框 default box xyxy
        iou_threshold: 0.5   IOU阈值，小于阈值设为背景
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets  gt和default box的iou
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors  和每个ground truth box 交集最大的 prior box
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets  和每个prior box 交集最大的 ground truth box
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    # 保证每一个ground truth 匹配它的都是具有最大IOU的prior
    # 根据 best_prior_dix 锁定 best_truth_idx里面的最大IOU prior
    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    # 保证每个ground truth box 与某一个prior box 匹配，固定值为 2 > threshold
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors  提取出所有匹配的ground truth box
    labels = gt_labels[best_target_per_prior_index]
    # 把 iou < threshold 的框类别设置为 bg,即为0
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    # 匹配好boxes
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.  [bs,24564] 所有anchor属于背景的损失
        labels (N, num_priors): the labels. 所有anchor的真实框类别标签 0表示背景类 [bs, 24564]
        neg_pos_ratio:  the ratio between the negative examples and positive examples. 负样本:正样本=3:1
    """
    pos_mask = labels > 0  # 正类True 负类背景类False
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)   # [1个,25个] 第一张图1个正样本 第二张图25个正样本
    num_neg = num_pos * neg_pos_ratio  # [3, 75] 第一张图3个负样本 第二张图75个负样本

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)  # 对负样本损失进行排序
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg  # 选出损失最高的前num_neg个负样本
    return pos_mask | neg_mask


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)
