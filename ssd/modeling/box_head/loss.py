import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import long

from ssd.utils import box_utils


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio  # 负样本/正样本=3:1

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """计算分类损失和回归损失
        Args:
            confidence: 分类预测结果 [bs,all_anchors,num_classes]=[bs, 24564, 6]  0代表背景类 真实类别=num_classes-1
            predicted_locations: 回归预测结果 [bs,all_anchors,xywh]=[bs, 24564, 4]
            labels: gt_labels 所有anchor的真实框类别标签 0表示背景类 [bs, 24564]
            gt_locations: 所有anchor的真实框回归标签 [bs, 24564, 4]
        Returns:
        """
        num_classes = confidence.size(2)  # 6
        with torch.no_grad():  # 困难样本挖掘
            # derived from cross_entropy=sum(log(p))
            # log_softmax(confidence, dim=2): [bs,24564,6] 对confidence的num_classes维度进行softmax出来再log
            # -F.log_softmax(confidence, dim=2)[:, :, 0]: [bs,24564] 所有anchor属于背景的损失
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]  # [bs,anchor_nums]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]  # [104,6] 正样本[1,25] 负样本[3,75]  正样本+负样本=1+25 + 3+75 = 104
        # 分类损失 sum  正样本:负样本=3:1  预测输入: [pos+neg, num_classes]  gt输入: [pos+neg]
        # F.cross_entropy: 1、对预测输入进行softmax+log运算 -> [pos+neg, num_classes]
        #                  2、对gt输入进行one-hot编码 [pos+neg] -> [pos+neg, num_classes]
        #                  3、再对预测和gt进行交叉熵损失计算
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)  # 正样本预测回归参数 [26,4]
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)  # 正样本gt label [26,4]
        # 回归loss smooth l1 sum（只计算正样本）
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)  # 正样本个数 26
        # 回归总损失（正样本）/正样本个数   分类总损失（正样本+负样本）/正样本个数
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class Focal_loss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(Focal_loss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        l = FocalLoss(size_average = False)
        classification_loss = l(confidence.view(-1, num_classes), labels[mask])

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos