from torch import nn
import torch.nn.functional as F

from ssd.modeling import registry
from ssd.modeling.anchors.prior_box import PriorBox
from ssd.modeling.box_head.box_predictor import make_box_predictor
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss
from .loss import Focal_loss


@registry.BOX_HEADS.register('SSDBoxHead')
class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)  # 预测器 类型SSDBoxPredictor
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)  # 损失函数 类型MutiBoxLoss
        # self.loss_evaluator = Focal_loss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)  # 后处理
        self.priors = None  # default box

    def forward(self, features, targets=None):
        """
        Args:
            features: tuple7 [bs,512,64,64] [bs,1024,32,32] [bs,512,16,16] [bs,256,8,8] [bs,256,4,4] [bs,256,2,2] [bs,256,1,1]
            targets: 'boxes': [bs, 24564, 4]   'labels': [bs, 24564]
        Returns:

        """
        # cls_logits: [bs,all_anchors,num_classes]=[bs, 24564, 6]
        # bbox_pred: [bs,all_anchors,xywh]=[bs, 24564, 4]
        cls_logits, bbox_pred = self.predictor(features)  # 预测器前线传播
        if self.training:  # 训练返回预测结果和loss
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:  # 测试和验证返回预测结果和{}
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        """
        Args:
            cls_logits: 分类预测结果 [bs,all_anchors,num_classes]=[bs, 24564, 6]  0代表背景类 真实类别=num_classes-1
            bbox_pred: 回归预测结果 [bs,all_anchors,xywh]=[bs, 24564, 4]
            targets: gt 'boxes': [bs, 24564, 4]   'labels': [bs, 24564]
        Returns: 预测结果+loss
        """
        # gt_boxes: [bs, 24564, 4]   gt_labels: [bs, 24564]
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        # 计算损失
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)  # 回归损失 + 分类损失
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict  # 返回预测结果和loss

    def _forward_test(self, cls_logits, bbox_pred):
        """
        Args:
            cls_logits: 分类预测结果 [bs,all_anchors,num_classes]=[bs, 24564, 6]
            bbox_pred: 回归预测结果  [bs,all_anchors,xywh]=[bs, 24564, 4]
        Returns:
        """
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)  # 分类结果在classes_num维度进行softmax [bs,24564,num_classes]
        # 利用default box将预测回归参数进行解码  [bs, num_anchors, xywh(归一化后的真实的预测坐标)]
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        # xywh->xyxy  将归一化后的预测坐标由xywh->xyxy
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)  # 后处理
        return detections, {}
