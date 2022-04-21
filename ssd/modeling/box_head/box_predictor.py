import torch
from torch import nn

from ssd.layers import SeparableConv2d
from ssd.modeling import registry


class BoxPredictor(nn.Module):  # base box predictor
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # predictor配置文件
        self.cls_headers = nn.ModuleList()   # 7个预测特征层对应的7个分类头 全是3x3conv s=1 p=1
        self.reg_headers = nn.ModuleList()   # 7个预测特征层对应的7个回归头 全是3x3cong s=1 p=1
        # 每层default box个数 [4, 6, 6, 6, 6, 4, 4]
        # 7个预测特征层输出channel数=7个分类头/回归头的输入channel (512, 1024, 512, 256, 256, 256, 256)
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            # conv3x3 s=1 p=1 in_channel=(512, 1024, 512, 256, 256, 256, 256) out_channel=[4, 6, 6, 6, 6, 4, 4]*num_classes
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            # conv3x3 s=1 p=1 in_channel=(512, 1024, 512, 256, 256, 256, 256) out_channel=[4, 6, 6, 6, 6, 4, 4]*4
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []  # 存放所有head分类预测结果
        bbox_pred = []   # 存放所有head回归预测结果
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            # [bs,h,w,num_classes*box_nums]  其中num_classes=6 box_nums=[4, 6, 6, 6, 6, 4, 4]
            # [bs,64,64,24] [bs,32,32,32] [bs,16,16,32] [bs,64,64,32] [bs,64,64,32] [bs,64,64,24] [bs,64,64,24]
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            # [bs,h,w,num_classes*4] 其中num_classes=6 4=xywh
            # [bs,64,64,16] [bs,32,32,24] [bs,16,16,24] [bs,64,64,24] [bs,64,64,24] [bs,64,64,16] [bs,64,64,16]
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        # [bs,all_anchors,num_classes]=[2,24564,6]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        # [bs,all_anchors,xywh]=[bs,24564,4]
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred


@registry.BOX_PREDICTORS.register('SSDBoxPredictor')
class SSDBoxPredictor(BoxPredictor):
    # 继承自base box predict BoxPredictor
    # 重写分类预测头
    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)
    # 重写回归预测头
    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


@registry.BOX_PREDICTORS.register('SSDLiteBoxPredictor')
class SSDLiteBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


def make_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)
