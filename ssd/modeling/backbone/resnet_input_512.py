import torch.nn as nn
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)   # sequeeze
        y = self.fc(y).view(b, c, 1, 1)   # expansion

        return x * y.expand_as(x)         # scale


class Channel_Attention(nn.Module):   # CAM

    def __init__(self, channel, r=16):
        super(Channel_Attention, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)

        self._fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // r, channel, 1, bias=False)
        )

        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self._avg_pool(x)   # avg pooling
        y1 = self._fc(y1)

        y2 = self._max_pool(x)   # max pooling
        y2 = self._fc(y2)

        y = self._sigmoid(y1 + y2)  # add sigmoid
        return x * y                # scale


class Spatial_Attention(nn.Module):

    def __init__(self, kernel_size=3):
        super(Spatial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self._layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)    # avg pool in every pixel
        max_mask, _ = torch.max(x, dim=1, keepdim=True)  # max pool in every pixel
        mask = torch.cat([avg_mask, max_mask], dim=1)    # concat

        mask = self._layer(mask)   # conv
        return x * mask            # scale


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # resnet18 and resnet34
    expansion = 1   # 每一层内部channel是否发生变换  1=不变

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16, se=False, cbam=False):
        """
        Args:
            inplanes: 模块输入特征图channel
            planes: 模块输出特征图channel
            stride: 步长 模块第一个卷积的步长 =1实线残差结构 不下采样  =2虚线残差结构 下采样
            downsample: 下采样 虚线残差结构 conv1x1 s=2 + bn
            groups: 分组卷积组数  1=普通卷积  BasicBlock只支持=1
            base_width:
            dilation: 空洞卷积 BasicBlock不支持
            norm_layer: bn
            reduction: 模块中间层的channel
            se: se注意力机制
            cbam: cbam注意力机制
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only support groups = 1 and base_width = 64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not support in BasicBlock")
        # Both self.conv1 and self.downsample layer downsample the input when stride != 1
        # 是否使用注意力机制
        self.se = se
        self.cbam = cbam
        # conv+bn+relu
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu =nn.ReLU(inplace=True)
        # conv + bn
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # attention layers
        self.se_layer = SELayer(planes, reduction)
        self.ca_layer = Channel_Attention(planes, reduction)
        self.sa_layer = Spatial_Attention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x          # 残差分支

        out = self.conv1(x)   # conv + bn + relu
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # conv + bn
        out = self.bn2(out)

        if self.se and not self.cbam:   # se
            out = self.se_layer(out)
        if not self.se and self.cbam:   # cbam
            out = self.ca_layer(out)
            out = self.sa_layer(out)

        # 是否需要下采样  实线残差结构不需要下采样  虚线残差结构需要下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity           # add
        out = self.relu(out)      # relu

        return out


class Bottleneck(nn.Module):
    # resnet50 and resnet101
    expansion = 4   # 每一层内部channel是否发生变换  第三个卷积是第一个和第二个卷积channel的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16, se=False, cbam=False):
        """
        Args:
            inplanes: 模块输入特征图channel
            planes: 模块输出特征图channel
            stride: 步长 1  第二个卷积的步长  =1不下采样=实线残差结构  =2下采样=虚线残差结构
            downsample: 下采样 虚线残差结构 conv1x1 s=2 + bn
            groups: 分组卷积组数  1=普通卷积
            base_width: 64
            dilation: 空洞卷积
            norm_layer: bn
            reduction: 模块中间层的channel  16
            se: se注意力机制 默认False
            cbam: cbam注意力机制 默认False
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # 默认情况下inplanes = width

        # 是否使用attention机制
        self.se = se
        self.cbam = cbam
        #
        self.conv1 = conv1x1(inplanes, width)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.se_layer = SELayer(planes * self.expansion, reduction)
        self.ca_layer = Channel_Attention(planes * self.expansion, reduction)
        self.sa_layer = Spatial_Attention()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差结构
        identity = x

        # conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # conv + bn + relu
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # conv + bn
        out = self.conv3(out)
        out = self.bn3(out)

        if self.se and not self.cbam:   # se
            out = self.se_layer(out)
        if not self.se and self.cbam:   # cbam
            out = self.ca_layer(out)
            out = self.sa_layer(out)

        if self.downsample is not None:  # 空=实线残差结构   非空=虚线残差结构
            identity = self.downsample(x)

        out += identity         # add
        out = self.relu(out)    # relu

        return out


class ResNet(nn.Module):
    def __init__(self, block=None, blocks=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, extras=None, se=False, cbam=False, ff=False):
        """
        Args:
            block: res18/34=BasicBlock   res50/101=Bottleneck
            blocks: [3, 4, 6, 3] conv2_x  conv3_x  conv4_x  conv5_x中堆叠的次数
            zero_init_residual:
            groups:
            width_per_group:
            replace_stride_with_dilation:
            norm_layer: bn
            extras: resnet    [512, 256, 128, 64, 128]  一系列额外添加层的输出channel
                    resnet-ff [128, 256, 512, 256, 128, 64, 128]
            se: 是否使用se
            cbam: 是否使用cbam
            ff: 是否使用feature fusion结构
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer  # bn

        self.inplanes = 64  # max pool之后第一个卷积层的输入channel
        self.dilation = 1

        self.blocks = blocks  # [3, 4, 6, 3]
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.se = se  # Squeeze-and-Excitation Module
        self.cbam = cbam  # Convolutional Block Attention Module
        self.ff = ff  # Feature Fusion Module
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, self.blocks[0])               # conv2_x
        self.layer2 = self._make_layer(block, 128, self.blocks[1], stride=2,    # conv3_x
                                       dilate=replace_stride_with_dilation[0])
        self.conv2 = nn.Conv2d(512, 256, 1)
        self.layer3 = self._make_layer(block, 256, self.blocks[2], stride=2,    # conv4_x
                                       dilate=replace_stride_with_dilation[1])
        self.conv3 = nn.Conv2d(1024, 256, 1)
        self.bi1 = nn.UpsamplingBilinear2d(scale_factor=2)   # 2倍上采样
        self.layer4 = self._make_layer(block, 512, self.blocks[3], stride=2,    # conv5_x
                                       dilate=replace_stride_with_dilation[2])

        self.conv4 = nn.Conv2d(2048, 256, 1)
        self.bi2 = nn.UpsamplingBilinear2d(scale_factor=4)         # 4倍上采样

        self.conv5 = nn.Conv2d(768, 512, 1)
        self.bn2 = nn.BatchNorm2d(512)

        if self.ff:
            self.extra_layers_ff = nn.Sequential(* self._add_extras_ff(block, extras))
        else:
            self.extra_layers = nn.Sequential(*self._add_extras(block, extras))

        for m in self.modules():       # init
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, se=self.se, cbam=self.cbam))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, se=self.se, cbam=self.cbam))

        return nn.Sequential(*layers)

    def _add_extras(self, block, extras):
        layers = []
        layers += self._make_layer(block, extras[1], 2, stride=2)
        layers += self._make_layer(block, extras[2], 2, stride=2)
        layers += self._make_layer(block, extras[3], 2, stride=2)
        in_channels = extras[3] * block.expansion
        layers += [nn.Conv2d(in_channels, extras[4] * block.expansion, kernel_size=2)]
        return layers

    def _add_extras_ff(self, block, extras):
        self.inplanes = 512
        layers = []
        layers += self._make_layer(block, extras[0], 1)
        layers += self._make_layer(block, extras[1], 1, stride=2)
        layers += self._make_layer(block, extras[2], 1, stride=2)
        layers += self._make_layer(block, extras[3], 1, stride=2)
        layers += self._make_layer(block, extras[4], 1, stride=2)
        layers += self._make_layer(block, extras[5], 1, stride=2)
        layers += self._make_layer(block, extras[6], 1, stride=2)
        return layers

    def forward(self, x):
        if not self.ff:           # 不使用特征融合
            features = []         # 存放7个预测特征层
            x = self.conv1(x)     # conv1+bn+relu
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)   # max pool

            x = self.layer1(x)    # layer1

            x = self.layer2(x)    # layer2
            features.append(x)    # 预测特征层1

            x = self.layer3(x)    # layer3
            features.append(x)    # 预测特征层2

            x = self.layer4(x)    # layer4
            features.append(x)    # 预测特征层3

            x = self.extra_layers[0](x)
            x = self.extra_layers[1](x)
            features.append(x)              # 预测特征层4

            x = self.extra_layers[2](x)
            x = self.extra_layers[3](x)
            features.append(x)              # 预测特征层5

            x = self.extra_layers[4](x)
            x = self.extra_layers[5](x)
            features.append(x)              # 预测特征层6

            x = self.extra_layers[6](x)
            features.append(x)              # 预测特征层7

            return tuple(features)
        else:      # 使用特征融合
            features = []  # 输入x=[bs,3,512,512]

            x = self.conv1(x)     # conv1+bn+relu
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)   # max pool   [bs,64,128,128]

            x = self.layer1(x)    # layer1  conv2_x   [bs,256,128,128]

            x = self.layer2(x)               # layer2   conv3_x  [bs,512,64,64]
            features.append(self.conv2(x))   # 预测特征层1

            x = self.layer3(x)                         # layer3   conv4_x  [bs,1024,32,32]
            features.append(self.bi1(self.conv3(x)))   # 预测特征层3  2倍上采样

            x = self.layer4(x)                         # layer4   conv5_x  [bs,2048,16,16]
            features.append(self.bi2(self.conv4(x)))   # 预测特征层4  4倍上采样

            # features: [bs,256,64,64] x 3
            x = torch.cat((features), 1)               # 特征融合  [bs,768,64,64]

            x = self.conv5(x)                          # 在融合后的特征图上进行conv
            x = self.bn2(x)                            # [bs,512,64,64]

            feature_map = []

            x = self.extra_layers_ff[0](x)             # 1  [bs,512,64,64]
            feature_map.append(x)

            x = self.extra_layers_ff[1](x)             # 2  [bs,1024,32,32]
            feature_map.append(x)

            x = self.extra_layers_ff[2](x)             # 3 [bs,2048,16,16]
            feature_map.append(x)

            x = self.extra_layers_ff[3](x)             # 4 [bs,1024,8,8]
            feature_map.append(x)

            x = self.extra_layers_ff[4](x)             # 5 [bs,512,4,4]
            feature_map.append(x)

            x = self.extra_layers_ff[5](x)             # 6 [bs,256,2,2]
            feature_map.append(x)

            x = self.extra_layers_ff[6](x)             # 7 [bs,512,1,1]
            feature_map.append(x)

            return tuple(feature_map)


@registry.BACKBONES.register('R50_512')
def R50_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks=cfg.MODEL.RESNET.BLOCKS, extras=cfg.MODEL.RESNET.EXTRAS,
                   se=cfg.MODEL.RESNET.SE, cbam=cfg.MODEL.RESNET.CBAM, ff=cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('R101_512')
def R101_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks=cfg.MODEL.RESNET.BLOCKS, extras=cfg.MODEL.RESNET.EXTRAS,
                   se=cfg.MODEL.RESNET.SE, cbam=cfg.MODEL.RESNET.CBAM, ff=cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    # resnet = ResNet(block=Bottleneck, blocks=[3, 4, 6, 3], extras=[128, 256, 512, 256, 128, 64, 128],
    #                 se = False, cbam=False, ff=True)
    resnet = ResNet(block=Bottleneck, blocks=[3, 4, 6, 3], extras=[512, 256, 128, 64, 128],
                    se=False, cbam=False, ff=False)
    summary(resnet.to('cuda'), (3, 512, 512))
    print(resnet)
