import torch.nn as nn
import torch.nn.functional as F

from ssd.layers import L2Norm
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}

# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
def add_vgg(cfg, batch_norm=False):
    """根据配置文件搭建SSD中的backbone(Conv1_1->Conv7)模型结构
    pool3x3/1替换vgg中的pool2x2/2  conv6（空洞卷积 r=6）、conv7替换fc6、fc7  其中conv4_3、conv7都得到预测特征图
    Args:
        cfg: Conv1_1->Conv7配置文件 [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
             数字表示当前卷积层的输出channel 'M'表示maxpooling采用向下取整的形式如9x9->4x4 'C'相反表示向上取整如9x9->5x5
        batch_norm: True conv+bn+relu
                    False conv relu  原论文是没有bn层的
    Returns: backbone层结构 卷积层+maxpool共17层
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # ceil mode feature map size是奇数时 最大池化下采样向上取整 如 9x9->5x5
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # 下一层的输入=这一层的输出
    # 论文里有说将vgg中的pool5的2x2/2变为3x3/1 这样这层的池化后特征图尺寸不变
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # conv6使用了空洞卷积 原论文也是这样说的
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, size=300):
    """
    backbone之后的4个额外添加层 生成4个预测特征图
    Extra layers added to VGG for feature scaling
    Args:
        cfg: 4个额外添加层的配置文件  [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]
             数字代表当前卷积层输出channel当前卷积层s=1  'S'代表当前卷积层s=2
        i: 1024 第一个额外特征层 也就是conv8_1的输入channel
        size: 输入图片大小
    Returns: 4个额外添加层结构
    """
    layers = []
    in_channels = i
    flag = False  # 用来控制 kernel_size= 1 or 3
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE  # 输入图片大小
        # ssd-vgg16 backbone配置信息 [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        vgg_config = vgg_base[str(size)]
        # vgg之后额外的一些特征提取层配置信息 [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]
        extras_config = extras_base[str(size)]

        # 初始化backbone
        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        # 初始化额外特征提取层
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.l2_norm = L2Norm(512, scale=20)
        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):  # 参数初始化
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):  # 载入backbone预训练权重
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []  # 存放6个预测特征层
        # 前23层conv1_1->conv4_3(包括relu层)前序传播
        for i in range(23):
            x = self.vgg[i](x)
        # Conv4_3 L2 normalization  论文里这样说的 不知道为什么
        # ssd没有bn 网络层靠前，方差比较大，需要加一个L2标准化，以保证和后面的检测层差异不是很大
        s = self.l2_norm(x)
        features.append(s)   # append conv4_3

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)  # append conv7

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)  # append conv8_2 conv9_2 conv10_2 conv11_2

        # ssd300返回六个预测特征层  ssd512返回7个预测特征层
        # tuple7 [bs,512,64,64] [bs,1024,32,32] [bs,512,16,16] [bs,256,8,8] [bs,256,4,4] [bs,256,2,2] [bs,256,1,1]
        return tuple(features)


@registry.BACKBONES.register('vgg')
def vgg(cfg, pretrained=True):
    """搭建vgg模型+加载预训练权重
    Args:
        cfg: vgg配置文件
        pretrained: 是否加载预训练模型
    Returns:
        返回加载完预训练权重的vgg模型
    """
    model = VGG(cfg)  # 搭建vgg模型
    if pretrained:    # 加载预训练权重
        # 这里默认会从amazonaws平台下载vgg模型预训练权重文件
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    from ssd.config import cfg
    from thop.profile import profile
    cfg.merge_from_file("../../../configs/vgg_ssd512_voc12.yaml")
    model = VGG(cfg)
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 512, 512)).to(device)
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    print("%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))
    print()
