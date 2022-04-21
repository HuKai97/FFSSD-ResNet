from itertools import product

import torch
from math import sqrt


class PriorBox:
    def __init__(self, cfg):
        # 这里的min_size和max_size是直接给出的  论文是公式求出来的
        self.image_size = cfg.INPUT.IMAGE_SIZE  # 512 图片大小
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS  # 所有层的featue map size [64, 32, 16, 8, 4, 2, 1]
        self.min_sizes = prior_config.MIN_SIZES        # 所有层的min_size[35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
        self.max_sizes = prior_config.MAX_SIZES        # 所有层的max_size[76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.65]
        self.strides = prior_config.STRIDES            # 所有层的stride[8, 16, 32, 64, 128, 256, 512]
        self.aspect_ratios = prior_config.ASPECT_RATIOS  # 所有层的aspect ratio[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = prior_config.CLIP  # True

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y   default box中心点坐标(相对特征图)
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # 生成default box
                # small sized square box  w/h=1:1
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # big sized square box  w/h=1:1
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box  w/h=ratio and 1/ratio
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
