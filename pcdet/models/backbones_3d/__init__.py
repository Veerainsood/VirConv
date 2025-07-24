from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VirConv8x,VirConvL8x
from .spconv_mini_unet import MiniUNetV1

__all__ = {
    'VirConv8x': VirConv8x,
    'VirConvL8x': VirConvL8x,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'MiniUNetV1': MiniUNetV1
}
