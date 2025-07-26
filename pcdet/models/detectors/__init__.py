from .detector3d_template import Detector3DTemplate

from .voxel_rcnn import VoxelRCNN
from .pvt_ssd import PVTSSD
from .point_vit import PointVit
__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'VoxelRCNN': VoxelRCNN,
    'PVTSSD': PVTSSD,
    'PointVit': PointVit
}

def build_detector(model_cfg, num_class, dataset,logger=None):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset,logger=logger
    )

    return model
