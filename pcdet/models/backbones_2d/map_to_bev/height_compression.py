import torch.nn as nn
import numpy as np
from pcdet.datasets.augmentor.X_transform import X_TRANS
import torch, torch_scatter
from spconv.pytorch import SparseConvTensor

class HeightCompression(nn.Module):
    def __init__(self, model_cfg,  voxel_size=None, point_cloud_range=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.x_trans = X_TRANS()
        self.point_cloud_range = point_cloud_range
        self.voxel_size=voxel_size
        self.type = self.model_cfg.get('TYPE', 'cat')
        self.to_sparse = self.model_cfg.get('TO_SPARSE', False)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        
        spatial_features = encoded_spconv_tensor.dense()
        N, H, W = spatial_features.shape[0], spatial_features.shape[-2], spatial_features.shape[-1] # Batch , C , Y , X
        if self.type == 'cat':
            spatial_features = spatial_features.view(N, -1, H, W)
        else:
            indices = encoded_spconv_tensor.indices.long()
            flat_indices = indices[:, 0] * H * W + indices[:, 2] * W + indices[:, 3]

            if self.type == 'max':
                spatial_features = torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='max')
            elif self.type == 'avg':
                spatial_features = torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='mean')
            elif self.type == 'max_avg':
                spatial_features = torch.cat([
                    torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='max'),
                    torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='mean')
                ], dim=-1)
            else:
                raise NotImplementedError
            spatial_features = spatial_features.view(N, H, W, -1).permute(0, 3, 1, 2)  # (N, C, H, W)

        if self.to_sparse:
            spatial_features = spatial_features.permute(0, 2, 3, 1).contiguous()
            spatial_features = SparseConvTensor.from_dense(spatial_features)

        batch_dict['spatial_features'] = spatial_features


        return batch_dict
