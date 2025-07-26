import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    # import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDatasetMM
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import pickle
from pathlib import Path

class DemoDataset(KittiDatasetMM):
    def __init__(self, dataset_cfg, class_names, root_path,
                 info_path, training=False, ext='.npy',logger=None,args=None):
        super().__init__(                 # initialise the real MM dataset
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=Path(root_path),
            training=training,
            logger=logger
        )
        if args is not None and getattr(args, 'frames', None):
            wanted = set(args.frames.split(','))
            # prune infos
            self.kitti_infos = [
                info for info in self.kitti_infos
                if info['point_cloud']['lidar_idx'] in wanted
            ]
            print(f"[DemoDataset] restricted to {len(self.kitti_infos)} frames: {sorted(wanted)}")
    
        self.ext = ext
        self.info_path = Path(info_path)

        # Let the base class load kitti_infos and the DataProcessor.
        # Then optionally limit the sample list to a subset / single file:
        if self.root_path.is_file():        # single .npy
            self.sample_file_list = [self.root_path]
            # print("Came here")
        elif self.ext == '.npy':            # folder of fused clouds
            mm_folder = self.root_path / dataset_cfg.MM_PATH
            self.sample_file_list = sorted(mm_folder.glob('*'+self.ext))
        else:                               # raw LiDAR folder
            self.sample_file_list = sorted(
                (self.root_path / 'velodyne').glob('*'+self.ext))

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/models/kitti/PointVit.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/kitti',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='PointVit.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--info_path', type=str, default='../data/kitti/kitti_dbinfos_train_mm.pkl',help='path to your kitti infos .pkl')
    parser.add_argument('--frames', type=str, default='',
    help='Comma-separated lidar_idx values to keep, e.g. "000123,000456" ' '(empty string = keep the whole split)')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

from pcdet.datasets import build_dataloader
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=2,
        logger=logger,
        training=False
    )

    logger.info(f'Total number of samples: \t{len(test_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set, logger=logger)
    # print(model)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for batch_dict in test_loader:
            # logger.info(f'Visualized sample index: \t{idx}')
            # lidar_idx = test_set.kitti_infos[idx]['point_cloud']['lidar_idx']
            # print("Current frame:", lidar_idx) 
            # # print(data_dict.keys())
            # data_dict = test_set.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)
            model.eval()
            pred_dicts = None
            
            with torch.no_grad():
                pred_dicts, ret_dict, batch_dict= model(batch_dict)
                # print(pred_dicts)
                # exit(0)
                # print(len(out))
                print("batch_dict keys:", batch_dict.keys())
                print("ret_dict:", ret_dict)
                print("pred_dicts:", pred_dicts)
            breakpoint()
            pred_boxes  = pred_dicts[0]['pred_boxes'].cpu().numpy()   # (M,7)
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()  # (M,)
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()  # (M,)
            # print(pred_boxes.shape)

            # get GT boxes from dataset (if your DatasetTemplate provides 'gt_boxes')
            # shape will be (1, N, 7) after collate; take [0]
            if len(pred_dicts['pred_boxes'])==0:
                print("➜ No boxes above SCORE_THRESH; try lowering it in cfg.MODEL.POST_PROCESSING")
                continue
            gt_boxes = batch_dict['gt_boxes'][0]
            gt_boxes = gt_boxes.cpu().numpy()         # (N,8)
            gt_labels = gt_boxes[:, 7].astype(int)
            gt_boxes  = gt_boxes[:, :7]

            gt_t = torch.from_numpy(gt_boxes).float().cuda().contiguous()
            pd_t = torch.from_numpy(pred_boxes).float().cuda().contiguous()
            # if(pred_boxes.shape[0] == 0) :
            #     print("No predicted box for this!!")
            #     print("Will exit")
            #     exit(0)
            # compute IoU matrix (N_gt × N_pred)
            # iou_mat = None
            if gt_boxes.shape[0] == 0:
            # no GT: nothing to miss
                missed_gt_boxes = np.zeros((0,7), dtype=float)
            elif pred_boxes.shape[0] == 0:
                # no preds: all GT are missed
                missed_gt_boxes = gt_boxes.copy()
            else:
                # normal IoU path
                for g, p in zip(gt_boxes, pred_boxes):
                    print('Δcenter:', np.round(p[:3] - g[:3], 3),
                        'Δsize:',   np.round(p[3:6] - g[3:6], 3),
                        'Δyaw:',    np.round(p[6]  - g[6],      3))
                    
                gt_t = torch.from_numpy(gt_boxes).float().cuda().contiguous()
                pd_t = torch.from_numpy(pred_boxes).float().cuda().contiguous()
                iou_mat = boxes_iou3d_gpu(gt_t, pd_t).cpu().numpy()   # (N_gt, N_pred)
                max_iou_per_gt = iou_mat.max(axis=1)                  # shape (N_gt,)
                IOU_THRESHOLD    = 0.05
                missed_mask      = max_iou_per_gt < IOU_THRESHOLD
                missed_gt_boxes  = gt_boxes.copy()
            # max_iou_per_gt = iou_mat.max(axis=1)
            # (K,7)

            # draw everything:
            #  - predicted boxes (green)
            #  - missed GT boxes (orange)
            if OPEN3D_FLAG:
                V.draw_scenes(
                    points = batch_dict['points'][:, 1:4],
                    gt_boxes = missed_gt_boxes,
                    ref_boxes = pred_boxes,
                    ref_labels = pred_labels,
                    ref_scores = pred_scores,
                )
            if not OPEN3D_FLAG:
                pass
                # mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
