import os
import pickle

import glob
mypath= './data/kitty'
possible_paths = glob.glob('data/kitti/*.pkl')
# print(possible_paths)
# Find and load the file
for path in possible_paths:
    if not os.path.exists(path):
        continue
    
    fname = path.split('/')[-1]
    
    if path is None:
        raise FileNotFoundError(f"Could not find '{fname}'. Checked paths: {possible_paths}")

    with open(path, 'rb') as f:
        kitti_infos_val = pickle.load(f)

    print(f"Loaded: {fname}")
    print(f"Number of samples in {fname}:", len(kitti_infos_val))
    print()