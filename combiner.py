import os
import shutil

# adjust these if your cwd is different
base_dir        = './data/odometry/'

for seq in os.listdir(base_dir):
    src_calib = os.path.join(base_dir, seq, 'calib.txt')
    dst_calib = os.path.join(base_dir, 'dataset', 'sequences', seq, 'calib.txt')
    print(src_calib)
    # dst_seq      = os.path.join(color_dir, seq, 'velodyne')

    if not os.path.isfile(src_calib):
        print(f"→ skipping {seq}, no calib file found.")
        continue
    else:
        os.remove(src_calib)
        shutil.copyfile(dst_calib,src_calib)
    # if os.path.exists(dst_seq):
    #     print(f"→ already exists: {dst_seq}")
    #     continue

    # move the whole folder:
    
    # if you'd rather copy, comment out the move above and uncomment:
    # shutil.copytree(src_velodyne, dst_seq)

    print(f"✔ moved {dst_calib} to {src_calib}")
