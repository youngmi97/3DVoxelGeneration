import os
import multiprocessing as mp
import h5py
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def voxelize(pts, vox_res=(128,128,128)):
    """
    pts: np.ndarray [N,3]
    vox_res: (height, width, length)
    vox_range: 
    """
    vox_res = np.array(vox_res)
    min_bounds = np.min(pts, axis=0)
    max_bounds = np.max(pts, axis=0)

    normalized_pts = (pts - min_bounds) / (max_bounds - min_bounds)
    scaled_pts = normalized_pts * (vox_res - 1)
    
    voxel_grid = np.zeros(vox_res, dtype=np.float32)
    voxel_indices=  np.floor(scaled_pts).astype(np.int32)

    for idx in voxel_indices:
        voxel_grid[tuple(idx)] = 1
    return voxel_grid


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    shapenet_part_seg_link = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
    zipfile = os.path.basename(shapenet_part_seg_link)

    shapenet_dir = os.path.join(DATA_DIR, "hdf5_data")

    if not os.path.exists(shapenet_dir):
        print(f"[*] ShapeNet PartSeg dataset will be downloaded under {DATA_DIR}")
        os.system(f"wget --no-check-certificate {shapenet_part_seg_link} -P {DATA_DIR}")
        os.system(f"unzip {os.path.join(DATA_DIR, zipfile)} -d {DATA_DIR}")
        os.system(f"rm {os.path.join(DATA_DIR, zipfile)}")
    
    for split in ["train", "val", "test"]:
        all_chair_voxels = []
        with open(os.path.join(shapenet_dir, f"{split}_hdf5_file_list.txt")) as f:
            train_files_list = [line.rstrip() for line in f]
        
        for train_fp in train_files_list:
            train_fp = os.path.join(shapenet_dir, train_fp)
            assert os.path.exists(train_fp), f"{train_fp}"
            f = h5py.File(train_fp)
            
            pc_all = np.array(f["data"][:]).astype(np.float32)
            label_all = np.array(f["label"][:]).squeeze()
            chair_indices = np.where(label_all == 4)[0]

            pc_chairs = pc_all[chair_indices]
            args = [pc_chairs[i] for i in range(len(pc_chairs))]
            with mp.Pool(mp.cpu_count()) as pool:
                res = pool.map_async(voxelize, args)
                res = res.get()

            voxels = np.stack(res, 0)
            all_chair_voxels.append(voxels)

            f.close()

        all_chair_voxels = np.concatenate(all_chair_voxels, 0)
        print(f"# of {split} chair voxels: {len(all_chair_voxels)}")

        np.save(os.path.join(shapenet_dir, f"chair_voxels_{split}.npy"), all_chair_voxels)

        
