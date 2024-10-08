import torch
import numpy as np
import os
import sys


def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between distributions p and q.
    Args:
        p (torch.Tensor): Distribution p.
        q (torch.Tensor): Distribution q.
    Returns:
        torch.Tensor: KL divergence value.
    """
    # To avoid division by zero or log(0), add a small epsilon value
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return torch.sum(p * torch.log(p / q))

def jensen_shannon_divergence(voxel_set1, voxel_set2):
    """
    Compute the Jensen-Shannon Divergence between two 3D voxel sets.
    Args:
        voxel_set1 (torch.Tensor): Binary 3D voxel grid (shape: N1 x 128 x 128 x 128).
        voxel_set2 (torch.Tensor): Binary 3D voxel grid (shape: N2 x 128 x 128 x 128).
        N1 and N2: # of samples
    Returns:
        float: Jensen-Shannon Divergence value.
    """
    # Count the number of points lying within each voxel grid across all voxel samples.
    voxel_set1 = voxel_set1.sum(0)
    voxel_set2 = voxel_set2.sum(0)

    # Normalize the voxel grids to make them probability distributions
    voxel_set1_prob = voxel_set1 / voxel_set1.sum()
    voxel_set2_prob = voxel_set2 / voxel_set2.sum()

    # Compute the average distribution (M)
    mean_distribution = (voxel_set1_prob + voxel_set2_prob) / 2

    # Compute KLD(P1 || M) and KLD(P2 || M)
    kl1 = kl_divergence(voxel_set1_prob, mean_distribution)
    kl2 = kl_divergence(voxel_set2_prob, mean_distribution)

    # Jensen-Shannon Divergence is the mean of the two KL divergences
    jsd = (kl1 + kl2) / 2
    return jsd.item()

if __name__ == "__main__":
    sample_path = sys.argv[1]
    shapenet_dir = "../data/hdf5_data" # TODO: CHANGE THE PATH IF NEEDED.
    X_gen = torch.load(sample_path)

    test_set_path = os.path.join(shapenet_dir, f"chair_voxels_test.npy")
    val_set_path = os.path.join(shapenet_dir, f"chair_voxels_val.npy")

    test_set = torch.from_numpy(np.load(test_set_path))
    val_set = torch.from_numpy(np.load(val_set_path))

    X_ref = torch.cat([test_set, val_set])
    
    print(f"JSD: {jensen_shannon_divergence(X_gen, X_ref)}")
