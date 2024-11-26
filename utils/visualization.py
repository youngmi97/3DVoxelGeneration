import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxels(voxels, threshold=0.5):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    voxels = voxels.squeeze().cpu().numpy() > threshold
    ax.voxels(voxels, edgecolor='k')
    
    plt.tight_layout()
    return fig