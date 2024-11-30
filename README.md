# Voxel Generator with DiT3D and Super Resolution




### Running

Train the super resolution network before sampling from the diffusion model
```
# For super resolution upscale training
python main.py --mode sr --categories airplane

# For diffusion training
python main.py --mode df --categories airplane

# Resume diffusion training from checkpoint
python main.py --mode df --resume_from checkpoints/DF/airplane_epoch_100.pt --categories airplane

```



To generate 1000 samples using checkpoints from df and sr
```
python generate_samples.py chair

```

## Reference
[1] [Learning Representations and Generative Models for 3D Point Clouds](https://arxiv.org/abs/1707.02392)
[2] [DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation](https://arxiv.org/abs/2307.01831)
[3] [NeRF-SR: High-Quality Neural Radiance Fields using Supersampling](https://arxiv.org/abs/2112.01759)
