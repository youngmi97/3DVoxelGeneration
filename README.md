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