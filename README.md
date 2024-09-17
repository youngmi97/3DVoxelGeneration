<div align=center>
  <h1>
  🪑 3D Volume Generation
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs492d-fall-2024/ target="_blank"><b>KAIST CS492(D): Diffusion Models and Their Applications (Fall 2024)</b></a><br>
    Course Project
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://63days.github.io/ target="_blank"><b>Juil Koo</b></a>  (63days [at] kaist.ac.kr)
  </p>
</div>

<div align=center>
   <img src="./assets/teaser.png">
   <figcaption>
	  3D chair shapes from the <a href=https://shapenet.org/ target="_blank">ShapeNet</a> dataset. The  <a href=https://shapenet.org/ target="_blank">ShapeNet</a> dataset is a large collection of 3D models spanning various object categories, with the 3D chair category being one of the most popular. It includes thousands of diverse chair models, ranging in style and complexity. These models are typically represented as 3D meshes but can be converted into various different formats: voxel grids or point clouds. In this project, we aim to build a diffusion model sampling 3D shapes represented by voxels.
    </figcaption>
</div>

## Description
This project aims to build a 3D diffusion model, specifically targeting 3D volume diffusion. We train the model using voxel data at a resolution of (128, 128, 128) from the <a href=https://shapenet.org/ target="_blank">ShapeNet</a> chair class. A major challenge will be efficiently handling this high-resolution data within limited VRAM constraints.


## Data Specification
The dataset consists of 2,658 chairs represented by binary voxels, where a value of 1 indicates the object's surface at that position. To obtain the voxel data, we voxelize point clouds from the ShapeNet dataset. Run the following command to preprocess the data:

```
python load_data.py
```

A 3D voxel visualization code is in `visualize.ipynb`.

## Tasks
Your task is to implement a diffusion model that generates 3D voxels. You have the freedom to explore any methods or techniques to handle the hih-resolution data efficiently. After implementing the model, run the evaluaiton code provided and report the results. Below are further details on the evaluation.

## Evaluation
⚠️ **_The evaluation script will be provided soon._**


Sample 1,000 voxels using your model and save them in `.npy` format with a shape of `(1000, 128, 128, 128)`. After saving the data, load the samples and run the following command to perform the quantitative evaluation:

```
python run_evaluation.py {PATH/TO/YOUR_SAMPLE_DATA.NPY}
```

## Acknowledgement 
The dataset is from <a href=https://shapenet.org/ target="_blank">ShapeNet</a>.
