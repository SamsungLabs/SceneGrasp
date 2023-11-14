# Scale-Shape-Grasp-Auto-Encoder
Instructions for data generation and training of scale-shape-grasp auto-encoder.

## Data Generation
To generate this data:
- Download [NOCS dataset](https://geometry.stanford.edu/projects/NOCS_CVPR2019/).
- Configure various project level paths inside [config_global_paths.py](common/config/config_global_paths.py) file. This include providing path to NOCS dataset root.
- Run the following script to generate data:
```
python scene_grasp/scale_shape_grasp_ae/generate_grasp_nocs_scale_dset.py
```


Inside above script, each mesh goes through the following steps:
- A random scale is applied to the mesh. This scale sampled from the scale distribution of the category. This scale distribution is obtained via
[ShapeNet-Sem dataset](https://graphics.stanford.edu/projects/semgeo/) which has meshes in their real-world scale.
- Grasp data is generated using our
[geometric grasp generation tool](scene_grasp/scale_shape_grasp_ae/grasp_gen_tool).
- Finally, the data is processed and saved in pickle files that is easy to be used for network training. 

### Visualize the generated data
Use [evaluate.py](scene_grasp/scale_shape_grasp_ae/evaluate.py)
script which will generate the html visualizations of the generated data.

## Training
For training, run the following command:
```bash
python scene_grasp/scale_shape_grasp_ae/train_ae_mine.py --dataset_root=<path-to-dataset-generated-above>
```

### Visualize the trained network predictions
- To visualize the shape and grasp success: 
```bash
python scene_grasp/scale_shape_ae/evaluate_ae.py --dataset_root <path-to-dataset-root> --model_path <model-path>
```
This will generate HTML visualizations of predicted shape and grasp success at different
scales and compare them against the ground-truths.
- To visualize the grasp orientations in detail:
```bash
python scene_grasp/scale_shape_ae/evaluate_ae.py --dataset_root <path-to-dataset-root> --model_path <model-path>
```
This will iterate over datapoints one-by-one, showing the ground-truth and prediction
one-by-one. This visualization will have grasp orientations and grasp width as well.
