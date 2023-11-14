# SceneGraspNet
This folder contains instructions and code for data generation and training of 
SceneGraspNet.

## Data Generation
- Download NOCS predictions from [here](https://drive.google.com/file/d/1p72NdY4Bie_sra9U8zoUNI4fTrQZdbnc/view?usp=sharing), extract and save `nocs_results` directory.

- Set `workers_per_gpu` and `GPUS` variables inside
[distributed_generate_data.py](scene_grasp/scene_grasp_net/data_generation/distributed_generate_data.py)
as per your resource constraints.

- To generate training and validation data for SceneGraspNet, run the following script
four times with `--type` chosen from the list `[camera_train, camera_val, real_train, real_val]`
respectively. Example:
```bash
python scene_grasp/scene_grasp_net/data_generation/distributed_generate_data.py \
 --data_dir data/NOCSDataset \
 --data_save_dir data/scene_grasp_net_preprpcessed_data/
 --model_path checkpoints/scale_ae.pth \
 --type camera_train \
 --object_deformnet_nocs_results_dir <path-to-nocs-results-dir> \
```

For a single data directory for all the four splits, I recommend passing the same
directory for `--data_save_dir`, such that one folder contains data for all four splits.
For generating a small sample of dataset for debugging, you can additionally pass
`--gen_small_sample`.

Once done, you should see `*.zstd` files inside:
```bash
<data_save_dir>/
├── CAMERA
│   └── train
│       ├── *.pickle.zstd
│       ├── *.pickle.zstd
│       ├── ...
│   └── val
│       ├── *.pickle.zstd
│       ├── ...
├── Real
│   └── test
│       ├── *.pickle.zstd
│       ├── ...
│   └── train
│       ├── *.pickle.zstd
│       ├── ...

```

## Training
We first train the model on the NOCS-Camera dataset:
```bash
python scene_grasp/scene_grasp_net/net_train.py \
    @configs/SceneGraspNet.txt \
    --train_path=<path-to-camera-train-data> \
    --val_path=<path-camera-cal-data> \
```
Please append both `--train_path` and `--val_path` with `file://`.
Example:
```bash
--train_path=file://data/scene_grasp_net_preprocessed_data/CAMERA/train \
--val_path=file://data/scene_grasp_net_preprocessed_data/CAMERA/val
```

## Finetuning
```bash
python scene_grasp/scene_grasp_net/net_train.py \
    @configs/SceneGraspNetRealFineTuning.txt \
    --train_path=<path-to-real-train-data> \
    --val_path=<path-to-real-test-data> \
    --checkpoint=<path-to-camera-train-checkpoint>
```


## Acknowledgment:
Most code here is adapted from [Centersnap](https://github.com/zubair-irshad/CenterSnap)
and [object-deformnet](https://github.com/mentian/object-deformnet) implementations.