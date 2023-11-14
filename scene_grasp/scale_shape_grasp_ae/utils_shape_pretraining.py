import sys
from pathlib import Path
import numpy as np
from common.config import config_global_paths, config_dataset_details


# centersnap
def add_to_sys_path(root_path):
    print("Adding to python path: ", root_path)
    sys.path = [str(root_path)] + sys.path


file_path = Path(__file__)
add_to_sys_path(file_path.parent.parent.parent.absolute())  # centersnap
add_to_sys_path(file_path.parent.parent.parent.parent.absolute())  # proj-root

import torch
from scene_grasp.scale_shape_grasp_ae.dataset.shape_dataset_grasp_map import (
    ShapeDatasetGraspMap,
)
from common.utils.misc_utils import visualize_list_xyz_colors_as_gif


def setup_dataloader_from_params(
    dataset_root,
    split_name,
    num_point,
    category_id,
    return_cat_id,
    return_datafile_ind,
    batch_size,
    num_workers,
):
    dataset = ShapeDatasetGraspMap(
        dataset_root,
        mode=split_name,
        augment=False,
        n_points=num_point,
        category_id=category_id,
        return_cat_id=return_cat_id,
        return_datafile_ind=return_datafile_ind,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return dataset, dataloader


def setup_dataloader(
    args, return_cat_id: bool = False, return_datafile_ind: bool = False
):
    return setup_dataloader_from_params(
        args.dataset_root,
        args.split_name,
        args.num_point,
        args.category_id,
        return_cat_id,
        return_datafile_ind,
        args.batch_size,
        args.num_workers,
    )


def visualize_gs_xyz_as_gif(xyz, gs, gif_path):
    # Create point-cloud
    colors = np.zeros((xyz.shape[0], 4), dtype=np.uint8)
    is_success = gs > 0.5
    colors[is_success, 1] = 255
    colors[~is_success, 0] = 255
    colors[:, 3] = 255
    visualize_list_xyz_colors_as_gif([xyz], [colors], gif_path)

def add_ae_dataset_args(parser):
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument(
        "--num_point",
        type=int,
        default=config_dataset_details.get_n_points(),
        help="number of points, needed if use points",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=12, help="number of data loading workers"
    )
    parser.add_argument("--split_name", default="val", type=str)
    parser.add_argument(
        "--category_id",
        type=str,
        default=None,
        help="If passed, the network will only be trained for this category",
    )

def add_ae_eval_args(parser):
    add_ae_dataset_args(parser)
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=config_dataset_details.get_emb_dim(),
        help="dimension of latent embedding",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--num_datapoints", type=int, default="30")
    parser.add_argument("--only_show_shape", action="store_true")
    parser.add_argument("--choose_bd_sign", type=bool, default=True)
