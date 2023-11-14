import sys
from pathlib import Path

import argparse
from collections import defaultdict
from random import shuffle
import numpy as np
from common.config import config_global_paths
from common.config.config_dataset_details import (
    get_emb_dim,
    get_n_points,
    get_shape_grasp_ae_logs_prefix,
)
from common.utils.html_vis_utils import visualize_helper
from scene_grasp.scale_shape_grasp_ae.utils_shape_pretraining import (
    setup_dataloader,
    visualize_gs_xyz_as_gif,
)
from common.utils.shape_utils import init_logs_dir


def evaluate_dataset(args):
    logs_root = (
        config_global_paths.PROJECT_LOGS_ROOT
        / f"{get_shape_grasp_ae_logs_prefix()}__evaluate_data"
    )
    logs_dir, logger = init_logs_dir(
        logs_root, logger_name="evaluate_data", input_args=vars(args)
    )
    dataset, _ = setup_dataloader(args)
    inp_path_to_scale_indices = defaultdict(list)
    for ind in range(len(dataset)):
        data_file_path = dataset.data_files[ind]
        scale = float(
            data_file_path.name.split(
                "network_train_grasp_parameter_data_without_fps_at_scale_"
            )[1].split(".pkl")[0]
        )
        inp_path_to_scale_indices[
            f"{data_file_path.parent.parent.name}_{data_file_path.parent.name}"
        ].append((scale, ind))
    final_inp_path_to_scale_indices = {}
    for inp_path in inp_path_to_scale_indices.keys():
        final_inp_path_to_scale_indices[inp_path] = sorted(
            inp_path_to_scale_indices[inp_path], key=lambda x: x[0]
        )
    inp_path_to_scale_indices = final_inp_path_to_scale_indices

    dump_paths = []
    counter = 0
    images_dir = logs_dir / "images"
    images_dir.mkdir()
    data_tuples = list(inp_path_to_scale_indices.items())
    shuffle(data_tuples)
    logger.info(f"Found {len(data_tuples)} valid datapoints")
    for inp_path, scale_indices in data_tuples:
        my_paths = {}
        for scale_ind, (est_scale, ind) in enumerate(scale_indices):
            inp, target = dataset[ind]
            xyz = inp["xyz"]
            scale = inp["scale"][0]
            if not np.abs(scale - est_scale) < 1e-4:
                logger.warn(f"Might have estimated wrong scale: {scale}!={est_scale}")
            gs = target["success"]
            gif_path = images_dir / f"{counter}_{scale_ind}.gif"
            visualize_gs_xyz_as_gif(xyz, gs, gif_path)
            my_paths[f"cat_path"] = inp_path
            my_paths[f"{scale_ind}-scale"] = f"{scale:.3f}"
            my_paths[f"{scale_ind}-gs"] = gif_path
        dump_paths.append(my_paths)
        counter += 1
        if counter >= args.num_datapoints:
            break
    visualize_helper(dump_paths, logs_dir, title="grasp-dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset parameters:
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(config_global_paths.PROJECT_DATA_ROOT / "nocs_grasp_dataset"),
    )
    parser.add_argument("--split_name", default="val", type=str)
    parser.add_argument(
        "--category_id",
        type=str,
        default=None,
        help="If passed, the network will only be trained for this category",
    )
    # model parameters:
    parser.add_argument(
        "--num_point",
        type=int,
        default=get_n_points(),
        help="number of points, needed if use points",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=get_emb_dim(),
        help="dimension of latent embedding",
    )
    # Resource parameters
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=12, help="number of data loading workers"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--num_datapoints", type=int, default=30)
    args = parser.parse_args()

    evaluate_dataset(args)
