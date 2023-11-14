import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent.absolute()
print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path
root_path = Path(__file__).parent.parent.parent.parent.absolute()
print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import argparse
import os
import numpy as np
import torch
import trimesh
from common.config import config_global_paths, config_dataset_details
from common.utils.trimesh_utils import get_tri_pcl, get_tri_pcl_uniform_color
from common.utils.misc_utils import (
    visualize_list_xyz_colors_as_gif,
    get_uniform_colors_pyrender,
)
from common.utils.html_vis_utils import visualize_helper
from scene_grasp.scale_shape_grasp_ae.model.auto_encoder_scale_grasp import (
    PointCloudScaleBasedGraspAE,
)
from scene_grasp.scale_shape_grasp_ae.utils_shape_pretraining import (
    setup_dataloader,
    visualize_gs_xyz_as_gif,
)
from common.utils.shape_utils import init_logs_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(config_global_paths.PROJECT_DATA_ROOT / "nocs_grasp_dataset"),
    )
    parser.add_argument(
        "--num_point",
        type=int,
        default=config_dataset_details.get_n_points(),
        help="number of points, needed if use points",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=config_dataset_details.get_emb_dim(),
        help="dimension of latent embedding",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=12, help="number of data loading workers"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--split_name", default="val", type=str)
    parser.add_argument(
        "--category_id",
        type=str,
        default=None,
        help="If passed, the network will only be trained for this category",
    )
    parser.add_argument("--num_datapoints", type=int, default="30")
    parser.add_argument("--choose_bd_sign", type=bool, default=True)
    args = parser.parse_args()

    args.batch_size = min(args.batch_size, args.num_datapoints)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # setup val dataset.
    if args.split_name != "val":
        print("=======>FIXME<======= using split ", args.split_name)

    logs_root = config_global_paths.PROJECT_LOGS_ROOT / "evaluate_scale_ae"
    logs_dir, logger = init_logs_dir(
        logs_root, logger_name=logs_root.name, input_args=vars(args)
    )
    eval_dataset, eval_dataloader = setup_dataloader(args, return_cat_id=True)

    # Load the model
    estimator = PointCloudScaleBasedGraspAE(
        args.emb_dim, args.num_point, args.choose_bd_sign
    )
    estimator.cuda()
    estimator.load_state_dict(torch.load(args.model_path))

    counter = 0
    dump_paths = []
    images_dir = logs_dir / "images"
    images_dir.mkdir()
    num_scales = 5
    _, _, cat_id_to_min_max_scale = config_dataset_details.get_nocs_information_dicts()
    for data in eval_dataloader:
        inp, batch_target = data
        batch_xyz = inp["xyz"]
        batch_gt_scales = inp["scale"]
        batch_xyz = batch_xyz[:, :, :3].cuda().to(torch.float)
        batch_gt_scales = batch_gt_scales.cuda().to(torch.float)
        batch_target = {
            key: value.cuda().to(torch.float) for key, value in batch_target.items()
        }
        gt_batch_gs = batch_target["success"]

        # Let's evaluate on 5 random scales
        def get_batch_min_max_scales(extrema_ind):
            batch_ext_scales = [
                cat_id_to_min_max_scale[cat_id.item()][extrema_ind]
                for cat_id in batch_target["cat_id"]
            ]
            batch_ext_scales = torch.tensor(batch_ext_scales).cuda().to(torch.float)
            batch_ext_scales = (
                batch_ext_scales.unsqueeze(-1).unsqueeze(-1).expand(-1, num_scales, -1)
            )
            return batch_ext_scales

        batch_min_scales = get_batch_min_max_scales(0)
        batch_max_scales = get_batch_min_max_scales(1)
        batch_size = len(batch_xyz)
        batch_rand_scales = torch.rand((batch_size, num_scales, 1))
        batch_rand_scales = batch_rand_scales.cuda().to(torch.float)
        batch_rand_scales = batch_min_scales + batch_rand_scales * (
            batch_max_scales - batch_min_scales
        )
        batch_rand_scales = torch.sort(batch_rand_scales, dim=1)[0]
        dump_paths_mini = []
        for scale_ind in range(num_scales + 1):
            if scale_ind == 0:
                batch_scales = batch_gt_scales
            else:
                batch_scales = batch_rand_scales[:, scale_ind - 1, :]

            _, endpoints = estimator(batch_xyz, batch_scales)
            pred_xyz = endpoints["xyz"]
            pred_batch_gs = endpoints["success"]

            for batch_ind in range(batch_size):
                curr_counter = counter + batch_ind
                gt_pcl = batch_xyz[batch_ind].detach().cpu().numpy()
                if scale_ind == 0:
                    dump_data = {}
                    gt_success_indices = (
                        gt_batch_gs[batch_ind].detach().cpu().numpy() > 0.5
                    )
                    gt_gif_path = (
                        images_dir
                        / f"{curr_counter}_gt_pcl_gs_at_scale_ind_{scale_ind}.gif"
                    )
                    visualize_gs_xyz_as_gif(gt_pcl, gt_success_indices, gt_gif_path)
                    dump_data[f"gt_pcl_gs_at_scale_ind_{scale_ind}"] = gt_gif_path
                else:
                    dump_data = dump_paths[curr_counter]

                dump_data[
                    f"scale_{scale_ind}"
                ] = f"{batch_scales[batch_ind].squeeze().detach().cpu().numpy():.4f}"

                pred_pcl = pred_xyz[batch_ind].detach().cpu().numpy()
                pred_success_indices = (
                    pred_batch_gs[batch_ind].detach().cpu().numpy() > 0.5
                )
                pred_gif_path = (
                    images_dir
                    / f"{curr_counter}_pred_pcl_gs_at_scale_ind_{scale_ind}.gif"
                )
                visualize_gs_xyz_as_gif(pred_pcl, pred_success_indices, pred_gif_path)
                dump_data[f"pred_pcl_gs_at_scale_ind_{scale_ind}"] = pred_gif_path

                gt_colors = get_uniform_colors_pyrender(
                    gt_pcl.shape[0], [0, 255, 0, 128]
                )
                pred_colors = get_uniform_colors_pyrender(
                    pred_pcl.shape[0], [255, 0, 0, 128]
                )
                gt_pred_shape_gif_path = (
                    images_dir
                    / f"{curr_counter}_gt_pred_shape_at_scale_ind_{scale_ind}.gif"
                )
                visualize_list_xyz_colors_as_gif(
                    [gt_pcl, pred_pcl], [gt_colors, pred_colors], gt_pred_shape_gif_path
                )
                dump_data[
                    f"gt_pred_pcl_at_scale_ind_{scale_ind}"
                ] = gt_pred_shape_gif_path
                if scale_ind == 0:
                    dump_paths.append(dump_data)
        counter += batch_size
        if counter >= args.num_datapoints:
            break
    vis_title = (
        f"Eval-Scale-AE\n"
        f"\t-Model: {args.model_path}\n"
        f"\t-Dataset: {args.dataset_root} -- Split: {args.split_name}\n"
    )
    visualize_helper(dump_paths, logs_dir, title=vis_title)


if __name__ == "__main__":
    main()
