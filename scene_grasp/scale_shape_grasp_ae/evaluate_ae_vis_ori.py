import argparse
import os
import numpy as np

import torch
from common.config import config_global_paths, config_dataset_details
from common.utils.misc_utils import visualize_datapoint_o3d
from common.utils.html_vis_utils import visualize_helper
from scene_grasp.scale_shape_grasp_ae.model.auto_encoder_scale_grasp import (
    PointCloudScaleBasedGraspAE,
)
from scene_grasp.scale_shape_grasp_ae.utils_shape_pretraining import (
    setup_dataloader,
    add_ae_eval_args,
)
from common.utils.shape_utils import init_logs_dir


def get_gw(batch_info, batch_ind, GRIPPER_OFFSET_BINS):
    gw = batch_info["grasp_width_one_hot"][batch_ind].detach().cpu().numpy()
    bin_indices = np.argmax(gw, axis=1)
    gw = (GRIPPER_OFFSET_BINS[bin_indices] + GRIPPER_OFFSET_BINS[bin_indices + 1]) / 2
    return gw


def main():
    parser = argparse.ArgumentParser()
    add_ae_eval_args(parser)
    args = parser.parse_args()

    args.batch_size = min(args.batch_size, args.num_datapoints)

    GRIPPER_FINGER_LENGTH = config_dataset_details.get_gripper_bounds()[0][2]
    GRIPPER_OFFSET_BINS = config_dataset_details.get_gripper_offset_bins()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # setup val dataset.
    if args.split_name != "val":
        print("=======>FIXME<======= using split ", args.split_name)

    logs_root = config_global_paths.PROJECT_LOGS_ROOT / "evaluate_scale_ae"
    logs_dir, logger = init_logs_dir(
        logs_root, logger_name=logs_root.name, input_args=vars(args)
    )
    eval_dataset, eval_dataloader = setup_dataloader(args, return_cat_id=True)
    print("Dataset size: ", len(eval_dataset))

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
    num_scales = 1
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
        # I want to change something. Instead of randomly sampling these scales,
        # I have scale information already in my dataset.
        # Yeah, let me read the data like that. So that I have something to compare
        #  with

        for scale_ind in range(num_scales + 1):
            if scale_ind == 0:
                batch_scales = batch_gt_scales
            else:
                batch_scales = batch_rand_scales[:, scale_ind - 1, :]

            _, endpoints = estimator(batch_xyz, batch_scales)
            pred_batch_xyz = endpoints["xyz"]
            # pred_batch_gs = endpoints["success"]

            for batch_ind in range(batch_size):
                curr_counter = counter + batch_ind
                # gt_pcl = batch_xyz[batch_ind].detach().cpu().numpy()
                if scale_ind == 0:
                    # dump_data = {}
                    gt_gs = gt_batch_gs[batch_ind].detach().cpu().numpy()
                    # gt_is_success = gt_gs > 0.5
                    # gt_gif_path = (
                    #     images_dir
                    #     / f"{curr_counter}_gt_pcl_gs_at_scale_ind_{scale_ind}.gif"
                    # )
                    # visualize_gs_xyz_as_gif(gt_pcl, gt_is_success, gt_gif_path)
                    # dump_data[f"gt_pcl_gs_at_scale_ind_{scale_ind}"] = gt_gif_path

                    gt_xyz = batch_xyz[batch_ind].detach().cpu().numpy()  # (N,3)
                    gt_scale = batch_scales[batch_ind].detach().cpu().numpy()[0]
                    gt_ad = (
                        batch_target["approach_dir"][batch_ind].detach().cpu().numpy()
                    )
                    gt_bd = (
                        batch_target["baseline_dir"][batch_ind].detach().cpu().numpy()
                    )
                    print("Showing gt: ")
                    gt_gw = None
                    if not args.only_show_shape:
                        gt_gwoh = (
                            batch_target["grasp_width_one_hot"][batch_ind]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        gt_gw = config_dataset_details.convert_gwoh_to_gwv_batch(
                            gt_gwoh, GRIPPER_OFFSET_BINS
                        )
                    visualize_datapoint_o3d(
                        gt_xyz,
                        gt_scale,
                        gt_gs,
                        gt_ad,
                        gt_bd,
                        gt_gw,
                        GRIPPER_FINGER_LENGTH,
                    )

                    pred_xyz = pred_batch_xyz[batch_ind].detach().cpu().numpy()  # (N,3)
                    pred_gs = endpoints["success"][batch_ind].detach().cpu().numpy()
                    pred_ad = (
                        endpoints["approach_dir"][batch_ind].detach().cpu().numpy()
                    )
                    pred_bd = (
                        endpoints["baseline_dir"][batch_ind].detach().cpu().numpy()
                    )
                    pred_gw = None
                    if not args.only_show_shape:
                        pred_gwoh = (
                            endpoints["grasp_width_one_hot"][batch_ind]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        pred_gw = config_dataset_details.convert_gwoh_to_gwv_batch(
                            pred_gwoh, GRIPPER_OFFSET_BINS
                        )
                    print("Showing pred: ")
                    visualize_datapoint_o3d(
                        pred_xyz,
                        gt_scale,
                        pred_gs,
                        pred_ad,
                        pred_bd,
                        pred_gw,
                        GRIPPER_FINGER_LENGTH,
                    )
                    continue
                else:
                    continue
                    dump_data = dump_paths[curr_counter]

                # dump_data[
                #     f"scale_{scale_ind}"
                # ] = f"{batch_scales[batch_ind].squeeze().detach().cpu().numpy():.4f}"

                # pred_pcl = pred_batch_xyz[batch_ind].detach().cpu().numpy()
                # pred_success_indices = (
                #     pred_batch_gs[batch_ind].detach().cpu().numpy() > 0.5
                # )
                # pred_gif_path = (
                #     images_dir
                #     / f"{curr_counter}_pred_pcl_gs_at_scale_ind_{scale_ind}.gif"
                # )
                # visualize_gs_xyz_as_gif(pred_pcl, pred_success_indices, pred_gif_path)
                # dump_data[f"pred_pcl_gs_at_scale_ind_{scale_ind}"] = pred_gif_path

                # gt_colors = get_uniform_colors_pyrender(
                #     gt_pcl.shape[0], [0, 255, 0, 128]
                # )
                # pred_colors = get_uniform_colors_pyrender(
                #     pred_pcl.shape[0], [255, 0, 0, 128]
                # )
                # gt_pred_shape_gif_path = (
                #     images_dir
                #     / f"{curr_counter}_gt_pred_shape_at_scale_ind_{scale_ind}.gif"
                # )
                # visualize_list_xyz_colors_as_gif(
                #     [gt_pcl, pred_pcl], [gt_colors, pred_colors], gt_pred_shape_gif_path
                # )
                # dump_data[
                #     f"gt_pred_pcl_at_scale_ind_{scale_ind}"
                # ] = gt_pred_shape_gif_path
                # if scale_ind == 0:
                #     dump_paths.append(dump_data)
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
