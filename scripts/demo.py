import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import sys
from common.utils.nocs_utils import load_depth
from common.utils.misc_utils import (
    convert_realsense_rgb_depth_to_o3d_pcl,
    get_o3d_pcd_from_np,
    get_scene_grasp_model_params,
)
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
    get_final_grasps_from_predictions_np,
    get_grasp_vis,
)


def get_demo_data_generator(demo_data_path):
    camera_k = np.loadtxt(demo_data_path / "camera_k.txt")

    for color_img_path in demo_data_path.rglob("*_color.png"):
        depth_img_path = color_img_path.parent / (
            color_img_path.stem.split("_")[0] + "_depth.png"
        )
        color_img = cv2.imread(str(color_img_path))  # type:ignore
        depth_img = load_depth(str(depth_img_path))
        yield color_img, depth_img, camera_k


def main(hparams):
    TOP_K = 200  # TODO: use greedy-nms for top-k to get better distributions!
    # Model:
    print("Loading model from checkpoint: ", hparams.checkpoint)
    scene_grasp_model = SceneGraspModel(hparams)

    demo_data_path = Path("outreach/demo_data")
    data_generator = get_demo_data_generator(demo_data_path)
    for rgb, depth, camera_k in data_generator:
        print("------- Showing results ------------")
        pred_dp = scene_grasp_model.get_predictions(rgb, depth, camera_k)
        if pred_dp is None:
            print("No objects found.")
            continue

        all_gripper_vis = []
        for pred_idx in range(pred_dp.get_len()):
            (
                pred_grasp_poses_cam_final,
                pred_grasp_widths,
                _,
            ) = get_final_grasps_from_predictions_np(
                pred_dp.scale_matrices[pred_idx][0, 0],
                pred_dp.endpoints,
                pred_idx,
                pred_dp.pose_matrices[pred_idx],
                TOP_K=TOP_K,
            )

            grasp_colors = np.ones((len(pred_grasp_widths), 3)) * [1, 0, 0]
            all_gripper_vis += [
                get_grasp_vis(
                    pred_grasp_poses_cam_final, pred_grasp_widths, grasp_colors
                )
            ]

        pred_pcls = pred_dp.get_camera_frame_pcls()
        pred_pcls_o3d = []
        for pred_pcl in pred_pcls:
            pred_pcls_o3d.append(get_o3d_pcd_from_np(pred_pcl))
        o3d_pcl = convert_realsense_rgb_depth_to_o3d_pcl(rgb, depth / 1000, camera_k)
        print(">Showing predicted shapes:")
        o3d.visualization.draw(  # type:ignore
            [o3d_pcl] + pred_pcls_o3d
        )
        print(">Showing predicted grasps:")
        o3d.visualization.draw(  # type:ignore
            pred_pcls_o3d + all_gripper_vis
        )


if __name__ == "__main__":
    args_list = None
    if len(sys.argv) > 1:
        args_list = sys.argv[1:]
    hparams = get_scene_grasp_model_params(args_list)
    main(hparams)
