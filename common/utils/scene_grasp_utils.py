import matplotlib
import numpy as np
import torch
from scene_grasp.scale_shape_grasp_ae.model.auto_encoder_scale_grasp import (
    PointCloudScaleBasedGraspAE,
)
from simnet.lib.net.panoptic_trainer import PanopticModel
from common.utils.nocs_utils import create_input_norm
from common.config.config_dataset_details import (
    get_max_depth_threshold,
    get_gripper_bounds,
)
from common.utils.misc_utils import (
    get_ids_from_seg_output,
    get_position_from_pose,
    get_grasps_from_prediction_np,
    get_gripper_points_lines,
    get_o3d_line_set_vis,
    get_ordered_colors,
)
from scene_grasp.dataset import NOCSDataPoint


class SceneGraspModel:
    def __init__(self, hparams):
        self.min_confidence = hparams.min_confidence
        self.MAX_DEPTH_THRESHOLD = get_max_depth_threshold()
        # Let's load the model here.
        self.model = PanopticModel(hparams, 0, None, None)
        self.model.eval()
        self.model.cuda()
        self.scale_ae = PointCloudScaleBasedGraspAE(
            hparams.scale_ae_emb_dim,
            hparams.scale_ae_num_point,
            choose_bd_sign=hparams.choose_bd_sign,
        )
        self.scale_ae.eval()
        self.scale_ae.load_state_dict(torch.load(hparams.scale_ae_path))
        self.scale_ae.cuda()

    def get_predictions(self, rgb, depth, camera_k):
        left_img = rgb
        far_indices = depth > self.MAX_DEPTH_THRESHOLD
        depth[far_indices] = self.MAX_DEPTH_THRESHOLD
        right_img = depth / 255
        input = create_input_norm(left_img, right_img)
        input = input[None, :, :, :]
        input = input.cuda()
        with torch.no_grad():
            seg_output, _, _, pose_output = self.model.forward(input)
            (
                latent_emb_outputs,
                abs_pose_outputs,
                img_output,
                scores_out,
                output_indices,
            ) = pose_output.compute_pointclouds_and_poses(
                self.min_confidence, is_target=False
            )

            if len(abs_pose_outputs) == 0:
                print("No object found. Continue")
                return None

            emb = torch.tensor(latent_emb_outputs).cuda().to(torch.float)
            scales = [
                abs_pose_outputs[j].scale_matrix[0, 0]
                for j in range(len(abs_pose_outputs))
            ]
            scales = torch.tensor(scales).cuda().to(torch.float).unsqueeze(dim=-1)
            _, endpoints = self.scale_ae(None, scales, emb)

        endpoints = {
            key: value.detach().cpu().numpy() for key, value in endpoints.items()
        }
        pred_xyzs = endpoints["xyz"]
        canonical_pcls = [pred_xyz for pred_xyz in pred_xyzs]
        pred_scale_matrices = np.empty((len(abs_pose_outputs), 4, 4))
        pred_pose_matrices = np.empty((len(abs_pose_outputs), 4, 4))
        for abs_pose_out_ind in range(len(abs_pose_outputs)):
            pred_pose = abs_pose_outputs[abs_pose_out_ind]
            pred_scale_matrices[abs_pose_out_ind, :, :] = pred_pose.scale_matrix
            pred_pose_matrices[abs_pose_out_ind, :, :] = pred_pose.camera_T_object

        pred_class_ids = get_ids_from_seg_output(seg_output, output_indices)
        nocs_dp = NOCSDataPoint(
            rgb=None,
            depth=None,
            camera_k=camera_k,
            seg_masks=None,
            class_ids=pred_class_ids,
            class_confidences=scores_out,
            obj_canonical_pcls=canonical_pcls,
            scale_matrices=pred_scale_matrices,
            pose_matrices=pred_pose_matrices,
            endpoints=endpoints,
            metadata={},
        )
        return nocs_dp


def choose_final_grasps_from_pred_and_pred_sym_grasps(
    pred_grasp_poses_cam,
    pred_grasp_poses_sym_cam,
    pred_pose_matrix,
):
    # Choose what grasp to use out of grasp-pose and grasp-pose-sym:
    center_pos = get_position_from_pose(pred_pose_matrix)
    is_grasp_pose_valid = np.zeros(len(pred_grasp_poses_cam), dtype=bool)
    is_sym_grasp_pose_valid = np.zeros(len(pred_grasp_poses_sym_cam), dtype=bool)
    for pred_pose_ind in range(pred_grasp_poses_cam.shape[0]):
        # FIXME: This can slow down real-time speed by a lot
        pos = get_position_from_pose(pred_grasp_poses_cam[pred_pose_ind])
        dist = np.sqrt(np.sum((pos - center_pos) ** 2))
        pos_sym = get_position_from_pose(pred_grasp_poses_sym_cam[pred_pose_ind])
        dist_sym = np.sqrt(np.sum((pos_sym - center_pos) ** 2))
        if dist < dist_sym:
            is_grasp_pose_valid[pred_pose_ind] = True
        else:
            is_sym_grasp_pose_valid[pred_pose_ind] = True
    assert np.all(np.logical_xor(is_grasp_pose_valid, is_sym_grasp_pose_valid))
    return is_grasp_pose_valid, is_sym_grasp_pose_valid


def get_final_grasps_from_predictions_np(
    gt_scale, endpoints, dp_ind, pred_pose_matrix, TOP_K=None
):
    (
        sorted_indices,
        pred_grasp_poses,
        pred_grasp_widths,
    ) = get_grasps_from_prediction_np(gt_scale, endpoints, dp_ind, TOP_K=TOP_K)
    pred_grasp_poses = pred_grasp_poses[sorted_indices]  # (n, 4, 4)
    pred_grasp_widths = pred_grasp_widths[sorted_indices]
    pred_grasp_poses_cam = pred_pose_matrix @ pred_grasp_poses
    return pred_grasp_poses_cam, pred_grasp_widths, pred_grasp_poses


def get_grasp_vis(grasp_poses_cam, grasp_widths, gripper_vis_colors):
    """
    gripper_vis_colors: np.ndarray (n, 3), values between 0, 1
    """
    GRIPPER_LIMS = get_gripper_bounds()
    GRIPPER_FINGER_LENGTH = GRIPPER_LIMS[0][2]
    gripper_vis_lines = []
    gripper_vis_points = []
    colors = []
    for grasp_ind in range(len(grasp_poses_cam)):
        points, lines = get_gripper_points_lines(
            grasp_widths[grasp_ind],
            GRIPPER_FINGER_LENGTH,
            grasp_poses_cam[grasp_ind],
        )
        lines = np.array(lines) + len(gripper_vis_points)
        gripper_vis_lines += lines.tolist()
        gripper_vis_points += points
        colors += [gripper_vis_colors[grasp_ind]] * len(lines)

    gripper_vis = get_o3d_line_set_vis(gripper_vis_points, gripper_vis_lines, colors)
    return gripper_vis


def get_colored_grasp_vis(grasp_poses, grasp_widths):
    """
    will return grasp vis color-coded by grasp-indices
    color_bgy_indices will be of the order from blue-green-red, blue being grap_indices[0]
    """
    np_colors = get_ordered_colors(
        len(grasp_poses)
    )  # blue-green-red for 0-mid-last indices
    return get_grasp_vis(grasp_poses, grasp_widths, np_colors)
