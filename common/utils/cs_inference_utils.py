import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import open3d as o3d
from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.models.auto_encoder import PointCloudAE
from utils.nocs_utils import load_img_NOCS, create_input_norm, convert_mm_depth_cs_depth
from utils.viz_utils import depth2inv, viz_inv_depth
from utils.transform_utils import (
    get_gt_pointclouds,
    transform_coordinates_3d,
    calculate_2d_projections,
)
from utils.transform_utils import project
from utils.viz_utils import save_projected_points, draw_bboxes, line_set_mesh
from scripts_rico.utils_rico import YCBVDataPoint


def get_cs_original_auto_encoder(model_path):
    emb_dim = 128
    n_pts = 2048
    ae = PointCloudAE(emb_dim, n_pts)
    ae.cuda()
    ae.load_state_dict(torch.load(model_path))
    ae.eval()
    return ae


def get_cs_model_from_args_file_path(
    args_file_path, checkpoint=None, ae_checkpoint=None
):
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    common.add_train_args(parser)
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    args_list = open(args_file_path, "r").read().split()
    hparams = parser.parse_args(args_list)
    if checkpoint is not None:
        hparams.checkpoint = checkpoint
    model = PanopticModel(hparams, 0, None, None)
    model.eval()
    model.cuda()

    if ae_checkpoint is not None:
        auto_encoder_path = ae_checkpoint
    else:
        auto_encoder_path = hparams.ae_checkpoint
    ae = get_cs_original_auto_encoder(auto_encoder_path)
    return model, ae


def get_cs_original_model():
    args_file_path = "centersnap/configs/centersnap_original_model_inference.txt"
    return get_cs_model_from_args_file_path(args_file_path)


class CS_Inference:
    def __init__(self, cs_model, cs_ae, min_confidence=0.5):
        self.cs_model = cs_model
        self.cs_ae = cs_ae
        self.min_confidence = min_confidence

    @classmethod
    def from_original_model(cls):
        cs_model, cs_ae = get_cs_original_model()
        return CS_Inference(cs_model, cs_ae)

    def save_predictions(
        self,
        color,
        depth_mm,
        camera_k,
        output_path,
        output_prefix,
        visualize_preds: bool = False,
        scale_depth: bool = False,
    ):
        dump_paths = {}
        orig_color_bgr = np.copy(color)
        orig_depth_mm = np.copy(depth_mm)

        scale_ratio = 1
        if scale_depth:
            scale_ratio = 4.5e3 / depth_mm.max()
            depth_mm = depth_mm * scale_ratio

        img_vis = np.copy(color)
        right_img = convert_mm_depth_cs_depth(depth_mm)
        input = create_input_norm(color, right_img)

        input = input[None, :, :, :]
        input = input.to(torch.device("cuda:0"))

        with torch.no_grad():
            _, _, _, pose_output = self.cs_model.forward(input)
            (
                latent_emb_outputs,
                abs_pose_outputs,
                img_output,
                _,
                _,
            ) = pose_output.compute_pointclouds_and_poses(
                self.min_confidence, is_target=False
            )

        rgb_save_path = str(output_path / f"{output_prefix}_image.png")
        cv2.imwrite(rgb_save_path, np.copy(np.copy(img_vis)))
        dump_paths["input_rgb"] = rgb_save_path

        peaks_save_path = str(output_path / f"{output_prefix}_peaks_output.png")
        cv2.imwrite(peaks_save_path, np.copy(img_output))
        dump_paths["pred_peaks"] = peaks_save_path

        depth_vis = depth2inv(torch.tensor(right_img).unsqueeze(0).unsqueeze(0))
        depth_vis = viz_inv_depth(depth_vis)
        depth_vis = depth_vis * 255.0
        depth_save_path = str(output_path / f"{output_prefix}_depth_vis.png")
        cv2.imwrite(depth_save_path, np.copy(depth_vis))
        dump_paths["input_depth"] = depth_save_path

        write_pcd = True
        cam_pcls = []
        points_2d = []
        box_obb = []
        axes = []

        for j in range(len(latent_emb_outputs)):
            emb = latent_emb_outputs[j]
            emb = torch.FloatTensor(emb).unsqueeze(0)
            emb = emb.cuda()
            _, shape_out = self.cs_ae(None, emb)
            shape_out = shape_out.cpu().detach().numpy()[0]

            rotated_pc, rotated_box, _ = get_gt_pointclouds(
                abs_pose_outputs[j], shape_out, camera_model=None
            )
            if scale_depth:
                rotated_pc = rotated_pc / scale_ratio
            cam_pcls.append(rotated_pc)
            if write_pcd:
                ply_path = output_path / f"{output_prefix}_obj_{j}_cam_pcl.ply"
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(rotated_pc)
                o3d.io.write_point_cloud(str(ply_path), pcd)

            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #     size=0.1, origin=[0, 0, 0]
            # )
            # T = abs_pose_outputs[j].camera_T_object
            # mesh_frame = mesh_frame.transform(T)
            # cam_pcls.append(mesh_frame)
            # cylinder_segments = line_set_mesh(rotated_box)
            # for k in range(len(cylinder_segments)):
            #     cam_pcls.append(cylinder_segments[k])

            points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
            points_2d_mesh = project(camera_k, points_mesh)
            points_2d_mesh = points_2d_mesh.T
            points_2d.append(points_2d_mesh)
            # 2D output
            points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
            points_2d_obb = project(camera_k, points_obb)
            points_2d_obb = points_2d_obb.T
            box_obb.append(points_2d_obb)
            xyz_axis = (
                0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            )
            sRT = abs_pose_outputs[j].camera_T_object @ abs_pose_outputs[j].scale_matrix
            transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
            projected_axes = calculate_2d_projections(
                transformed_axes, camera_k[:3, :3]
            )
            axes.append(projected_axes)

        # o3d.visualization.draw_geometries(rotated_pcds)  # type:ignore
        proj_pts_save_path = str(output_path / f"{output_prefix}_projections.png")
        save_projected_points(np.copy(img_vis), points_2d, proj_pts_save_path)
        dump_paths["pred_proj"] = proj_pts_save_path

        colors_box = [(63, 237, 234)]
        im = np.array(np.copy(img_vis)).copy()
        for k in range(len(colors_box)):
            for points_2d, axis in zip(box_obb, axes):
                points_2d = np.array(points_2d)
                im = draw_bboxes(im, points_2d, axis, colors_box[k])
        box_save_path = str(output_path / f"{output_prefix}_bbox3d.png")
        cv2.imwrite(box_save_path, np.copy(im))
        dump_paths["pred_bboxes"] = box_save_path

        if visualize_preds:
            YCBVDataPoint.visualize_from_data(
                orig_color_bgr, orig_depth_mm, camera_k, cam_pcls
            )

        print("done with image: ", output_path, output_prefix)
        return dump_paths
