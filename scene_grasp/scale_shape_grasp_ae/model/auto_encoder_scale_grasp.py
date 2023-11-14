from copy import deepcopy  # type:ignore
import itertools
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.config.config_dataset_details import (
    get_gripper_bounds,
)
from common.utils.misc_utils import (
    get_grasp_pose_at_gripper_origin_pt,
    get_gripper_points_lines,
    get_o3d_line_set_vis,
    get_o3d_pcd_from_np_colors,
    normalize_vectors_batch,
)
from scene_grasp.scale_shape_grasp_ae.model.auto_encoder import PointCloudEncoder


class PointCloudScaleBasedGraspDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts, choose_bd_sign: bool, scale_dimension: int = 1):
        """
        @param choose_bd_sign: If true, decide the sign of baseline-direction
            analytically based on the predicted point-cloud
        """
        super(PointCloudScaleBasedGraspDecoder, self).__init__()
        self.choose_bd_sign = choose_bd_sign
        self.fc1 = nn.Linear(emb_dim + scale_dimension, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.out_dim = (
            3 + 1 + 10 + 3 + 3
        )  # (xyz, grasp success, grasp width bins, z1, z2)
        self.fc3 = nn.Linear(1024, self.out_dim * n_pts)

        self.GRIPPER_BOUNDS_GEN = None
        self.w_T_grs = None
        self.hmg_pts = None

    def forward(self, embedding, scale):
        """
        Args:
            embedding: (B, 512)
            scale: (B, 1)
        """
        embedding_and_scale = torch.cat((embedding, scale), dim=1)
        bs = embedding_and_scale.size()[0]
        out = F.relu(self.fc1(embedding_and_scale))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out_pc = out.view(bs, -1, self.out_dim)

        # Now. Let's process some of these seperately:
        endpoints = {}
        endpoints["xyz"] = out_pc[:, :, :3]
        endpoints["success"] = F.sigmoid(out_pc[:, :, 3])
        endpoints["grasp_width_one_hot"] = F.softmax(out_pc[:, :, 4:14])
        baseline_dir = F.normalize(out_pc[:, :, 14:17], dim=2)
        endpoints["baseline_dir"] = baseline_dir
        approach_dir = out_pc[:, :, 17:]
        approach_dir_head = F.normalize(
            approach_dir
            - ((approach_dir * baseline_dir).sum(dim=2)).unsqueeze(dim=-1)
            * baseline_dir,
            dim=2,
        )
        endpoints["approach_dir"] = approach_dir_head
        if self.choose_bd_sign:
            scaled_xyz = scale.unsqueeze(-1) * endpoints["xyz"]
            if self.GRIPPER_BOUNDS_GEN is None:
                B, N = scaled_xyz.shape[:2]
                device, dtype = scaled_xyz.device, scaled_xyz.dtype
                self.GRIPPER_BOUNDS_GEN = torch.tensor(
                    np.array(get_gripper_bounds()), device=device, dtype=dtype
                )
                self.w_T_grs = torch.empty(
                    (10, N, 4, 4), device=device, dtype=dtype
                )  # world_T_grippers
                self.hmg_pts = torch.empty(
                    (10, 1, 4, N), device=device, dtype=dtype
                )  # Homogeous points
                self.grs_T_w = torch.empty((10, N, 4, 4), device=device, dtype=dtype)

            # Tiny bit slower, but more accurate (About 23 FPS as compared to 30 FPS)
            # endpoints["baseline_dir"] = choose_between_bd_and_sym_bd(
            #     scaled_xyz,
            #     endpoints["approach_dir"],
            #     endpoints["baseline_dir"],
            #     self.GRIPPER_BOUNDS_GEN,
            #     self.w_T_grs,
            #     self.hmg_pts,
            #     self.grs_T_w,
            # )
            endpoints["baseline_dir"] = choose_between_bd_and_sym_bd_fast(
                scaled_xyz,
                endpoints["baseline_dir"],
                self.GRIPPER_BOUNDS_GEN,
            )
        return endpoints


class PointCloudScaleBasedGraspAE(nn.Module):
    def __init__(self, emb_dim, n_pts, choose_bd_sign: bool):
        """
        @param choose_bd_sign: If true, decide the sign of baseline-direction
            analytically based on the predicted point-cloud
        """
        super(PointCloudScaleBasedGraspAE, self).__init__()
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = PointCloudScaleBasedGraspDecoder(emb_dim, n_pts, choose_bd_sign)

    def forward(self, in_pc, scale, emb=None):
        """
        Args:
            in_pc: (B, N, 3)
            emb: (B, 512)
            scale: (B, 1)
        Returns:
            emb: (B, emb_dim)
            out_pc: (B, n_pts, 3)

        """
        if emb is None:
            xyz = in_pc.permute(0, 2, 1)
            emb = self.encoder(xyz)
        endpoints = self.decoder(emb, scale)
        return emb, endpoints

    def get_embedding(self, in_pc):
        xyz = in_pc.permute(0, 2, 1)
        emb = self.encoder(xyz)
        return emb


@torch.no_grad()
def count_number_of_gripper_enclosed_points(
    scaled_xyz: torch.Tensor,  # (B, N, 3)
    ads: torch.Tensor,  # (B, N, 3) unit vectors
    bds: torch.Tensor,  # (B, N, 3) unit vectors
    GRIPPER_BOUNDS_GEN,
    w_T_grs,
    hmg_pts,
    grs_T_w,
) -> torch.Tensor:  # (B, N)
    """Returns number of points inside gripper bound"""
    device = scaled_xyz.device
    dtype = scaled_xyz.dtype
    DEBUG_VIS = False
    MAX_GRASP_WIDTH, FINGER_THICKNESS, FINGER_LENGTH = GRIPPER_BOUNDS_GEN[1, :]

    # Step 1: Compute the transformation from point-cloud to N gripper frames
    B, N = scaled_xyz.shape[:2]
    w_T_grs[:B, :, :3, 0] = ads
    w_T_grs[:B, :, :3, 1] = bds
    ads_cross_bds = torch.cross(ads, bds)  # (B, N, 3)
    w_T_grs[:B, :, :3, 2] = normalize_vectors_batch(ads_cross_bds)
    w_T_grs[:B, :, :3, 3] = scaled_xyz
    # - Assigning a new tensor on gpu is very costly:
    # w_T_grs[:B, :, 3, :] = torch.tensor( [0, 0, 0, 1])  # (B, N, 4) = (4,)
    w_T_grs[:B, :, 3, 0] = 0
    w_T_grs[:B, :, 3, 1] = 0
    w_T_grs[:B, :, 3, 2] = 0
    w_T_grs[:B, :, 3, 3] = 1

    # Step 2: Transform the points to each gripper frame
    # invert matrix (we don't use torch.linalg.inv as it is very slow)
    #   inverse(H(R, P)) = [R_transpose, -1 * R_transpose * P]
    # grs_T_w = torch.empty((B, N, 4, 4), device=device, dtype=dtype)
    w_R_grs = w_T_grs[:B, :, :3, :3]  # rotations: (B, N, 3, 3)
    grs_R_w = w_R_grs.transpose(2, 3)  # (B, N, 3, 3)
    grs_T_w[:B, :, :3, :3] = grs_R_w
    w_P_grs = (
        w_T_grs[:B, :, :3, 3:4] / w_T_grs[:B, :, 3:4, 3:4]
    )  # translations: (B, N, 3, 1)
    grs_T_w[:B, :, :3, 3:4] = torch.matmul(
        -1 * grs_R_w, w_P_grs
    )  # (B N 3 3) @ (B N 3 1)
    grs_T_w[:B, :, 3, 0] = 0
    grs_T_w[:B, :, 3, 1] = 0
    grs_T_w[:B, :, 3, 2] = 0
    grs_T_w[:B, :, 3, 3] = 1

    hmg_pts[:B, 0, :3, :] = scaled_xyz.transpose(1, 2)  # (B, 3, N)
    hmg_pts[:B, :, 3, :] = 1
    grs_P_pts = torch.matmul(
        grs_T_w[:B, ...], hmg_pts[:B, ...]
    )  # (B, N, 4, 4) @ (B, 1, 4, N) -> (B, N, 4, N)
    grs_pts = grs_P_pts[:, :, :3, :] / grs_P_pts[:, :, 3:4, :]  # (B, N, 3, N)

    # Step 3: Compute inside gripper points:
    # - points inside gripper
    # x in [-FL, 0]
    x_inside_indices = torch.logical_and(
        grs_pts[:, :, 0, :] >= -FINGER_LENGTH, grs_pts[:, :, 0, :] <= 0
    )
    # y in [0, MGW]
    y_inside_indices = torch.logical_and(
        grs_pts[:, :, 1, :] >= 0, grs_pts[:, :, 1, :] <= MAX_GRASP_WIDTH
    )
    # z in [-ft/2, ft/2]
    z_inside_indices = torch.logical_and(
        grs_pts[:, :, 2, :] >= -FINGER_THICKNESS / 2,
        grs_pts[:, :, 2, :] <= FINGER_THICKNESS / 2,
    )
    # x and y and z
    # inside_indices: (B, N grippers, N points)
    inside_indices = torch.logical_and(x_inside_indices, y_inside_indices)
    inside_indices = torch.logical_and(inside_indices, z_inside_indices)

    if DEBUG_VIS:
        # convert all required things to numpy
        to_cpu = lambda x: x.cpu().numpy()
        MAX_GRASP_WIDTH = to_cpu(MAX_GRASP_WIDTH)
        FINGER_LENGTH = to_cpu(FINGER_LENGTH)
        FINGER_THICKNESS = to_cpu(FINGER_THICKNESS)
        scaled_xyz = to_cpu(scaled_xyz)
        ads = to_cpu(ads)
        bds = to_cpu(bds)
        w_T_grs = to_cpu(w_T_grs)
        grs_pts = to_cpu(grs_pts)
        inside_indices = to_cpu(inside_indices)

        N_BATCH_VIS = 2
        N_GRASP_VIS = 2

        def get_gripper_vis(point, ad, bd):
            gripper_origin_pt = point + bd * MAX_GRASP_WIDTH / 2
            world_T_grasp = get_grasp_pose_at_gripper_origin_pt(
                gripper_origin_pt, ad, bd
            )
            points, lines = get_gripper_points_lines(
                MAX_GRASP_WIDTH, FINGER_LENGTH, world_T_grasp
            )
            colors = [[1, 0, 0] for _ in range(len(lines))]
            gripper_vis = get_o3d_line_set_vis(points, lines, colors)
            return gripper_vis

        # gripper in gripper frame
        o3d_grp_T_grp = get_gripper_vis(
            np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0])
        )
        # a box to visualize the gripper bounds
        box = o3d.geometry.TriangleMesh.create_box(
            width=FINGER_LENGTH, height=MAX_GRASP_WIDTH, depth=FINGER_THICKNESS
        )
        box_origin_T_box = np.eye(4)
        box_origin_T_box[:3, 3] = [-FINGER_LENGTH, 0, -FINGER_THICKNESS / 2]
        box.transform(box_origin_T_box)

        for b, i in itertools.product(range(N_BATCH_VIS), range(N_GRASP_VIS)):
            # for i in range(3):
            point, ad, bd = scaled_xyz[b, i], ads[b, i], bds[b, i]
            # - visualize original gripper and original point-cloud
            print("- world frame")
            np_colors = np.empty((N, 3))
            np_colors[inside_indices[b, i]] = [0, 0, 1]
            np_colors[~inside_indices[b, i]] = [1, 0, 0]
            o3d_w_T_points = get_o3d_pcd_from_np_colors(scaled_xyz[b], np_colors)
            o3d_world_T_grp = get_gripper_vis(point, ad, bd)
            w_T_grp_box = deepcopy(box)
            w_T_grp_box.transform(w_T_grs[b, i])
            o3d.visualization.draw(  # type:ignore
                [o3d_w_T_points, o3d_world_T_grp, w_T_grp_box]
            )
            # - visualize new gripper and transformed point-cloud
            print("- gripper frame")
            o3d_grp_T_points = get_o3d_pcd_from_np_colors(grs_pts[b, i].T, np_colors)
            o3d.visualization.draw(  # type:ignore
                [o3d_grp_T_points, o3d_grp_T_grp, box]
            )
        # convert back to torch
        inside_indices = torch.tensor(inside_indices, device=device, dtype=dtype)
    return torch.sum(inside_indices, dim=-1)


def choose_between_bd_and_sym_bd(
    scaled_xyz: torch.Tensor,  # (B, N, 3)
    ads: torch.Tensor,  # (B, N, 3) unit vectors
    bds: torch.Tensor,  # (B, N, 3) unit vectors
    GRIPPER_BOUNDS_GEN,
    w_T_grs,
    hmg_pts,
    grs_T_w,
):
    """Note: bds is edited in this function call"""
    bds_counts = count_number_of_gripper_enclosed_points(
        scaled_xyz, ads, bds, GRIPPER_BOUNDS_GEN, w_T_grs, hmg_pts, grs_T_w
    )
    sym_bds_counts = count_number_of_gripper_enclosed_points(
        scaled_xyz, ads, -1 * bds, GRIPPER_BOUNDS_GEN, w_T_grs, hmg_pts, grs_T_w
    )
    bds[sym_bds_counts > bds_counts] *= -1
    return bds


def choose_between_bd_and_sym_bd_fast(
    scaled_xyz: torch.Tensor,  # (B, N, 3)
    bds: torch.Tensor,  # (B, N, 3) unit vectors
    GRIPPER_BOUNDS_GEN,
):
    MAX_GRASP_WIDTH = GRIPPER_BOUNDS_GEN[1, 0]
    gripper_centers = scaled_xyz + bds * MAX_GRASP_WIDTH / 2
    gripper_centers_sym = scaled_xyz - bds * MAX_GRASP_WIDTH / 2
    dist_center = torch.linalg.norm(gripper_centers, dim=-1)
    dist_center_sym = torch.linalg.norm(gripper_centers_sym, dim=-1)
    bds[dist_center > dist_center_sym] *= -1
    return bds
