import argparse
from copy import deepcopy
from datetime import datetime
import open3d as o3d
import os
from imageio import get_writer
import matplotlib as mpl
import numpy as np
from pygifsicle import optimize
import pyrender
import torch
from simnet.lib.net.common import add_train_args
from common.config import config_dataset_details
from common.utils.trimesh_utils import get_revolving_cams


def save_images_as_gif(images, gif_path):
    if len(images) > 0:
        with get_writer(gif_path, mode="I") as writer:
            for image in images:
                writer.append_data(image)
        optimize(str(gif_path))


def visualize_list_xyz_colors_as_gif(list_xyz, list_colors, gif_path):
    # Setup renderer as EGL
    # - See https://pyrender.readthedocs.io/en/latest/examples/offscreen.html?highlight=backend#choosing-a-backend
    # - I found out that point_size is not reflected properly with the default backend
    # - hence I use egl.
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Create scene:
    scene = pyrender.Scene()
    for xyz, colors in zip(list_xyz, list_colors):
        mesh = pyrender.Mesh.from_points(xyz, colors=colors)
        scene.add(mesh)
    # - light
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=3.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    light_node = None
    # - cameras
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    cam_node = None
    cam_poses = get_revolving_cams()
    # - renderings
    r = pyrender.OffscreenRenderer(400, 400, point_size=5.0)
    images = []
    for cam_pose in cam_poses:
        if cam_node is not None:
            scene.remove_node(cam_node)
        cam_node = scene.add(camera, pose=cam_pose)

        if light_node is not None:
            scene.remove_node(light_node)
        light_node = scene.add(light, pose=cam_pose)
        color, _ = r.render(scene)
        images.append(color)
    # - save images
    save_images_as_gif(images, gif_path)


def get_uniform_colors_pyrender(num_points, color_rgba_255):
    """eg. color_rgba_255: [255, 0, 0, 255]"""
    colors = np.zeros((num_points, 4), dtype=np.uint8)
    colors = color_rgba_255
    return colors


def get_o3d_pcd_from_np(np_pcd, color=None):
    """
    np_pcd: nx3
    color: [r,g,b] in range [0,1]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def get_o3d_pcd_from_np_colors(np_pcd, np_colors):
    pcd = get_o3d_pcd_from_np(np_pcd)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    return pcd


def get_o3d_pcd_from_np_color_dict(np_pcd, list_color_indices):
    """
    np_pcd: nx3
    list_color_indices: [(color, [idx1,idx2...]), ...]
        color should be like [r,g,b] with values ranging in [0, 1]
    """
    colors = np.zeros_like(np_pcd, dtype=float)
    for color, pcd_indices in list_color_indices:
        colors[pcd_indices] = color
    pcd = get_o3d_pcd_from_np_colors(np_pcd, colors)
    return pcd


def get_o3d_line_set_vis(points, lines, colors=None):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if colors is not None:
        line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_ordered_colors(num_colors, cmap_name="jet"):
    """
    @return <num_colors> colors from the cmap. (num_colors, 3) in range [0,1]
    For "jet" cmap 0 index is blue and last index is red
    """
    cmap = mpl.colormaps[cmap_name]  # blue - green - red
    lin_colors = cmap(np.arange(num_colors) / num_colors)
    lin_colors = lin_colors[:, :3]
    return lin_colors


def visualize_datapoint_o3d(
    xyz,
    scale,
    gs,
    ads,
    bds,
    widths,
    gripper_finger_length,
    num_success_grasp_vis: int = None,
    show_ad_bd: bool = False,
):
    APP_HEIGHT = 0.03  # Size of (approach/baseline) direction vector
    AD_COLOR = [0, 0, 1]
    BD_COLOR = [128 / 255, 0, 0]
    WIDTH_COLOR = [0, 0, 0]

    xyz_scaled = xyz * scale
    is_success = gs > 0.5
    np_colors = np.empty((xyz_scaled.shape[0], 3))
    np_colors[is_success] = [0, 1, 0]
    np_colors[~is_success] = [1, 0, 0]
    o3d_xyz_scaled = get_o3d_pcd_from_np_colors(xyz_scaled, np_colors)

    success_indices = np.where(is_success)[0]

    if num_success_grasp_vis is not None:
        success_indices = np.random.choice(
            success_indices,
            size=min(len(success_indices), num_success_grasp_vis),
            replace=False,
        )
    ad_vis_points = []
    ad_vis_lines = []
    ad_vis_colors = []
    bd_vis_points = []
    bd_vis_lines = []
    bd_vis_colors = []
    width_vis_points = []
    width_vis_lines = []
    width_vis_colors = []
    gripper_vis_points = []
    gripper_vis_lines = []
    gripper_vis_colors = []
    for success_index in success_indices:
        point = xyz_scaled[success_index]
        ad = ads[success_index]
        ad_start = point
        ad_end = ad_start + APP_HEIGHT * ad
        ad_vis_lines += [[len(ad_vis_points), len(ad_vis_points) + 1]]
        ad_vis_points += [ad_start, ad_end]
        ad_vis_colors.append(AD_COLOR)

        bd = bds[success_index]
        bd_start = point
        bd_end = bd_start + APP_HEIGHT * bd
        bd_vis_lines += [[len(bd_vis_points), len(bd_vis_points) + 1]]
        bd_vis_points += [bd_start, bd_end]
        bd_vis_colors.append(BD_COLOR)

        # The width will start from the start point.
        # Will go in the direction of baseline direction by grasp-width amount
        if widths is not None:
            width = widths[success_index]
            width_vis_start = point
            width_vis_end = width_vis_start + width * bd
            width_vis_lines += [[len(width_vis_points), len(width_vis_points) + 1]]
            width_vis_points += [width_vis_start, width_vis_end]
            width_vis_colors.append(WIDTH_COLOR)

            world_T_grasp, adj_width = get_grasp_pose(point, ad, bd, width)
            points, lines = get_gripper_points_lines(
                adj_width, gripper_finger_length, world_T_grasp
            )
            lines = np.array(lines) + len(gripper_vis_points)
            lines = lines.tolist()
            # Ideally I want color based on grasp-success confidence
            gripper_vis_points += points
            gripper_vis_lines += lines
            gripper_vis_colors += [
                np_colors[success_index].tolist() for _ in range(len(lines))
            ]

        # normal = normal_dirs[success_index]
        # normal_start = xyz_scaled[success_index]
        # normal_end = normal_start + APP_HEIGHT * normal
        # normal_vis_lines += [
        #     [len(normal_vis_points), len(normal_vis_points) + 1]
        # ]
        # normal_vis_points += [normal_start, normal_end]
        # normal_vis_colors.append([1,0,1])

    things_to_vis = [o3d_xyz_scaled]
    if show_ad_bd:
        ad_vis = get_o3d_line_set_vis(ad_vis_points, ad_vis_lines, ad_vis_colors)
        things_to_vis.append(ad_vis)
        bd_vis = get_o3d_line_set_vis(bd_vis_points, bd_vis_lines, bd_vis_colors)
        things_to_vis.append(bd_vis)

    if widths is not None:
        width_vis = get_o3d_line_set_vis(
            width_vis_points, width_vis_lines, width_vis_colors
        )
        things_to_vis.append(width_vis)
        gripper_vis = get_o3d_line_set_vis(
            gripper_vis_points, gripper_vis_lines, gripper_vis_colors
        )
        things_to_vis.append(gripper_vis)
    # normal_vis = get_o3d_line_set_vis(
    #     normal_vis_points, normal_vis_lines, normal_vis_colors
    # )
    # things_to_vis.append(normal_vis)
    o3d.visualization.draw(things_to_vis)


def get_gripper_points_lines(
    width: float, finger_length: float, world_T_grasp: np.ndarray
):
    points = [
        [width / 2, 0, 0],
        [width / 2, 0, -finger_length],
        [0, 0, -finger_length],
        [-width / 2, 0, -finger_length],
        [-width / 2, 0, 0],
        [0, 0, -2 * finger_length],
    ]
    lines = [[0, 1], [1, 3], [3, 4], [2, 5]]
    points = np.array(points)
    grasp_T_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T
    world_T_points = world_T_grasp @ grasp_T_points
    world_points = world_T_points[:3, :] / world_T_points[3, :]
    world_points = (world_points.T).tolist()
    return world_points, lines


def get_o3d_vector_vis(vector: np.ndarray, point: np.ndarray, length: float):
    """
    @param vector: (3,)
    @param point: (3, )
    @param length: length of the vector
    @return return o3d arrow of length <length> starting from <point> and pointing in
    the <vector> direction
    """
    # o3d.visualization.draw
    o3d_vector = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.1 * length,
        cone_radius=0.2 * length,
        cylinder_height=0.8 * length,
        cone_height=0.2 * length,
    )
    transformation = np.eye(4)
    transformation[:3, 3] = point
    vector_z = normalize_vector(vector)
    up_dir = np.array([0, 1, 0])
    if not np.isclose(np.dot(vector_z, up_dir), 0):
        vector_y = normalize_vector(np.cross(vector_z, up_dir))
    else:
        vector_y = np.array([1, 0, 0])
    vector_x = normalize_vector(np.cross(vector_y, vector_z))
    for i, v in enumerate([vector_x, vector_y, vector_z]):
        transformation[:3, i] = v
    o3d_vector.transform(transformation)
    return o3d_vector


def normalize_vector(x: np.ndarray):
    """x: (3,)"""
    norm = np.linalg.norm(x)
    return x / norm if not np.isclose(norm, 0) else x


def normalize_vectors(xs: np.ndarray):
    """x: (n, 3)"""
    norms = np.linalg.norm(xs, axis=-1)  # (n)
    norms[np.isclose(norms, 0)] = 1
    norms = norms[:, None]  # (n, 1)
    return xs / norms  # (n, 3)


def normalize_vectors_batch(xs: torch.Tensor) -> torch.Tensor:  # (B, N, 3)
    norms = torch.linalg.norm(xs, dim=-1)  # (B, N)
    norms[norms == 0] = 1
    norms = norms.unsqueeze(dim=-1)  # (B, N, 1)
    return xs / norms


def get_grasp_pose_at_gripper_origin_pt(origin_pt, ad, bd):
    grasp_z = normalize_vector(ad)
    grasp_x = normalize_vector(-1 * bd)
    grasp_y = normalize_vector(np.cross(grasp_z, grasp_x))
    world_T_grasp = np.eye(4)
    world_T_grasp[:3, 0] = grasp_x
    world_T_grasp[:3, 1] = grasp_y
    world_T_grasp[:3, 2] = grasp_z
    world_T_grasp[:3, 3] = origin_pt
    return world_T_grasp


def get_grasp_pose(point, ad, bd, width):
    POINT_BD_THRESHOLD = config_dataset_details.get_point_bd_distance()
    adj_point = point - bd * POINT_BD_THRESHOLD
    adj_width = width + 2 * POINT_BD_THRESHOLD
    gripper_origin_pt = adj_point + bd * adj_width / 2
    world_T_grasp = get_grasp_pose_at_gripper_origin_pt(gripper_origin_pt, ad, bd)
    return world_T_grasp, adj_width


def get_ads_bds_from_grasp_poses(grasp_poses: np.ndarray):
    """
    grasp_pose: nx4x4
    returns ads (n, 3), bds (n, 3,)
    """
    # ads = grasp_poses @ np.array([[0, 0, 1, 1]]).T  # nx4x1
    # ads = normalize_vectors(ads[..., :3, 0] / ads[..., 3:4, 0])  # nx3
    # bds = grasp_poses @ np.array([[-1, 0, 0, 1]]).T
    # bds = normalize_vectors(bds[..., :3, 0] / bds[..., 3:4, 0])
    ads = grasp_poses[:, :3, :3] @ np.array([[0, 0, 1]]).T  # nx3x1
    ads = normalize_vectors(ads[..., 0])  # nx3
    bds = grasp_poses[:, :3, :3] @ np.array([[-1, 0, 0]]).T
    bds = normalize_vectors(bds[..., 0])
    return ads, bds


def get_position_from_pose(pose):
    return pose[:3, 3] / pose[3, 3]


def get_positions_from_poses(poses):
    """
    poses: nx4x4
    returns: n,3
    """
    return poses[..., :3, 3] / poses[..., 3:4, 3]


def geom_grasp_check(
    o3dpcd_frameF_collisionobj,
    T_frameF_grasp,
    gripper_lims,
    antipodal_threshold=0.5,
    PTS_IN_GRASP_THRESHOLD=10,
):
    lim_X, lim_Y, lim_Z = gripper_lims
    # lim_Z = 0.045
    ### Transforming object pcd into the gripper/grasp frame
    # T_grasp_frameF = T_frameF_grasp.inverse()
    T_grasp_frameF = np.linalg.inv(T_frameF_grasp)
    o3dpcd_grasp_obj = deepcopy(o3dpcd_frameF_collisionobj)
    o3dpcd_grasp_obj.transform(T_grasp_frameF)
    o3dpcd_grasp_obj.paint_uniform_color([0.7, 0.7, 0.7])

    crop_grasp_slice = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-0.060, -0.035, -0.075),
        max_bound=(0.060, 0.035, 0.005),
    )

    cropped_o3dpcd_grasp_obj = o3dpcd_grasp_obj.crop(crop_grasp_slice)
    cropped_o3dpcd_grasp_obj.paint_uniform_color([1, 0, 0])

    pcd_grasp_obj_list = (np.asarray(cropped_o3dpcd_grasp_obj.points)).tolist()
    normals_grasp_obj_list = (np.asarray(cropped_o3dpcd_grasp_obj.normals)).tolist()

    grasp_collision_free = 1
    pts_in_grasp = 0
    pts_on_exitpjface = 0
    pcdX_in_grasp = None
    pcd_in_grasp = None
    rejection_reason = None
    for i, pt_in_pcd in enumerate(pcd_grasp_obj_list):
        ### collision if a point in is the Y slice and
        ### (less than Z limit or outside X limits)
        collision_first_cond = (abs(pt_in_pcd[1]) < lim_Y / 2.0) and (
            (abs(pt_in_pcd[0]) < lim_X / 2.0 and pt_in_pcd[2] < -lim_Z)
            or (abs(pt_in_pcd[0]) > lim_X / 2.0 and -lim_Z < pt_in_pcd[2] < 0)
        )
        collision_second_cond = pt_in_pcd[2] < -lim_Z and abs(pt_in_pcd[1]) < 2 * lim_Y
        if collision_first_cond or collision_second_cond:
            # the second condition is added to prevent tilted grasps
            grasp_collision_free = 0  # grasp is NOT collision free
            if collision_first_cond and (not collision_second_cond):
                rejection_reason = "collision_first"
            elif (not collision_first_cond) and collision_second_cond:
                rejection_reason = "collision_second"
            else:
                rejection_reason = "collision_both"
            # print("Collision deteceted")
            break

        # We use griper lims for collision check,
        # but use hard coded gripper bounds for checking points inside gripper
        if (
            (abs(pt_in_pcd[1]) < lim_Y / 2.0)
            and (abs(pt_in_pcd[0]) < lim_X / 2.0)
            and (pt_in_pcd[2] < 0 and pt_in_pcd[2] > -lim_Z)
        ):
            ### Check if normal at this point is appostite to grasp X axis
            # We are working in grasp frame, so the normals should be close to [-1, 0 ,0]
            #
            # so we will say that if norm(y,z) components of the normal are almost 0
            # and x component is negative then it's good
            # print(normals_grasp_obj_list[i][1:3])

            # smaller antipodal_threshold value will enforce the parallel-jaw constraint more strictly
            # antipodal_condition = (
            #     np.linalg.norm(np.asarray(normals_grasp_obj_list[i][1:3]))
            #     < antipodal_threshold
            # )  # 0.65
            if True:
                pts_in_grasp += 1
                if pcdX_in_grasp is None:
                    pcdX_in_grasp = [pt_in_pcd[0]]
                    pcd_in_grasp = [pt_in_pcd]
                else:
                    pcdX_in_grasp.extend([pt_in_pcd[0]])
                    pcd_in_grasp.extend([pt_in_pcd])

                if normals_grasp_obj_list[i][0] < 0:
                    pts_on_exitpjface += 1

    # grasp is collision free and there are some min pts in grasp
    if (
        # pts_in_grasp - pts_on_exitpjface > 5
        # and pts_on_exitpjface > 5
        pts_in_grasp > PTS_IN_GRASP_THRESHOLD
        and grasp_collision_free == 1
    ):
        good_grasp = True
        grasp_width = abs(max(pcdX_in_grasp) - min(pcdX_in_grasp))
    else:
        # print("pts_in_grasp=", pts_in_grasp)
        # print("pts_on_exitpjface=", pts_on_exitpjface)
        good_grasp = False
        grasp_width = 0
        if rejection_reason is None:
            rejection_reason = f"less_points_{pts_in_grasp}"
        # if grasp_collision_free == 1:
        #     viz_grasp_crop_pcd(T_frameF_grasp, o3dpcd_frameF_collisionobj)

    return good_grasp, grasp_width, rejection_reason


def get_grasps_from_prediction_np(gt_scale, endpoints, dp_ind, TOP_K=None):
    GRIPPER_OFFSET_BINS = config_dataset_details.get_gripper_offset_bins()
    GRASP_SUCCESS_THRESHOLD = config_dataset_details.get_grasp_success_threshold()

    pred_xyz = endpoints["xyz"][dp_ind]
    pred_xyz_scaled = pred_xyz * gt_scale
    pred_gs = endpoints["success"][dp_ind]
    pred_ad = endpoints["approach_dir"][dp_ind]
    pred_bd = endpoints["baseline_dir"][dp_ind]
    pred_gwoh = endpoints["grasp_width_one_hot"][dp_ind]
    pred_gwv = config_dataset_details.convert_gwoh_to_gwv_batch(
        np.expand_dims(pred_gwoh, axis=0), GRIPPER_OFFSET_BINS
    )[0]
    num_points = pred_xyz.shape[0]

    sorted_indices = np.argsort(pred_gs)
    cutoff_index = np.searchsorted(pred_gs[sorted_indices], GRASP_SUCCESS_THRESHOLD)
    sorted_indices = sorted_indices[cutoff_index:][::-1]
    if TOP_K is not None:
        sorted_indices = sorted_indices[:TOP_K]

    # Basically, now estimate the grasp poses for these points. That's it.
    pred_grasp_poses = np.ones((num_points, 4, 4)) * np.nan
    pred_grasp_widths = np.ones((num_points)) * np.nan
    for point_ind in sorted_indices:
        point = pred_xyz_scaled[point_ind]
        ad = pred_ad[point_ind]
        bd = pred_bd[point_ind]
        width = pred_gwv[point_ind]
        world_T_grasp, adj_width = get_grasp_pose(point, ad, bd, width)
        pred_grasp_poses[point_ind] = world_T_grasp
        pred_grasp_widths[point_ind] = adj_width

    return (sorted_indices, pred_grasp_poses, pred_grasp_widths)


def compute_o3d_normals(o3d_pcl):
    o3d_pcl.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=20)
    )
    o3d_pcl.orient_normals_consistent_tangent_plane(5)
    return o3d_pcl


def get_aligned_ids_nocs(
    mrcnn_masks,
    output_indices,
    class_ids,
    class_ids_predicted,
    scores,
    scores_predicted,
    depth,
):
    mask_out = []
    for p in range(mrcnn_masks.shape[-1]):
        mask = np.logical_and(mrcnn_masks[:, :, p], depth > 0)
        mask_out.append(mask)
    mask_out = np.array(mask_out)
    index_centers = []
    for m in range(mask_out.shape[0]):
        pos = np.where(mask_out[m, :, :])
        center_x = np.average(pos[0])
        center_y = np.average(pos[1])
        index_centers.append([center_x, center_y])
    new_masks = []
    new_ids = []
    new_scores = []
    index_centers = np.array(index_centers)
    if np.any(np.isnan(index_centers)):
        index_centers = index_centers[~np.any(np.isnan(index_centers), axis=1)]
    mask_out = np.array(mask_out)
    for l in range(len(output_indices)):
        point = output_indices[l]
        if len(output_indices) == 0:
            continue
        distances = np.linalg.norm(index_centers - point, axis=1)
        min_index = np.argmin(distances)
        if distances[min_index] < 28:
            new_masks.append(mask_out[min_index, :, :])
            new_ids.append(class_ids[min_index])
            new_scores.append(scores[min_index])
        else:
            new_masks.append(None)
            new_ids.append(class_ids_predicted[l])
            new_scores.append(scores_predicted[l])
    masks = np.array(new_masks)
    class_ids = np.array(new_ids)
    scores = np.array(new_scores)
    return masks, class_ids, scores


def get_ids_from_seg_pred(seg_pred, output_indices):
    category_seg_output = np.ascontiguousarray(seg_pred)
    category_seg_output = np.argmax(category_seg_output[0], axis=0)
    class_ids_predicted = []
    for k in range(len(output_indices)):
        center = output_indices[k]
        class_ids_predicted.append(category_seg_output[center[0], center[1]])
    return class_ids_predicted


def get_ids_from_seg_output(seg_output, output_indices):
    return get_ids_from_seg_pred(seg_output.seg_pred.cpu().numpy(), output_indices)


def get_o3d_chamfer_distance(o3d_pcl1, o3d_pcl2):
    def one_way_chamfer(src, target):
        dists = src.compute_point_cloud_distance(target)
        dists = (np.asarray(dists) ** 2).mean()
        return dists

    dist1 = one_way_chamfer(o3d_pcl1, o3d_pcl2)
    dist2 = one_way_chamfer(o3d_pcl2, o3d_pcl1)
    return dist1 + dist2


def np_pcl_to_homo(np_pcl):
    # np_pcl: nx3
    return np.concatenate((np_pcl, np.ones((np_pcl.shape[0], 1))), axis=1).T


def np_homo_pcl_to_pcl(homo_pcl):
    # homo_pcl: 4xn
    homo_pcl[:3, :] /= homo_pcl[3, :]
    return homo_pcl[:3, :].T


def get_scale_matrix_from_scalar_scale(scale):
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    return scale_matrix


def transform_pcl(np_pcl, pose_matrix, scale_matrix):
    pcl_homo = np_pcl_to_homo(np_pcl)
    trans_pcl_homo = pose_matrix @ scale_matrix @ pcl_homo
    trans_pcl = np_homo_pcl_to_pcl(trans_pcl_homo)
    return trans_pcl


def evaluate_grasp(o3d_pcl, grasp_pose, grasp_width, grasp_pose_sym):
    GRIPPER_LIMS = config_dataset_details.get_gripper_bounds()
    GRIPPER_WIDTH_TOL = config_dataset_details.get_gripper_width_tolerance()
    PTS_IN_GRASP_THRESHOLD = config_dataset_details.get_points_in_grasp_theshold()

    gripper_lims = np.copy(GRIPPER_LIMS)[0]
    adj_grasp_width = grasp_width + GRIPPER_WIDTH_TOL
    gripper_lims[0] = adj_grasp_width
    est_grasp_success, _, first_rej_reason = geom_grasp_check(
        o3d_pcl,
        grasp_pose,
        gripper_lims,
        antipodal_threshold=1,
        PTS_IN_GRASP_THRESHOLD=PTS_IN_GRASP_THRESHOLD,
    )
    second_rej_reason = None
    if not est_grasp_success and grasp_pose_sym is not None:
        est_grasp_success, _, second_rej_reason = geom_grasp_check(
            o3d_pcl,
            grasp_pose_sym,
            gripper_lims,
            antipodal_threshold=1,
            PTS_IN_GRASP_THRESHOLD=PTS_IN_GRASP_THRESHOLD,
        )
    return est_grasp_success, adj_grasp_width, first_rej_reason, second_rej_reason


def convert_realsense_rgb_depth_to_o3d_pcl(
    rgb: np.ndarray, depth: np.ndarray, color_camera_k: np.ndarray
):  # -> o3d.geometry.PointCloud:
    """
    # NOTE: Assumes that the realsense values (depth, camera_k) is in meters.
    @param rgb (H, W, 3): RGB image from realsense
    @param depth (H, W, 1): Depth image from realsense (in meters)
    @param color_camera_k (3, 3): Color Camera intrinsics from realsense (in meters, i.e. unchanged)
    @return: Open3D point cloud
    """
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1, convert_rgb_to_intensity=False, depth_trunc=2
    )
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d_rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=float(color_camera_k[0, 0]),
            fy=float(color_camera_k[1, 1]),
            cx=float(color_camera_k[0, 2]),
            cy=float(color_camera_k[1, 2]),
        ),
    )
    return o3d_pcd


def convert_realsense_rgb_mm_depth_to_o3d_pcl(
    rgb: np.ndarray, depth: np.ndarray, color_camera_k: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    # NOTE: Assumes that the realsense values (depth, camera_k) are in mm.
    @param rgb (H, W, 3): RGB image from realsense
    @param depth (H, W, 1): Depth image from realsense (in mm, i.e. unchanged)
    @param color_camera_k (3, 3): Color Camera intrinsics from realsense (in mm, i.e. unchanged)
    @return: Open3D point cloud
    """
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1000, convert_rgb_to_intensity=False
    )
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d_rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=float(color_camera_k[0, 0]),
            fy=float(color_camera_k[1, 1]),
            cx=float(color_camera_k[0, 2]),
            cy=float(color_camera_k[1, 2]),
        ),
    )
    return o3d_pcd


def adjust_cgn_grasp_pose_to_our_grasp_pose(cgn_grasp_poses):
    """
    cgn_grasp_poses: Nx(4x4)
    """
    cgn_T_our = config_dataset_details.get_our_gripper_origin_in_cgn_frame()
    world_T_cgn = cgn_grasp_poses
    world_T_ours = world_T_cgn @ cgn_T_our
    return world_T_ours


def get_scene_grasp_model_params(args_list=None):
    if args_list is None:
        args_list = [
            "@configs/SceneGraspNetInference.txt",
            "--checkpoint",
            "checkpoints/scene_grasp.ckpt",
            "--scale_ae_path",
            "checkpoints/scale_ae.pth",
        ]
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_train_args(parser)
    app_group = parser.add_argument_group("app")
    app_group.add_argument("--min_confidence", type=float, default=0.5)
    hparams = parser.parse_args(args_list)
    return hparams


def get_angle_between_two_n_vectors(vec1s: np.ndarray, vec2s: np.ndarray):
    """
    vec1s: np.ndarray (n, 3)
    vec2s: np.ndarray (n, 3)
    """
    angle_radians = np.arccos(
        (vec1s * vec2s).sum(axis=-1)
        / (np.linalg.norm(vec1s, axis=-1) * np.linalg.norm(vec2s, axis=-1))
    )
    return angle_radians


def get_data_dump_stamp():
    return datetime.today().strftime("%Y-%m-%d--%H-%M-%S")
