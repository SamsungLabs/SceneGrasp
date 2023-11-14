# from IPython import embed
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool, set_start_method
import numpy as np
import open3d as o3d
import pickle
from random import shuffle
from tqdm import tqdm
from common.config.config_global_paths import PROJECT_DATA_ROOT, NOCS_ROOT
from common.config.config_dataset_details import (
    get_grasp_dataset_num_scales,
    get_gripper_bounds,
    get_gripper_offset_bins,
    get_nocs_information_dicts,
    get_num_cpus,
    get_n_points,
)
from common.utils.misc_utils import get_data_dump_stamp, get_ads_bds_from_grasp_poses
from scene_grasp.scale_shape_grasp_ae.grasp_gen_tool import grasp_map_utils
from scene_grasp.scale_shape_grasp_ae.grasp_gen_tool.utils import get_o3d_centered_scale_mesh


def gen_grasps_on_obj(obj_o3dmesh, gripper_bounds_gen, num_grasp_points):
    # sample multiple
    sample_multiple = 4
    # generate point cloud
    o3dpcd_obj_gtobj = obj_o3dmesh.sample_points_poisson_disk(
        number_of_points=sample_multiple * num_grasp_points
    )

    o3dpcd_obj_gtobj.paint_uniform_color([0, 0, 1])

    # o3d.visualization.draw_geometries([obj_o3dmesh, o3dpcd_obj_gtobj])

    # PCD grasps
    grasp_map = grasp_map_utils.build_grasp_map(
        o3dpcd_obj_gtobj,
        o3dpcd_obj_gtobj,
        gripper_bounds_gen,
        num_grasp_points,
        visualize_map=False,
    )

    # Convert return values to numpy arrays instead of internal classes (for pickling
    #  purposes)
    for i in range(len(grasp_map["T_cam_grasps"])):
        if grasp_map["T_cam_grasps"][i] is not None:
            T_cam_grasp = grasp_map["T_cam_grasps"][i]
            grasp_map["T_cam_grasps"][i] = np.eye(4)
            grasp_map["T_cam_grasps"][i][:3, :3] = np.asarray(T_cam_grasp.rotation)
            grasp_map["T_cam_grasps"][i][:3, 3] = np.asarray(T_cam_grasp.translation)[
                :, 0
            ]
    for i in range(len(grasp_map["T_cam_infgrasps"])):
        if grasp_map["T_cam_infgrasps"][i] is not None:
            T_cam_infgrasp = grasp_map["T_cam_infgrasps"][i]
            grasp_map["T_cam_infgrasps"][i] = np.eye(4)
            grasp_map["T_cam_infgrasps"][i][:3, :3] = np.asarray(
                T_cam_infgrasp.rotation
            )
            grasp_map["T_cam_infgrasps"][i][:3, 3] = np.asarray(
                T_cam_infgrasp.translation
            )[:, 0]

    return grasp_map


def format_data_for_network_usage(
    grasp_data,
    grasp_data_path,
    N_POINTS,
    GRIPPER_OFFSET_BINS,
):
    grasp_pcd = np.asarray(grasp_data["pcd_pos"])
    assert (
        grasp_pcd.shape[0] == N_POINTS
    ), f"{grasp_data_path} pcd size is {grasp_pcd.shape[0]} != {N_POINTS}."

    scale = 1 / grasp_data["model_scale"]
    grasp_pcd_scaled = scale * grasp_pcd

    final_grasp_success = np.asarray(grasp_data["grasps_feasibility"]).astype(bool)
    final_grasp_width = np.asarray(grasp_data["grasps_width"])
    feas_grasp_poses = grasp_data["T_cam_grasps"]
    infeas_grasp_poses = grasp_data["T_cam_infgrasps"]
    final_grasp_width_one_hot = np.zeros((N_POINTS, 10), dtype=bool)
    final_approach_directions = np.zeros((N_POINTS, 3))
    final_baseline_directions = np.zeros((N_POINTS, 3))
    for point_ind in range(grasp_pcd_scaled.shape[0]):
        grasp_width = final_grasp_width[point_ind]
        # Grasp-width-one-hot:
        gripper_bin_index = None
        for bin_boundary_index in range(len(GRIPPER_OFFSET_BINS) - 1):
            if (
                GRIPPER_OFFSET_BINS[bin_boundary_index] <= grasp_width
                and grasp_width < GRIPPER_OFFSET_BINS[bin_boundary_index + 1]
            ):
                gripper_bin_index = bin_boundary_index
                break
        if gripper_bin_index is None:
            print(
                f"Unknown grasp width {grasp_width} obtained at {grasp_data_path}"
                f" and index {point_ind}"
            )
            gripper_bin_index = len(GRIPPER_OFFSET_BINS) - 2  # Assign the last bin
        final_grasp_width_one_hot[point_ind][gripper_bin_index] = True

        w_T_grasp = (
            feas_grasp_poses[point_ind]
            if final_grasp_success[point_ind]
            else infeas_grasp_poses[point_ind]
        )
        ads, bds = get_ads_bds_from_grasp_poses(w_T_grasp[None, ...])
        final_approach_directions[point_ind] = ads[0]
        final_baseline_directions[point_ind] = bds[0]

    data = {
        "xyz": grasp_pcd_scaled,
        "scale": grasp_data["model_scale"],
        # Grasp parameters
        "success": final_grasp_success,
        "grasp_width": final_grasp_width,
        "grasp_width_one_hot": final_grasp_width_one_hot,
        "approach_dirs": final_approach_directions,
        "baseline_dirs": final_baseline_directions,
        # Some meta-data:
        "grasp_data_path": grasp_data_path,
    }
    save_path = (
        grasp_data_path.parent
        / f"network_train_grasp_parameter_data_without_fps_at_scale_{grasp_data['model_scale']:.6f}.pkl"
    )
    with open(save_path, "wb") as fp:
        pickle.dump(data, fp)
    return save_path


def generate_grasp_helper(func_args):
    (
        model_path,
        final_max_dimension,
        save_path,
        gripper_bounds_gen,
        nocs_root,
        num_points,
        gripper_offset_bins,
    ) = func_args

    try:
        # --------- generate grasp data
        save_path.parent.mkdir(exist_ok=True, parents=True)
        # Step 1: let's scale the mesh to my scales.
        o3d_mesh, scale = get_o3d_centered_scale_mesh(model_path, final_max_dimension)
        # Step 2: Generate grasps
        grasp_map = gen_grasps_on_obj(o3d_mesh, gripper_bounds_gen, num_points)
        # * let's add some information inside grasp map to make the dict self-suff
        grasp_map["model_path"] = str(model_path.relative_to(nocs_root))
        grasp_map["model_scale"] = scale
        grasp_map["final_max_dimension"] = final_max_dimension
        # Step 3: Save this data
        with open(save_path, "wb") as fp:
            pickle.dump(grasp_map, fp)

        # --------- format the data for easier usage by the network
        formatted_data_path = format_data_for_network_usage(
            grasp_map,
            save_path,
            num_points,
            gripper_offset_bins,
        )
        print(
            f"Processed data for {model_path.relative_to(nocs_root)} saved at: "
            f"{save_path}, {formatted_data_path}",
        )
    except Exception as e:
        print(
            f"Exception occured while processing {model_path} for final max dim:"
            f" {final_max_dimension}"
        )
        print(e)


def main():
    parser = ArgumentParser("Grasp Dataset Generation")
    parser.add_argument(
        "--num_scales_per_object",
        type=int,
        default=get_grasp_dataset_num_scales(),
        help="Every object is scaled by this many random scales for dataset generation",
    )
    parser.add_argument(
        "--nocs_root",
        type=str,
        default=NOCS_ROOT,
        help="path to NOCS dataset root",
    )
    parser.add_argument("--save_root", type=str, default=None, help="path to save root")
    parser.add_argument(
        "--generate_small_sample",
        action="store_true",
        help="generate small sample for testing",
    )
    args = parser.parse_args()

    # Setup paths:
    if args.save_root is not None:
        save_root = Path(args.save_root)
        save_root.mkdir(exist_ok=True, parents=True)
    else:
        stamp = get_data_dump_stamp()
        save_root = PROJECT_DATA_ROOT / "shape_grasp_datsaet" / f"{stamp}"
    print("Save root: ", save_root)
    nocs_root = Path(args.nocs_root)

    # Constants:
    (
        _,
        shapenet_id_to_cat_id,
        cat_id_to_min_max_scale,
    ) = get_nocs_information_dicts()
    GRIPPER_BOUNDS_GEN = get_gripper_bounds()
    NUM_POINTS = get_n_points()
    GRIPPER_OFFSET_BINS = get_gripper_offset_bins()

    my_args = []
    for split_name in ["train", "val"]:
        for shapenet_id, cat_id in shapenet_id_to_cat_id.items():
            for model_path in (
                nocs_root / "obj_models" / split_name / shapenet_id
            ).rglob("model.obj"):
                final_max_dimensions = np.random.uniform(
                    cat_id_to_min_max_scale[cat_id][0],
                    cat_id_to_min_max_scale[cat_id][1],
                    args.num_scales_per_object,
                )
                for final_max_dimension in final_max_dimensions:
                    save_path = save_root / model_path.relative_to(nocs_root)
                    save_path = (
                        save_path.parent
                        / f"{save_path.stem}_grasp_data_at_max_dim_{final_max_dimension:.6f}.pkl"
                    )
                    if save_path.exists():
                        print(
                            f"\tData already exist for {save_path.relative_to(save_root)}."
                            " Continue"
                        )
                        continue
                    my_args.append(
                        (
                            model_path,
                            final_max_dimension,
                            save_path,
                            GRIPPER_BOUNDS_GEN,
                            nocs_root,
                            NUM_POINTS,
                            GRIPPER_OFFSET_BINS,
                        )
                    )
    shuffle(my_args)

    if args.generate_small_sample:
        my_args = my_args[:10]

    print("Datapoints to be generated: ", len(my_args))

    set_start_method("forkserver")  # https://github.com/isl-org/Open3D/issues/1552
    num_cpus = get_num_cpus()
    with Pool(num_cpus) as p:
        list(tqdm(p.imap(generate_grasp_helper, my_args), total=len(my_args)))


if __name__ == "__main__":
    main()
