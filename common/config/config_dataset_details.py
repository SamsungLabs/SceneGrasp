import torch
from multiprocessing import cpu_count
import numpy as np


def get_nocs_information_dicts():
    cat_name_to_cat_id = {
        "bottle": 1,  # "02876657"
        "bowl": 2,  # "02880940"
        "camera": 3,  # "02942699"
        "can": 4,  # "02946921"
        "laptop": 5,  # "03642806"
        "mug": 6,  # "03797390"
    }
    shapenet_id_to_cat_id = {
        "02876657": 1,  # bottle
        "02880940": 2,  # bowl
        "02942699": 3,  # camera
        "02946921": 4,  # can
        "03642806": 5,  # laptop
        "03797390": 6,  # mug
    }
    cat_id_to_min_max_scale = {
        1: [0.14275527758335257, 0.5729143751960505],  # bottle
        2: [0.24453030170535167, 0.5041014630929455],  # bowl
        3: [0.14551531158388295, 0.2815040624309093],  # camera
        4: [0.12200900531954012, 0.35],  # can
        5: [0.23275898043694532, 0.54838251631973],  # laptop
        6: [0.09590471283673042, 0.3522977930464345],  # mug
    }

    return cat_name_to_cat_id, shapenet_id_to_cat_id, cat_id_to_min_max_scale


def get_nocs_cat_id_to_cat_name():
    NOCS_CAT_NAME_TO_CAT_ID, _, _ = get_nocs_information_dicts()
    NOCS_CAT_ID_TO_CAT_NAME = {
        value: key for key, value in NOCS_CAT_NAME_TO_CAT_ID.items()
    }
    NOCS_CAT_ID_TO_CAT_NAME[0] = "BG"
    return NOCS_CAT_ID_TO_CAT_NAME


def get_gripper_offset_bins():
    GRIPPER_OFFSET_BINS = np.array([
        0,
        0.00794435329,
        0.0158887021,
        0.0238330509,
        0.0317773996,
        0.0397217484,
        0.0476660972,
        0.055610446,
        0.0635547948,
        0.0714991435,
        0.08,
    ])
    return GRIPPER_OFFSET_BINS


def get_grasp_dataset_num_scales():
    GRASP_DATASET_NUM_SCALES = 10
    return GRASP_DATASET_NUM_SCALES


def convert_gwoh_to_gwv_batch(batch_gw_oh, gripper_offset_bins):
    """
    batch_gw_oh: (B, N, 10)
    """
    if torch.is_tensor(batch_gw_oh):
        bin_indices = torch.argmax(batch_gw_oh, axis=-1)
    else:
        bin_indices = np.argmax(batch_gw_oh, axis=-1)
    batch_gw_vals = (
        gripper_offset_bins[bin_indices] + gripper_offset_bins[bin_indices + 1]
    ) / 2
    return batch_gw_vals


def get_n_points():
    N_POINTS = 2048
    return N_POINTS


def get_emb_dim():
    EMB_DIM = 128
    return EMB_DIM


def get_min_chamfer_loss_for_grasp_losses():
    MIN_CHAMFER_LOSS_FOR_GRASP_LOSSES = 2 * 0.0014
    return MIN_CHAMFER_LOSS_FOR_GRASP_LOSSES


def get_grasp_loss_topk():
    grasp_loss_topk = 512
    return grasp_loss_topk


def get_weight_shape_loss():
    weight_shape_loss = 0.8
    return weight_shape_loss


def get_shape_grasp_ae_logs_prefix():
    PREFIX = "scale_shape_grasp_ae"
    return PREFIX


def get_shape_grasp_ae_initial_lr():
    INITIAL_LR = 0.0001
    return INITIAL_LR


def get_gripper_bounds():
    MIN_GRASP_WIDTH = 25  # mm
    MAX_GRASP_WIDTH = 80
    FINGER_THICKNESS = 25
    FINGER_LENGTH = 45
    gripper_bounds_gen = [
        1e-3 * np.array([MIN_GRASP_WIDTH, FINGER_THICKNESS, FINGER_LENGTH]),
        1e-3 * np.array([MAX_GRASP_WIDTH, FINGER_THICKNESS, FINGER_LENGTH]),
    ]
    return gripper_bounds_gen


def get_our_gripper_origin_in_cgn_frame():
    cgn_T_our = np.eye(4)
    cgn_T_our[2, 3] = 0.1
    return cgn_T_our


def get_grasp_cover_threshold():
    return 0.02


def get_grasp_success_threshold():
    return 0.5


def get_gripper_width_tolerance():
    return 2e-2


def get_point_bd_distance():
    """
    We dont' want to place gripper pad exactly at the predicted / observed point.
    This is the distnace we place the gripper pad from the point in the direction
    opposite to the baseline direction
    """
    return 5e-3


def get_points_in_grasp_theshold():
    return 5


def get_top_k_grasp_number():
    return 15


def get_max_depth_threshold():
    # TODO: This should be tuned based on training data
    return 3e3


def get_num_cpus():
    num_cpus = max(1, cpu_count() - 2)
    return num_cpus
